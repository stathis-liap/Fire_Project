#include "planner.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <functional>
#include <thread>

namespace rapid {

namespace {

const char* kResName[3] = {"aircraft", "fire truck", "bulldozer"};
const char* kResPlural[3] = {"aircraft", "fire trucks", "bulldozers"};
// A village (weight 4) fully protected is worth as much as 200 ha of land.
constexpr double kHaValue = 0.02;

void parallel_for(size_t n, const std::function<void(size_t)>& fn) {
    std::atomic<size_t> next{0};
    unsigned nt = std::max(1u, std::min<unsigned>(std::thread::hardware_concurrency(), unsigned(n)));
    std::vector<std::thread> pool;
    for (unsigned k = 0; k < nt; ++k)
        pool.emplace_back([&] {
            for (size_t i; (i = next++) < n;) fn(i);
        });
    for (auto& t : pool) t.join();
}

// Deterministic per (scenario, action) draw, so an action breaches in the
// same scenarios whatever else is in the plan.
double unit_hash(const std::string& id, size_t m) {
    size_t h = std::hash<std::string>{}(id) ^ (m * 0x9E3779B97F4A7C15ull);
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdull;
    h ^= h >> 33;
    return double(h % 1000003) / 1000003.0;
}

struct Eval {
    double value = 0;
    std::vector<double> member_value;
    std::vector<double> delay;     // per target: mean minutes later (threatened members)
    std::vector<double> kept_out;  // per target: share of threatened members kept out
    double saved_ha = 0;
};

std::string fmt_minutes(double m) {
    int r = int(std::round(m / 5.0) * 5);
    if (r < 60) return std::to_string(std::max(r, 5)) + " min";
    return std::to_string(r / 60) + " h" + (r % 60 ? " " + std::to_string(r % 60) + " min" : "");
}

}  // namespace

double place_weight(const std::string& kind) {
    if (kind == "city") return 10;
    if (kind == "town") return 6;
    if (kind == "hospital" || kind == "nursing_home" || kind == "kindergarten" || kind == "school" || kind == "user") return 5;
    if (kind == "village" || kind == "clinic") return 4;
    if (kind == "suburb") return 3;
    if (kind == "hamlet" || kind == "fire_station") return 2;
    if (kind == "road") return 0.5;
    if (kind == "river") return 0.2;
    return 1;
}

int resource_index(const std::string& r) {
    return r == "aircraft" ? 0 : r == "trucks" ? 1 : r == "dozers" ? 2 : -1;
}

PlannerOutput choose_plan(const PlannerInput& in, std::vector<Recommendation>& cands, const Resources& budget) {
    auto tic = std::chrono::steady_clock::now();
    PlannerOutput out;
    const size_t M = in.members.size();
    const auto& base = *in.base_arrival;
    std::atomic<int> sims{0};

    // Base (no action) per member: arrival at each target and burned area.
    const size_t T = in.targets.size();
    std::vector<std::vector<float>> b0(M, std::vector<float>(T, kInf));
    std::vector<double> area0(M, 0);
    for (size_t m = 0; m < M; ++m) {
        for (size_t t = 0; t < T; ++t)
            for (int c : in.targets[t].cells) b0[m][t] = std::min(b0[m][t], base[m][c]);
        size_t n = 0;
        for (float a : base[m]) n += a <= in.t_end;
        area0[m] = n * in.cell_ha;
    }

    // Value of a set of interventions, measured on all scenarios.
    auto evaluate = [&](const std::vector<const Recommendation*>& set) {
        std::vector<Intervention> ivs = in.fixed;
        std::vector<double> bp = in.fixed_breach;
        std::vector<std::string> ids;
        for (auto& f : in.fixed) ids.push_back(f.id);
        for (auto* r : set) { ivs.push_back(r->iv); bp.push_back(r->breach_p); ids.push_back(r->id); }
        InterventionRaster raster;
        raster.build(in.solver->land().grid, ivs);
        for (auto& l : raster.layers) l.breach_p = float(bp[l.src]);

        Eval e;
        e.member_value.assign(M, 0);
        std::vector<std::vector<float>> b1(M, std::vector<float>(T, kInf));
        std::vector<double> area1(M, 0);
        parallel_for(M, [&](size_t m) {
            RunConfig c = in.cfg;
            c.p = in.members[m].p;
            c.iv = &raster;
            c.detail = false;
            c.breached = 0;
            for (size_t l = 0; l < raster.layers.size(); ++l)
                if (raster.layers[l].kind == IvKind::Firebreak && unit_hash(ids[raster.layers[l].src], m) < raster.layers[l].breach_p)
                    c.breached |= 1u << l;
            RunOutput o;
            in.solver->run(c, o);
            ++sims;
            for (size_t t = 0; t < T; ++t)
                for (int cell : in.targets[t].cells) b1[m][t] = std::min(b1[m][t], o.arrival[cell]);
            size_t n = 0;
            for (float a : o.arrival) n += a <= in.t_end;
            area1[m] = n * in.cell_ha;
        });
        e.delay.assign(T, 0);
        e.kept_out.assign(T, 0);
        std::vector<int> threatened(T, 0);
        for (size_t m = 0; m < M; ++m) {
            double v = 0;
            for (size_t t = 0; t < T; ++t) {
                if (b0[m][t] > in.t_end) continue;
                ++threatened[t];
                bool out_ = b1[m][t] > in.t_end;
                double prot = out_ ? 1.0 : std::clamp((b1[m][t] - b0[m][t]) / 120.0, 0.0, 1.0);
                v += in.targets[t].weight * prot;
                e.kept_out[t] += out_;
                e.delay[t] += std::min<double>(b1[m][t], in.t_end) - b0[m][t];
            }
            double saved = std::max(0.0, area0[m] - area1[m]);
            v += kHaValue * saved;
            e.member_value[m] = v;
            e.value += v / M;
            e.saved_ha += saved / M;
        }
        for (size_t t = 0; t < T; ++t)
            if (threatened[t]) { e.delay[t] /= threatened[t]; e.kept_out[t] /= threatened[t]; }
        return e;
    };

    // Plain-language difference between two evaluations.
    auto describe = [&](const Eval& before, const Eval& after) -> std::string {
        int best = -1;
        double best_score = 0;
        for (size_t t = 0; t < T; ++t) {
            double s = in.targets[t].weight * ((after.kept_out[t] - before.kept_out[t]) + (after.delay[t] - before.delay[t]) / 120.0);
            if (s > best_score + 1e-9) { best_score = s; best = int(t); }
        }
        if (best >= 0) {
            const auto& name = in.targets[best].name;
            double ko = after.kept_out[best] - before.kept_out[best];
            if (ko >= 0.3) return "keeps the fire out of " + name + " in " + std::to_string(int(std::round(after.kept_out[best] * 100))) + "% of scenarios";
            double d = after.delay[best] - before.delay[best];
            if (d >= 5) return "fire reaches " + name + " ~" + fmt_minutes(d) + " later";
        }
        double ha = after.saved_ha - before.saved_ha;
        if (ha >= 1) return "saves ~" + std::to_string(int(std::round(ha))) + " ha";
        return "";
    };

    // ── Budget left after the user's own actions ──
    int left[3] = {budget.aircraft, budget.trucks, budget.dozers};
    for (int k = 0; k < 3; ++k) out.available[k] = left[k];
    for (auto& f : in.fixed) {
        int k = f.kind == IvKind::AirDrop ? 0 : f.kind == IvKind::Truck ? 1 : 2;
        --left[k];
        ++out.used[k];
    }

    // ── Start from what the user forced on ──
    std::vector<const Recommendation*> chosen;
    for (auto& c : cands)
        if (c.forced && c.resource != "") {
            chosen.push_back(&c);
            int k = resource_index(c.resource);
            --left[k];
            ++out.used[k];
        }
    Eval cur = evaluate(chosen);

    std::vector<size_t> pool;
    for (size_t i = 0; i < cands.size(); ++i)
        if (!cands[i].forced && cands[i].enabled && resource_index(cands[i].resource) >= 0) pool.push_back(i);
    std::vector<double> stale(cands.size(), 1e18);
    std::vector<Eval> last(cands.size()), last_ref(cands.size());
    std::vector<uint8_t> taken(cands.size(), 0);
    const double eps = 0.03;  // ≈ 1.5 ha: below this an action is not worth a resource
    const size_t batch = std::max<size_t>(3, std::thread::hardware_concurrency() / std::max<size_t>(1, M / 2));

    auto fresh_gains = [&](const std::vector<size_t>& idx) {
        std::vector<Eval> evs(idx.size());
        std::vector<std::thread> th;
        for (size_t j = 0; j < idx.size(); ++j)
            th.emplace_back([&, j] {  // candidates in parallel; scenarios in parallel inside
                auto set = chosen;
                set.push_back(&cands[idx[j]]);
                evs[j] = evaluate(set);
            });
        for (auto& t : th) t.join();
        for (size_t j = 0; j < idx.size(); ++j) {
            stale[idx[j]] = evs[j].value - cur.value;
            last[idx[j]] = std::move(evs[j]);
            last_ref[idx[j]] = cur;
        }
    };

    while (true) {
        std::vector<size_t> elig;
        for (size_t i : pool)
            if (!taken[i] && left[resource_index(cands[i].resource)] > 0) elig.push_back(i);
        if (elig.empty()) break;
        std::sort(elig.begin(), elig.end(), [&](size_t a, size_t b) { return stale[a] > stale[b]; });
        // Re-simulate the most promising until the best fresh gain beats every stale bound.
        std::vector<uint8_t> fresh(cands.size(), 0);
        size_t pos = 0;
        while (pos < elig.size()) {
            std::vector<size_t> b;
            for (; pos < elig.size() && b.size() < batch; ++pos) b.push_back(elig[pos]);
            fresh_gains(b);
            for (size_t i : b) fresh[i] = 1;
            double best_fresh = -1e18, best_stale = -1e18;
            for (size_t i : elig) (fresh[i] ? best_fresh : best_stale) = std::max(fresh[i] ? best_fresh : best_stale, stale[i]);
            if (best_fresh >= best_stale) break;
        }
        size_t best = elig[0];
        for (size_t i : elig)
            if (fresh[i] && (!fresh[best] || stale[i] > stale[best])) best = i;
        if (stale[best] < eps) break;  // nothing left that helps

        Recommendation& r = cands[best];
        r.selected = true;
        r.gain = stale[best];
        r.effect = describe(cur, last[best]);
        if (!r.effect.empty()) r.effect = char(std::toupper(r.effect[0])) + r.effect.substr(1) + ".";
        int good = 0;
        for (size_t m = 0; m < M; ++m) good += last[best].member_value[m] - cur.member_value[m] >= eps;
        r.confidence = 100.0 * good / std::max<size_t>(1, M);
        taken[best] = 1;
        chosen.push_back(&r);
        int k = resource_index(r.resource);
        --left[k];
        ++out.used[k];
        cur = std::move(last[best]);
        // Remaining gains stay as upper bounds (lazy greedy).
    }

    // Effects of forced actions (measured without each one).
    for (auto* r : chosen)
        if (r->forced) {
            std::vector<const Recommendation*> without;
            for (auto* x : chosen)
                if (x != r) without.push_back(x);
            Eval w = evaluate(without);
            auto& rr = const_cast<Recommendation&>(*r);
            rr.gain = cur.value - w.value;
            rr.effect = describe(w, cur);
            if (!rr.effect.empty()) rr.effect[0] = char(std::toupper(rr.effect[0]));
            int good = 0;
            for (size_t m = 0; m < M; ++m) good += cur.member_value[m] - w.member_value[m] >= eps;
            rr.confidence = 100.0 * good / std::max<size_t>(1, M);
        }

    // ── What one more unit of an exhausted resource would buy ──
    for (int k = 0; k < 3; ++k) {
        if (left[k] > 0) {
            if (out.available[k] > out.used[k])
                out.notes.push_back(std::to_string(out.available[k] - out.used[k]) + " " +
                                    (out.available[k] - out.used[k] > 1 ? kResPlural[k] : kResName[k]) +
                                    " not needed in this plan — keep in reserve.");
            continue;
        }
        size_t best = cands.size();
        for (size_t i : pool)
            if (!taken[i] && resource_index(cands[i].resource) == k && (best == cands.size() || stale[i] > stale[best])) best = i;
        if (best == cands.size()) continue;
        fresh_gains({best});
        if (stale[best] < eps) continue;
        Recommendation& r = cands[best];
        r.gain = stale[best];
        r.effect = describe(cur, last[best]);
        std::string what = r.effect.empty() ? "would help further" : r.effect;
        out.notes.push_back("1 more " + std::string(kResName[k]) + ": " + what + " (" + r.title + ").");
        if (!r.effect.empty()) r.effect = "With 1 more " + std::string(kResName[k]) + ": " + r.effect;
    }
    // Estimated value of the options that were not used (for "other options").
    for (size_t i : pool)
        if (!taken[i] && cands[i].effect.empty() && stale[i] < 1e17) {
            cands[i].gain = std::max(0.0, stale[i]);
            std::string e = describe(last_ref[i], last[i]);
            cands[i].effect = e.empty() ? "Little effect." : "On its own: " + e + ".";
        }
    for (int k = 0; k < 3; ++k) out.used[k] = std::max(0, out.used[k]);
    out.simulations = sims.load();
    out.runtime_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tic).count();
    return out;
}

}  // namespace rapid
