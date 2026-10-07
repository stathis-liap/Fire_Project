#include "tactics.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace rapid {

namespace {

std::string fmt_m(double m) {
    std::ostringstream o;
    if (m >= 1000) {
        o.precision(1);
        o << std::fixed << m / 1000.0 << " km";
    } else {
        o << int(std::round(m / 10.0) * 10) << " m";
    }
    return o.str();
}

std::string fmt_flame(double f) {
    std::ostringstream o;
    o.precision(f < 10 ? 1 : 0);
    o << std::fixed << f << " m";
    return o.str();
}

std::string clock_token(double t0_epoch, double t_min) {
    return "{t:" + std::to_string(static_cast<long long>(t0_epoch + t_min * 60.0)) + "}";
}

}  // namespace

std::vector<int> trace_path(const RunOutput& run, int cell, int max_len) {
    std::vector<int> path;
    for (int k = cell; k >= 0 && int(path.size()) < max_len; k = run.pred[k]) path.push_back(k);
    return path;
}

double spread_bearing(const Grid& g, const RunOutput& run, int cell, int lookback) {
    int k = cell;
    for (int i = 0; i < lookback && run.pred[k] >= 0; ++i) k = run.pred[k];
    if (k == cell) return -1;
    return bearing_deg(g.center_of(k / g.cols, k % g.cols), g.center_of(cell / g.cols, cell % g.cols));
}

namespace {

int snap_to_road(const Landscape& L, int cell, double radius_m) {
    const Grid& g = L.grid;
    int r0 = cell / g.cols, c0 = cell % g.cols, rad = int(std::ceil(radius_m / g.cell_m));
    int best = -1;
    double bd = 1e18;
    for (int dr = -rad; dr <= rad; ++dr)
        for (int dc = -rad; dc <= rad; ++dc) {
            if (!g.inside(r0 + dr, c0 + dc)) continue;
            int k = g.idx(r0 + dr, c0 + dc);
            double d = std::hypot(dr, dc);
            if (L.road[k] && d < bd && d * g.cell_m <= radius_m) { bd = d; best = k; }
        }
    return best;
}

LatLon cell_ll(const Grid& g, int k) { return g.center_of(k / g.cols, k % g.cols); }

}  // namespace

std::vector<Recommendation> plan_tactics(const TacticsInput& in, const TacticsTiming& tm) {
    const Landscape& L = *in.L;
    const Grid& g = L.grid;
    const RunOutput& B = *in.best;
    const auto& A = B.arrival;
    const auto& ft = fuel_table();
    std::vector<Recommendation> recs;
    int next_id = 1;
    auto new_id = [&](const char* p) { return std::string(p) + std::to_string(next_id++); };
    auto clock = [&](double t) { return clock_token(in.t0_epoch, t); };
    auto burnable = [&](int k) { return ft[L.fuel[k]].burnable(); };

    // ── 0. Warnings for places the fire may reach before any crew can act ──
    for (const auto& th : in.threats) {
        if (th.prob < 0.4 || th.eta - in.t_now > 120) continue;
        Recommendation r;
        r.id = new_id("e");
        r.type = "evacuate";
        r.simulated = false;
        r.target_id = th.dest_id;
        r.target_name = th.name;
        r.title = "Warn / evacuate " + th.name;
        r.detail = "Fire may reach " + th.name + " around " + clock(th.eta) + " (" +
                   std::to_string(int(std::round(th.prob * 100))) + "% of forecasts). Start warning people now.";
        r.act_before = in.t_now;
        r.iv.pts = {cell_ll(g, th.cell)};
        recs.push_back(r);
    }

    // ── 1. Firebreaks / dead zones across the path to threatened places ────
    std::vector<uint8_t> covered(g.size(), 0);  // cells already protected by a break
    int n_breaks = 0;
    std::vector<Threat> break_targets = in.threats;
    if (in.head_cell >= 0) break_targets.push_back({"", "", in.head_cell, A[in.head_cell], 1.0});  // across the head
    for (const auto& th : break_targets) {
        if (n_breaks >= tm.max_breaks || th.prob < 0.25) break;
        std::vector<int> path = trace_path(B, th.cell);  // path[0] = place, back towards the fire
        if (path.size() < 4) continue;
        bool already = false;
        for (int k : path) already |= covered[k] != 0;
        if (already) continue;

        // Walk from the fire towards the place; take the first point the
        // fire reaches late enough for the line to be finished in time.
        double est_len = 1500;
        int q = -1;
        std::vector<LatLon> line;
        double length = 0, build_min = 0;
        bool anchored_l = false, anchored_r = false;
        for (int attempt = 0; attempt < 4; ++attempt) {
            build_min = est_len / tm.dozer_m_per_h * 60.0;
            double need = in.t_now + std::max(20.0, build_min) + tm.safety;
            q = -1;
            for (int i = int(path.size()) - 1; i >= 0; --i)
                if (A[path[i]] >= need) { q = path[i]; break; }
            if (q < 0) break;
            double brg = spread_bearing(g, B, q);
            if (brg < 0) brg = in.head_bearing;
            LatLon qc = cell_ll(g, q);
            std::vector<LatLon> sides[2];
            bool anch[2] = {false, false};
            for (int s = 0; s < 2; ++s) {
                double side = brg + (s == 0 ? -90.0 : 90.0);
                int extra = -1;
                for (double d = g.cell_m; d <= 3000; d += g.cell_m) {
                    LatLon p = offset_m(qc, side, d);
                    int r, c;
                    if (!g.locate(p, r, c)) break;
                    int k = g.idx(r, c);
                    sides[s].push_back(p);
                    if (!burnable(k)) { anch[s] = true; break; }          // anchored on a barrier
                    if (extra < 0 && (A[k] >= kInf || A[k] > A[q] + 90)) extra = 3;  // beyond the threatening front
                    if (extra >= 0 && --extra <= 0) break;
                }
            }
            line.clear();
            for (auto it = sides[0].rbegin(); it != sides[0].rend(); ++it) line.push_back(*it);
            line.push_back(qc);
            for (auto& p : sides[1]) line.push_back(p);
            length = 0;
            for (size_t i = 1; i < line.size(); ++i) length += haversine_m(line[i - 1], line[i]);
            anchored_l = anch[0];
            anchored_r = anch[1];
            if (std::abs(length - est_len) < 150) break;
            est_len = std::max(200.0, length);
        }
        if (q < 0 || line.size() < 3) continue;
        build_min = length / tm.dozer_m_per_h * 60.0;
        double flame = B.flame_m[q];
        double width = std::clamp(1.5 * flame, 10.0, 40.0);

        Recommendation r;
        r.id = new_id("b");
        r.type = "firebreak";
        r.resource = "dozers";
        r.target_id = th.dest_id;
        r.target_name = th.name;
        r.iv.kind = IvKind::Firebreak;
        r.iv.id = r.id;
        r.iv.pts = {line.front(), cell_ll(g, q), line.back()};
        r.iv.width_m = width;
        r.iv.t_on = in.t_now + std::max(20.0, build_min);
        r.breach_p = std::clamp(0.05 + 0.07 * std::max(0.0, flame - 1.5) + 0.015 * std::max(0.0, in.wind_kmh - 20), 0.05, 0.85);
        r.act_before = float(A[q] - tm.safety - build_min);
        r.title = th.name.empty() ? std::string("Cut a firebreak across the head of the fire")
                                  : "Cut a firebreak to protect " + th.name;
        std::string anchors = anchored_l && anchored_r ? "Both ends tie into ground that does not burn."
                              : anchored_l || anchored_r ? "One end ties into ground that does not burn."
                                                         : "Ends are not anchored — watch the flanks.";
        r.detail = fmt_m(length) + " long, at least " + fmt_m(width) + " wide (bulldozer + crew, ~" +
                   std::to_string(int(std::round(build_min))) + " min). " + anchors + " Start before " +
                   clock(r.act_before) + "; fire could reach the line from " + clock(A[q]) + ".";
        if (flame > 8) r.detail += " High flames (" + fmt_flame(flame) + "): expect spot fires across the line.";
        for (const auto& p : line) {
            int rr, cc;
            if (g.locate(p, rr, cc))
                for (int k : trace_path(B, g.idx(rr, cc), 1)) covered[k] = 1;
        }
        for (int k : path)
            if (A[k] >= A[q]) covered[k] = 1;
        recs.push_back(r);
        ++n_breaks;
    }

    // ── 2. Aircraft: sustained drops on the most active parts of the front ──
    {
        float w0 = float(in.t_now + tm.air_eta), w1 = float(in.t_now + tm.air_eta + 25);
        struct Cand { int cell; std::string target_id, target_name; };
        std::vector<Cand> picks;
        auto far_enough = [&](int k) {
            for (auto& p : picks)
                if (haversine_m(cell_ll(g, p.cell), cell_ll(g, k)) < 700) return false;
            return true;
        };
        // Where the front heads for each threatened place (protective drops).
        for (const auto& th : in.threats) {
            if (int(picks.size()) >= tm.max_drops) break;
            for (int k : trace_path(B, th.cell))
                if (A[k] >= w0 && A[k] <= w1 + 30) {
                    if (far_enough(k)) picks.push_back({k, th.dest_id, th.name});
                    break;
                }
        }
        // The most intense parts of the front when aircraft can arrive.
        std::vector<int> front;
        for (size_t k = 0; k < g.size(); ++k)
            if (A[k] >= w0 && A[k] <= w1 && B.flame_m[k] > 1.2) front.push_back(int(k));
        std::sort(front.begin(), front.end(), [&](int x, int y) { return B.ros[x] > B.ros[y]; });
        for (int k : front) {
            if (int(picks.size()) >= tm.max_drops) break;
            if (far_enough(k)) picks.push_back({k, "", ""});
        }
        for (const auto& pk : picks) {
            const int k = pk.cell;
            double brg = spread_bearing(g, B, k);
            if (brg < 0) brg = in.head_bearing;
            double flame = B.flame_m[k];
            Recommendation r;
            r.id = new_id("a");
            r.type = "air_drop";
            r.resource = "aircraft";
            r.iv.kind = IvKind::AirDrop;
            r.iv.id = r.id;
            r.iv.pts = {cell_ll(g, k)};
            r.iv.bearing = std::fmod(brg + 90.0, 360.0);
            r.iv.length_m = 400;
            r.iv.width_m = 40;
            r.iv.t_on = in.t_now + tm.air_eta;
            r.iv.t_off = r.iv.t_on + tm.aircraft_hold;
            r.act_before = in.t_now;
            r.target_id = pk.target_id;
            r.target_name = pk.target_name;
            bool is_head = k == (front.empty() ? -1 : front.front());
            r.title = !r.target_name.empty() ? "Aircraft: drops to shield " + r.target_name
                      : is_head            ? "Aircraft: drops on the head of the fire"
                                           : "Aircraft: drops on the " + compass_name(bearing_deg(in.origin, cell_ll(g, k))) + " front";
            r.detail = "~400 m drop line across the fire's path, first drop at " + clock(r.iv.t_on) +
                       ", then repeat every ~45 min for " + std::to_string(int(tm.aircraft_hold / 60)) +
                       " h. Flames there ~" + fmt_flame(flame) + ".";
            if (flame > 10) r.detail += " Extreme fire: drops will slow it but not stop it.";
            recs.push_back(r);
        }
    }

    // ── 3. Fire trucks between the fire and threatened places ──────────────
    int n_trucks = 0;
    for (const auto& th : in.threats) {
        if (n_trucks >= tm.max_trucks - 2 || th.prob < 0.25) break;
        if (th.eta < in.t_now + tm.truck_eta) continue;  // the warning card covers this case
        std::vector<int> path = trace_path(B, th.cell);
        LatLon dpos = cell_ll(g, th.cell);
        int t = th.cell;
        for (int k : path) {
            t = k;
            if (haversine_m(cell_ll(g, k), dpos) >= 250) break;
        }
        bool in_place = A[t] < in.t_now + tm.truck_eta + 10;
        if (in_place) t = th.cell;
        int road = snap_to_road(L, t, 400);
        bool on_road = road >= 0;
        if (on_road) t = road;
        double flame = B.flame_m[t] > 0 ? B.flame_m[t] : B.flame_m[th.cell];
        Recommendation r;
        r.id = new_id("t");
        r.type = "truck";
        r.resource = "trucks";
        r.target_id = th.dest_id;
        r.target_name = th.name;
        r.iv.kind = IvKind::Truck;
        r.iv.id = r.id;
        r.iv.pts = {cell_ll(g, t)};
        r.iv.radius_m = 150;
        r.iv.t_on = in.t_now + tm.truck_eta;
        r.iv.t_off = r.iv.t_on + 360;
        r.act_before = float(std::max<double>(in.t_now, th.eta - tm.truck_eta - 20));
        r.title = "Fire truck to protect " + th.name;
        r.detail = std::string(in_place ? "Defend the edge of " + th.name + ". " : "Hold position between the fire and " + th.name + ". ") +
                   (on_road ? "Position is on a road. " : "No road found nearby — check access. ") +
                   "Fire could arrive from ~" + clock(th.eta) + ".";
        if (flame > 3.4)
            r.detail += " Flames ~" + fmt_flame(flame) + " — too intense for direct attack: protect buildings and people.";
        recs.push_back(r);
        ++n_trucks;
    }
    {
        // Flank attack from the rear, the safe side (useful whether or not
        // places are threatened; the planner decides).
        for (double side : {-90.0, 90.0}) {
            double want = std::fmod(in.head_bearing + side + 360.0, 360.0);
            int best = -1;
            double best_score = 1e18;
            for (size_t k = 0; k < g.size(); ++k) {
                if (A[k] < in.t_now + 15 || A[k] > in.t_now + 60) continue;
                double brg = bearing_deg(in.origin, cell_ll(g, int(k)));
                double off = std::abs(std::remainder(brg - want, 360.0));
                if (off > 35) continue;
                double score = B.flame_m[k] + off * 0.02;
                if (score < best_score) { best_score = score; best = int(k); }
            }
            if (best < 0) continue;
            int road = snap_to_road(L, best, 300);
            int t = road >= 0 ? road : best;
            Recommendation r;
            r.id = new_id("t");
            r.type = "truck";
            r.resource = "trucks";
            r.iv.kind = IvKind::Truck;
            r.iv.id = r.id;
            r.iv.pts = {cell_ll(g, t)};
            r.iv.radius_m = 150;
            r.iv.t_on = in.t_now + tm.truck_eta;
            r.iv.t_off = r.iv.t_on + 360;
            r.act_before = float(in.t_now + 15);
            r.title = std::string("Fire truck on the ") + (side < 0 ? "left" : "right") + " flank";
            r.detail = "Attack the " + compass_name(want) + " flank, working from the burned side. Flames there ~" +
                       fmt_flame(B.flame_m[best]) + "." + (road >= 0 ? " Position is on a road." : "");
            recs.push_back(r);
        }
    }

    std::stable_sort(recs.begin(), recs.end(), [](const Recommendation& a, const Recommendation& b) {
        if (a.type == "evacuate" || b.type == "evacuate") return a.type == "evacuate" && b.type != "evacuate";
        return a.act_before < b.act_before;
    });
    for (size_t i = 0; i < recs.size(); ++i) recs[i].order = int(i) + 1;
    return recs;
}

}  // namespace rapid
