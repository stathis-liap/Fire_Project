#include "calibrate.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <thread>

namespace rapid {

namespace {

constexpr int D = 4;
using Vec = std::array<double, D>;

// Search space is normalised so one unit is a "large but plausible" error:
// spread rate ×e^0.7, wind speed ×e^0.4, wind direction 25°, moisture 2.5 %.
constexpr double kUnit[D] = {0.7, 0.4, 25.0, 0.025};

SpreadParams decode(const Vec& x, const SpreadParams& base) {
    SpreadParams p = base;
    p.ros_mult = std::clamp(base.ros_mult * std::exp(kUnit[0] * x[0]), 0.15, 6.0);
    p.wind_mult = std::clamp(base.wind_mult * std::exp(kUnit[1] * x[1]), 0.3, 3.0);
    p.wind_dir_offset = std::clamp(base.wind_dir_offset + kUnit[2] * x[2], -60.0, 60.0);
    p.moisture_offset = std::clamp(base.moisture_offset + kUnit[3] * x[3], -0.08, 0.10);
    return p;
}

// Runs f(i) for i in [0, n) on up to `threads` threads.
template <class F>
void parallel_for(size_t n, unsigned threads, F f) {
    unsigned nt = std::max(1u, std::min<unsigned>(threads, unsigned(n)));
    std::atomic<size_t> next{0};
    std::vector<std::thread> pool;
    for (unsigned k = 0; k < nt; ++k)
        pool.emplace_back([&] {
            for (size_t i; (i = next++) < n;) f(i);
        });
    for (auto& t : pool) t.join();
}

unsigned hw_threads() { return std::max(1u, std::thread::hardware_concurrency()); }

// Radical inverse in base b: the Halton low-discrepancy sequence.
double halton(int i, int b) {
    double f = 1, r = 0;
    for (; i > 0; i /= b) {
        f /= b;
        r += f * (i % b);
    }
    return r;
}

// Interval with everything the score needs precomputed once.
struct Prepared {
    const CalibrationInterval* iv;
    float t_start = 0, t_run = 0, tau = 20;
    std::vector<uint8_t> prior;  // burned at the start of the interval
    std::vector<int> edge;       // observed edge cells that are new growth
    size_t obs_growth = 0;
};

Prepared prepare(const Grid& g, const CalibrationInterval& iv) {
    Prepared P;
    P.iv = &iv;
    P.prior.assign(g.size(), 0);
    for (auto [c, t] : iv.sources)
        if (c >= 0 && size_t(c) < g.size()) {
            P.prior[c] = 1;
            P.t_start = std::max(P.t_start, t);
        }
    const float dt = std::max(1.f, iv.t_obs - P.t_start);
    P.tau = std::max(20.f, 0.5f * dt);
    // Run past the observation so a too-slow model still shows *how* late it is.
    P.t_run = iv.t_obs + std::clamp(0.5f * dt, 15.f, 120.f);
    if (!iv.observed.empty()) {
        for (int r = 0; r < g.rows; ++r)
            for (int c = 0; c < g.cols; ++c) {
                int k = g.idx(r, c);
                if (!iv.observed[k] || P.prior[k]) continue;
                ++P.obs_growth;
                bool edge = r == 0 || c == 0 || r == g.rows - 1 || c == g.cols - 1 || !iv.observed[k - 1] ||
                            !iv.observed[k + 1] || !iv.observed[k - g.cols] || !iv.observed[k + g.cols];
                if (edge) P.edge.push_back(k);
            }
    }
    return P;
}

// Chamfer (3-4) distance in cells from every cell to the nearest seed cell.
void distance_from(const Grid& g, const std::vector<uint8_t>& seed, std::vector<float>& d) {
    const int R = g.rows, C = g.cols;
    d.assign(g.size(), 1e9f);
    for (size_t i = 0; i < g.size(); ++i)
        if (seed[i]) d[i] = 0;
    auto relax = [&](int r, int c, int rr, int cc, float w) {
        if (rr < 0 || cc < 0 || rr >= R || cc >= C) return;
        float v = d[g.idx(rr, cc)] + w;
        if (v < d[g.idx(r, c)]) d[g.idx(r, c)] = v;
    };
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < C; ++c) {
            relax(r, c, r, c - 1, 1.f);
            relax(r, c, r - 1, c - 1, 1.4142f);
            relax(r, c, r - 1, c, 1.f);
            relax(r, c, r - 1, c + 1, 1.4142f);
        }
    for (int r = R - 1; r >= 0; --r)
        for (int c = C - 1; c >= 0; --c) {
            relax(r, c, r, c + 1, 1.f);
            relax(r, c, r + 1, c + 1, 1.4142f);
            relax(r, c, r + 1, c, 1.f);
            relax(r, c, r + 1, c - 1, 1.4142f);
        }
}

// Mean distance (m) between the simulated fire edge at t_obs and the observed edge.
double front_error_m(const Grid& g, const RunOutput& run, const CalibrationInterval& iv) {
    if (iv.observed.empty()) return -1;
    const int R = g.rows, C = g.cols;
    auto edge_of = [&](auto in) {
        std::vector<uint8_t> e(g.size(), 0);
        size_t n = 0;
        for (int r = 0; r < R; ++r)
            for (int c = 0; c < C; ++c) {
                int k = g.idx(r, c);
                if (!in(k)) continue;
                bool b = r == 0 || c == 0 || r == R - 1 || c == C - 1 || !in(k - 1) || !in(k + 1) || !in(k - C) || !in(k + C);
                if (b) { e[k] = 1; ++n; }
            }
        return std::make_pair(e, n);
    };
    auto [pe, np] = edge_of([&](int k) { return run.arrival[k] <= iv.t_obs; });
    auto [oe, no] = edge_of([&](int k) { return iv.observed[k] != 0; });
    if (!np || !no) return -1;
    std::vector<float> dp, dobs;
    distance_from(g, pe, dp);
    distance_from(g, oe, dobs);
    double a = 0, b = 0;
    for (size_t k = 0; k < g.size(); ++k) {
        if (oe[k]) a += dp[k];
        if (pe[k]) b += dobs[k];
    }
    return 0.5 * (a / no + b / np) * g.cell_m;
}

FitDetail score_prepared(const Grid& g, const RunOutput& run, const Prepared& P) {
    const CalibrationInterval& iv = *P.iv;
    FitDetail f;
    const bool has_obs = !iv.observed.empty();
    if (has_obs) {
        size_t inter = 0, uni = 0;
        for (size_t i = 0; i < run.arrival.size(); ++i) {
            if (P.prior[i]) continue;
            bool p = run.arrival[i] <= iv.t_obs, o = iv.observed[i] != 0;
            inter += p && o;
            uni += p || o;
        }
        f.growth_iou = uni ? double(inter) / double(uni) : 1.0;
        if (!P.edge.empty()) {
            double err = 0;
            for (int c : P.edge) {
                float a = run.arrival[c];
                err += a >= kInf ? 1.0 : std::min(1.0, double(std::abs(a - iv.t_obs) / P.tau));
            }
            f.timing = 1.0 - err / double(P.edge.size());
        } else {
            f.timing = f.growth_iou;  // the fire did not grow: only overlap matters
        }
    }
    if (!iv.hotspots.empty()) {
        // Hotspots show where the fire is burning *now*: the front should
        // arrive shortly before the observation.  Arriving far too early
        // means the model runs too fast, never arriving means too slow.
        double hs = 0;
        for (int c : iv.hotspots) {
            double lag = iv.t_obs - run.arrival[c];  // minutes the cell has been burning
            if (run.arrival[c] >= kInf || lag < -15) continue;
            hs += lag <= 90 ? 1.0 : std::max(0.0, 1.0 - (lag - 90) / 90.0);
        }
        f.hotspots = hs / double(iv.hotspots.size());
    }
    if (!has_obs) f.score = f.hotspots;
    else if (iv.hotspots.empty()) f.score = 0.8 * f.growth_iou + 0.2 * f.timing;
    // With hotspots the "observed area" is only their hull: trust its edge less.
    else f.score = 0.5 * f.growth_iou + 0.15 * f.timing + 0.35 * f.hotspots;
    return f;
}

struct Eval {
    Vec x;
    double f;
};

// Per-direction spread correction near the observed front.  Sectors around
// the centre of the previous fire state compare how far the fire actually
// grew with how far the model grew it; the log-ratios are smoothed,
// interpolated by bearing and faded with distance from the observed burned
// area (e-folding ≈ 1.5 × the observed growth distance, ≥ 600 m).
std::shared_ptr<const std::vector<float>> local_correction(const SpreadSolver& solver, double t0_epoch,
                                                           const Prepared& P, const SpreadParams& p, double& lo,
                                                           double& hi) {
    const Grid& g = solver.land().grid;
    const CalibrationInterval& iv = *P.iv;
    const size_t N = g.size();
    constexpr int S = 16;
    double cr = 0, cc = 0, n0 = 0;
    for (size_t k = 0; k < N; ++k)
        if (P.prior[k]) { cr += double(k / g.cols); cc += double(k % g.cols); n0 += 1; }
    if (n0 == 0) return nullptr;
    cr /= n0;
    cc /= n0;
    auto polar = [&](size_t k, double& ang, double& dist) {
        double dr = double(k / g.cols) - cr, dc = double(k % g.cols) - cc;
        dist = std::hypot(dr, dc);
        ang = std::fmod(std::atan2(dc, -dr) / kDeg + 360.0, 360.0);
    };
    auto sector = [&](double ang) { return std::min(S - 1, int(ang / (360.0 / S))); };
    std::array<double, S> prior_r{}, obs_r{}, obs_n{};
    for (size_t k = 0; k < N; ++k) {
        double a, d;
        if (P.prior[k]) {
            polar(k, a, d);
            prior_r[sector(a)] = std::max(prior_r[sector(a)], d);
        } else if (iv.observed[k]) {
            polar(k, a, d);
            obs_r[sector(a)] = std::max(obs_r[sector(a)], d);
            obs_n[sector(a)] += 1;
        }
    }
    double grow = 0;
    int ng = 0;
    for (int s = 0; s < S; ++s)
        if (obs_n[s] > 0) { grow += std::max(0.0, obs_r[s] - prior_r[s]); ++ng; }
    if (!ng) return nullptr;
    const double fade_cells = std::max(600.0 / g.cell_m, 1.5 * grow / ng);
    // Distance (cells) from the observed burned area.
    std::vector<float> dobs;
    {
        std::vector<uint8_t> seed(N, 0);
        for (size_t k = 0; k < N; ++k) seed[k] = iv.observed[k] || P.prior[k];
        distance_from(g, seed, dobs);
    }
    std::array<double, S> total{};
    auto field = std::make_shared<std::vector<float>>(N, 1.f);
    SpreadParams q = p;
    for (int pass = 0; pass < 2; ++pass) {
        RunConfig cfg;
        cfg.t0_epoch = t0_epoch;
        cfg.t_end = iv.t_obs;
        cfg.sources = iv.sources;
        cfg.p = q;
        cfg.detail = false;
        RunOutput out;
        solver.run(cfg, out);
        std::array<double, S> pred_r{}, pred_n{};
        for (size_t k = 0; k < N; ++k) {
            if (P.prior[k] || out.arrival[k] > iv.t_obs) continue;
            double a, d;
            polar(k, a, d);
            pred_r[sector(a)] = std::max(pred_r[sector(a)], d);
            pred_n[sector(a)] += 1;
        }
        std::array<double, S> lr{};
        for (int s = 0; s < S; ++s) {
            if (obs_n[s] + pred_n[s] < 4) continue;  // too little evidence in this direction
            double o = std::max(0.0, obs_r[s] - prior_r[s]) + 1.0, m = std::max(0.0, pred_r[s] - prior_r[s]) + 1.0;
            lr[s] = std::clamp(std::log(o / m), std::log(0.4), std::log(2.5));
        }
        for (int s = 0; s < S; ++s)
            total[s] += 0.25 * lr[(s + S - 1) % S] + 0.5 * lr[s] + 0.25 * lr[(s + 1) % S];
        for (int s = 0; s < S; ++s) total[s] = std::clamp(total[s], std::log(0.3), std::log(3.0));
        for (size_t k = 0; k < N; ++k) {
            double a, d;
            polar(k, a, d);
            double x = a / (360.0 / S) - 0.5;
            int s0 = int(std::floor(x));
            double w = x - s0;
            s0 = (s0 % S + S) % S;
            double l = (1 - w) * total[s0] + w * total[(s0 + 1) % S];
            (*field)[k] = float(std::exp(l * std::exp(-dobs[k] / fade_cells)));
        }
        q.local_ros = field;
        if (pass == 0) field = std::make_shared<std::vector<float>>(*field);  // next pass writes a fresh copy
    }
    lo = 1e9;
    hi = 0;
    for (int s = 0; s < S; ++s) { lo = std::min(lo, std::exp(total[s])); hi = std::max(hi, std::exp(total[s])); }
    return q.local_ros;
}

}  // namespace

FitDetail evaluate_fit(const Grid& g, const RunOutput& run, const CalibrationInterval& iv, bool front_distance) {
    Prepared P = prepare(g, iv);
    FitDetail f = score_prepared(g, run, P);
    if (front_distance) f.front_err_m = front_error_m(g, run, iv);
    return f;
}

double agreement_score(const Grid& g, const RunOutput& run, const CalibrationInput& in) {
    CalibrationInterval iv{in.sources, in.t_obs, in.observed, in.hotspots, 1.0};
    return evaluate_fit(g, run, iv).score;
}

CalibrationResult calibrate(const SpreadSolver& solver, double t0_epoch, const CalibrationInput& in) {
    auto tic = std::chrono::steady_clock::now();
    const Grid& g = solver.land().grid;
    CalibrationResult res;

    std::vector<CalibrationInterval> ivs;
    ivs.push_back({in.sources, in.t_obs, in.observed, in.hotspots, 1.0});
    for (const auto& h : in.history)
        if (h.weight > 0.02 && !h.sources.empty()) ivs.push_back(h);
    std::vector<Prepared> prep;
    double wsum = 0;
    for (const auto& iv : ivs) {
        prep.push_back(prepare(g, iv));
        wsum += iv.weight;
    }

    // The global fit starts clean: an older local correction belongs to an
    // older front.
    SpreadParams start = in.start;
    start.local_ros.reset();
    std::atomic<int> evals{0};
    auto run_with = [&](const SpreadParams& p, const Prepared& P, RunOutput& out) {
        RunConfig cfg;
        cfg.t0_epoch = t0_epoch;
        cfg.t_end = P.t_run;
        cfg.sources = P.iv->sources;
        cfg.p = p;
        cfg.detail = false;
        solver.run(cfg, out);
        ++evals;
    };
    // Loss: recency-weighted disagreement over all intervals plus a gentle
    // pull towards the prior so ambiguous observations do not produce wild
    // parameters.
    auto loss = [&](const Vec& x) {
        SpreadParams p = decode(x, start);
        double tot = 0;
        RunOutput out;
        for (const auto& P : prep) {
            run_with(p, P, out);
            tot += P.iv->weight * (1.0 - score_prepared(g, out, P).score);
        }
        double reg = 0;
        for (double v : x) reg += v * v;
        return tot / wsum + 0.01 * reg;
    };

    // 1. Quasi-random scan of the whole space (Halton, ±2.2 units) plus the prior.
    std::vector<Eval> seeds(size_t(std::max(0, in.seeds)) + 1);
    seeds[0].x = Vec{};
    for (size_t i = 1; i < seeds.size(); ++i) {
        static const int base[D] = {2, 3, 5, 7};
        for (int k = 0; k < D; ++k) seeds[i].x[k] = (2.0 * halton(int(i) + 16, base[k]) - 1.0) * 2.2;
    }
    parallel_for(seeds.size(), hw_threads(), [&](size_t i) { seeds[i].f = loss(seeds[i].x); });
    std::sort(seeds.begin(), seeds.end(), [](const Eval& a, const Eval& b) { return a.f < b.f; });

    // 2. Nelder–Mead from the best few seeds that lie in different basins.
    std::vector<Vec> starts;
    for (const auto& s : seeds) {
        bool far = true;
        for (const auto& t : starts) {
            double d2 = 0;
            for (int k = 0; k < D; ++k) d2 += (s.x[k] - t[k]) * (s.x[k] - t[k]);
            far &= d2 > 0.6 * 0.6;
        }
        if (far) starts.push_back(s.x);
        if (int(starts.size()) >= std::max(1, in.starts)) break;
    }
    std::vector<Eval> found(starts.size());
    parallel_for(starts.size(), unsigned(starts.size()), [&](size_t si) {
        auto f = [&](const Vec& x) { return loss(x); };
        std::array<Vec, D + 1> s;
        std::array<double, D + 1> fs;
        s[0] = starts[si];
        fs[0] = f(s[0]);
        for (int i = 0; i < D; ++i) {
            s[i + 1] = s[0];
            s[i + 1][i] += 0.35;
            fs[i + 1] = f(s[i + 1]);
        }
        for (int it = 0; it < 80; ++it) {
            std::array<int, D + 1> ord;
            for (int i = 0; i <= D; ++i) ord[i] = i;
            std::sort(ord.begin(), ord.end(), [&](int a, int b) { return fs[a] < fs[b]; });
            auto s2 = s;
            auto f2 = fs;
            for (int i = 0; i <= D; ++i) { s[i] = s2[ord[i]]; fs[i] = f2[ord[i]]; }
            if (fs[D] - fs[0] < 1e-3 && it > 10) break;

            Vec c{};
            for (int i = 0; i < D; ++i)
                for (int k = 0; k < D; ++k) c[k] += s[i][k] / D;
            auto along = [&](double t) {
                Vec x;
                for (int k = 0; k < D; ++k) x[k] = c[k] + t * (s[D][k] - c[k]);
                return x;
            };
            Vec xr = along(-1.0);
            double fr = f(xr);
            if (fr < fs[0]) {
                Vec xe = along(-2.0);
                double fe = f(xe);
                if (fe < fr) { s[D] = xe; fs[D] = fe; }
                else { s[D] = xr; fs[D] = fr; }
            } else if (fr < fs[D - 1]) {
                s[D] = xr;
                fs[D] = fr;
            } else {
                Vec xc = along(fr < fs[D] ? -0.5 : 0.5);
                double fc = f(xc);
                if (fc < std::min(fr, fs[D])) {
                    s[D] = xc;
                    fs[D] = fc;
                } else {
                    for (int i = 1; i <= D; ++i) {
                        for (int k = 0; k < D; ++k) s[i][k] = s[0][k] + 0.5 * (s[i][k] - s[0][k]);
                        fs[i] = f(s[i]);
                    }
                }
            }
        }
        int bi = int(std::min_element(fs.begin(), fs.end()) - fs.begin());
        found[si] = {s[bi], fs[bi]};
    });
    Eval best = seeds.front();
    for (const auto& e : found)
        if (e.f < best.f) best = e;

    // 3. How far each parameter can move before the fit gets clearly worse:
    //    that width is the uncertainty the ensemble should carry.
    const double delta = 0.02 + 0.15 * best.f;
    std::array<double, 2 * D> width{};
    parallel_for(2 * D, hw_threads(), [&](size_t j) {
        const int k = int(j / 2);
        const double sgn = (j % 2) ? 1.0 : -1.0;
        double prev = 0, w = 1.6;
        double f_prev = best.f;
        for (double t : {0.1, 0.2, 0.4, 0.8, 1.6}) {
            Vec x = best.x;
            x[k] += sgn * t;
            double v = loss(x);
            if (v > best.f + delta) {
                // Interpolate where the loss crosses the threshold.
                double a = (best.f + delta - f_prev) / std::max(1e-9, v - f_prev);
                w = prev + std::clamp(a, 0.0, 1.0) * (t - prev);
                break;
            }
            prev = t;
            f_prev = v;
        }
        width[j] = w;
    });
    double sig[D];
    for (int k = 0; k < D; ++k) sig[k] = 0.25 * (width[2 * k] + width[2 * k + 1]);  // ≈ σ: half of the mean half-width
    res.spread.ros_sigma = kUnit[0] * sig[0];
    res.spread.wind_speed_sigma = kUnit[1] * sig[1];
    res.spread.wind_dir_sigma = kUnit[2] * sig[2];
    res.spread.moisture_sigma = kUnit[3] * sig[3];

    res.params = decode(best.x, start);
    {
        double lo = 1, hi = 1;
        if (in.local_correction && !ivs[0].observed.empty() && prep[0].obs_growth >= 8)
            if (auto field = local_correction(solver, t0_epoch, prep[0], res.params, lo, hi)) {
                SpreadParams q = res.params;
                q.local_ros = field;
                RunOutput a, b;
                run_with(res.params, prep[0], a);
                run_with(q, prep[0], b);
                // Keep it only when it really explains the observation better.
                if (score_prepared(g, b, prep[0]).score > score_prepared(g, a, prep[0]).score + 0.005) {
                    res.params = q;
                    res.local_min = lo;
                    res.local_max = hi;
                }
            }
    }
    {
        RunOutput a, b;
        run_with(in.start, prep[0], a);  // the forecast as it was (with any older correction)
        run_with(res.params, prep[0], b);
        res.before = score_prepared(g, a, prep[0]);
        res.before.front_err_m = front_error_m(g, a, ivs[0]);
        res.after = score_prepared(g, b, prep[0]);
        res.after.front_err_m = front_error_m(g, b, ivs[0]);
    }
    res.score_before = res.before.score;
    res.score_after = res.after.score;
    res.evaluations = evals.load();
    res.runtime_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tic).count();
    return res;
}

}  // namespace rapid
