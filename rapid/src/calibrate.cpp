#include "calibrate.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <future>

namespace rapid {

namespace {

constexpr int D = 4;
using Vec = std::array<double, D>;

// Search space is normalised so one unit is a "large but plausible" error.
SpreadParams decode(const Vec& x, const SpreadParams& base) {
    SpreadParams p = base;
    p.ros_mult = std::clamp(base.ros_mult * std::exp(0.7 * x[0]), 0.15, 6.0);
    p.wind_mult = std::clamp(base.wind_mult * std::exp(0.4 * x[1]), 0.3, 3.0);
    p.wind_dir_offset = std::clamp(base.wind_dir_offset + 25.0 * x[2], -60.0, 60.0);
    p.moisture_offset = std::clamp(base.moisture_offset + 0.025 * x[3], -0.08, 0.10);
    return p;
}

}  // namespace

double agreement_score(const RunOutput& run, const CalibrationInput& in) {
    size_t inter = 0, uni = 0;
    if (!in.observed.empty()) {
        for (size_t i = 0; i < run.arrival.size(); ++i) {
            bool p = run.arrival[i] <= in.t_obs, o = in.observed[i] != 0;
            inter += p && o;
            uni += p || o;
        }
    }
    double iou = uni ? double(inter) / double(uni) : 1.0;
    if (in.hotspots.empty()) return iou;
    // Hotspots (drone / thermal) show where the fire is burning *now*: the
    // front should arrive shortly before the observation.  Arriving far too
    // early means the model runs too fast, never arriving means too slow.
    double hs = 0;
    for (int c : in.hotspots) {
        double lag = in.t_obs - run.arrival[c];  // minutes the cell has been burning
        if (run.arrival[c] >= kInf || lag < -15) continue;
        hs += lag <= 90 ? 1.0 : std::max(0.0, 1.0 - (lag - 90) / 90.0);
    }
    hs /= double(in.hotspots.size());
    return in.observed.empty() ? hs : 0.75 * iou + 0.25 * hs;
}

CalibrationResult calibrate(const SpreadSolver& solver, double t0_epoch, const CalibrationInput& in) {
    auto tic = std::chrono::steady_clock::now();
    CalibrationResult res;
    RunConfig base;
    base.t0_epoch = t0_epoch;
    // Run a little past the observation so over-prediction is also visible.
    base.t_end = in.t_obs + 5.0f;
    base.sources = in.sources;
    base.detail = false;

    auto score_of = [&](const SpreadParams& p) {
        RunConfig cfg = base;
        cfg.p = p;
        RunOutput out;
        solver.run(cfg, out);
        return agreement_score(out, in);
    };
    // Loss: disagreement plus a gentle pull towards the prior so that
    // ambiguous observations do not produce wild parameters.
    auto loss = [&](const Vec& x) {
        double reg = 0;
        for (double v : x) reg += v * v;
        return (1.0 - score_of(decode(x, in.start))) + 0.01 * reg;
    };

    res.score_before = score_of(in.start);

    // 1. Coarse parallel scan over direction bias × spread rate (the two
    //    parameters that dominate perimeter shape), keeping the best start.
    std::vector<Vec> grid;
    for (double d : {-1.6, -0.8, 0.0, 0.8, 1.6})
        for (double r : {-1.4, -0.7, 0.0, 0.7, 1.4}) grid.push_back({r, 0.0, d, 0.0});
    std::vector<std::future<double>> fut;
    for (auto& x : grid) fut.push_back(std::async(std::launch::async, [&, x] { return loss(x); }));
    Vec best = grid[0];
    double best_f = 1e9;
    for (size_t i = 0; i < grid.size(); ++i) {
        double f = fut[i].get();
        if (f < best_f) { best_f = f; best = grid[i]; }
    }
    int evals = int(grid.size()) + 1;

    // 2. Nelder–Mead refinement in all four dimensions.
    std::array<Vec, D + 1> s;
    std::array<double, D + 1> fs;
    s[0] = best;
    fs[0] = best_f;
    for (int i = 0; i < D; ++i) {
        s[i + 1] = best;
        s[i + 1][i] += 0.5;
    }
    {
        std::vector<std::future<double>> f2;
        for (int i = 1; i <= D; ++i) f2.push_back(std::async(std::launch::async, [&, i] { return loss(s[i]); }));
        for (int i = 1; i <= D; ++i) fs[i] = f2[i - 1].get();
        evals += D;
    }
    for (int it = 0; it < 60; ++it) {
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
        double fr = loss(xr);
        ++evals;
        if (fr < fs[0]) {
            Vec xe = along(-2.0);
            double fe = loss(xe);
            ++evals;
            if (fe < fr) { s[D] = xe; fs[D] = fe; }
            else { s[D] = xr; fs[D] = fr; }
        } else if (fr < fs[D - 1]) {
            s[D] = xr;
            fs[D] = fr;
        } else {
            Vec xc = along(fr < fs[D] ? -0.5 : 0.5);
            double fc = loss(xc);
            ++evals;
            if (fc < std::min(fr, fs[D])) {
                s[D] = xc;
                fs[D] = fc;
            } else {
                for (int i = 1; i <= D; ++i) {
                    for (int k = 0; k < D; ++k) s[i][k] = s[0][k] + 0.5 * (s[i][k] - s[0][k]);
                    fs[i] = loss(s[i]);
                    ++evals;
                }
            }
        }
    }
    int bi = int(std::min_element(fs.begin(), fs.end()) - fs.begin());
    res.params = decode(s[bi], in.start);
    res.score_after = score_of(res.params);
    if (res.score_after < res.score_before) {  // never make things worse
        res.params = in.start;
        res.score_after = res.score_before;
    }
    res.evaluations = evals + 2;
    res.runtime_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tic).count();
    return res;
}

}  // namespace rapid
