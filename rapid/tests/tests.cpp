// Offline unit tests for the rapid estimator (no network access needed).
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <tuple>

#include "calibrate.hpp"
#include "ensemble.hpp"
#include "fuels.hpp"
#include "geo.hpp"
#include "planner.hpp"
#include "landscape.hpp"
#include "spread.hpp"
#include "tactics.hpp"
#include "weather.hpp"

using namespace rapid;

static int g_fail = 0;
#define CHECK(cond, ...)                                       \
    do {                                                       \
        if (!(cond)) {                                         \
            ++g_fail;                                          \
            std::printf("FAIL %s:%d  %s  ", __FILE__, __LINE__, #cond); \
            std::printf(__VA_ARGS__);                          \
            std::printf("\n");                                 \
        }                                                      \
    } while (0)

static Landscape flat_land(int n, double cell, const char* fuel) {
    Landscape L;
    L.grid = Grid::centred({38.5, 22.0}, n * cell / 2, cell);
    L.elev.assign(L.grid.size(), 100.f);
    derive_terrain(L);
    L.fuel.assign(L.grid.size(), uint8_t(fuel_index(fuel)));
    L.road.assign(L.grid.size(), 0);
    return L;
}

static WeatherSeries const_wx(double ws, double from, double t = 30, double rh = 25) {
    WeatherSeries w;
    for (int h = -2; h < 48; ++h) w.hourly.push_back({h * 3600.0, ws, from, t, rh});
    w.source = "test";
    return w;
}

static double reach_m(const Landscape& L, const RunOutput& o, int dr, int dc, float t) {
    const Grid& g = L.grid;
    int r = g.rows / 2, c = g.cols / 2, steps = 0;
    while (g.inside(r, c) && o.arrival[g.idx(r, c)] <= t) {
        r += dr;
        c += dc;
        ++steps;
    }
    return steps * g.cell_m * std::hypot(dr, dc);
}

int main() {
    // 1. Rothermel: dry grass, 6 % moisture, no wind — expect ~1–3 m/min;
    //    with 3 m/s midflame wind expect tens of m/min (Scott & Burgan GR2).
    {
        auto rf = rothermel_prepare(fuel_table()[fuel_index("Dry_Grass")]);
        auto b = rothermel_base(rf, 0.06);
        double r0 = b.r0_ftmin * kFtMinToMMin;
        double u = 3.0 * kMsToFtMin;
        double phi_w = rf.C * std::pow(u, rf.B) * std::pow(rf.rel_beta, -rf.E);
        double rw = r0 * (1 + phi_w);
        std::printf("dry grass: R0 = %.2f m/min, R(3 m/s midflame) = %.1f m/min\n", r0, rw);
        CHECK(r0 > 0.5 && r0 < 6, "r0=%.2f", r0);
        CHECK(rw > 15 && rw < 120, "rw=%.1f", rw);
        auto shrub = rothermel_base(rothermel_prepare(fuel_table()[fuel_index("Maquis_Dense_Shrub")]), 0.06);
        std::printf("maquis: R0 = %.2f m/min\n", shrub.r0_ftmin * kFtMinToMMin);
        CHECK(!rothermel_prepare(fuel_table()[fuel_index("Water")]).ok, "water must not burn");
        CHECK(rothermel_base(rf, 0.20).r0_ftmin == 0, "above extinction moisture must not spread");
    }

    // 1b. Head-fire behaviour table (flat, 32 °C / 18 % RH) — printed for review;
    //     pine at 35 km/h must crown and run at km/h speeds, grass ≫ forest litter.
    {
        auto wx_dummy = const_wx(0, 0);
        std::printf("\n%-24s %8s %8s %8s   (km/h head spread, flame m @ 35)\n", "fuel", "10 km/h", "20 km/h", "35 km/h");
        double pine35 = 0, grass35 = 0;
        for (const auto& f : fuel_table()) {
            if (!f.burnable()) continue;
            Landscape P = flat_land(16, 30, f.name);
            double v[3], fl = 0;
            int i = 0;
            for (double kmh : {10.0, 20.0, 35.0}) {
                auto w = const_wx(kmh / 3.6, 0, 32, 18);
                SpreadSolver s(P, w);
                v[i++] = s.head_ros(P.grid.idx(8, 8), 0, 60, SpreadParams{}, &fl) * 0.06;
            }
            std::printf("%-24s %8.2f %8.2f %8.2f   %5.1f\n", f.name, v[0], v[1], v[2], fl);
            if (std::string(f.name) == "Aleppo_Pine") pine35 = v[2];
            if (std::string(f.name) == "Dry_Grass") grass35 = v[2];
        }
        CHECK(pine35 > 1.0, "pine at 35 km/h should crown: %.2f km/h", pine35);
        CHECK(grass35 > 2.0, "grass at 35 km/h: %.2f km/h", grass35);
        (void)wx_dummy;
    }

    // 2. EMC sanity: hot & dry → ~3–6 %.
    {
        double m = emc_fraction(35, 15);
        CHECK(m > 0.02 && m < 0.06, "emc=%.3f", m);
    }

    // 3. No wind, flat → near-circular fire.
    {
        auto L = flat_land(201, 30, "Dry_Grass");
        auto wx = const_wx(0, 0);
        SpreadSolver s(L, wx);
        RunConfig cfg;
        cfg.t_end = 1000;
        int mid = L.grid.idx(100, 100);
        cfg.sources = {{mid, 0.f}};
        RunOutput o;
        s.run(cfg, o);
        double n = reach_m(L, o, -1, 0, 900), e = reach_m(L, o, 0, 1, 900), d = reach_m(L, o, -1, 1, 900);
        std::printf("calm: 15-h reach N %.0f m, E %.0f m, NE %.0f m\n", n, e, d);
        CHECK(std::abs(n - e) <= 30, "N %.0f vs E %.0f", n, e);
        CHECK(std::abs(d - n) / n < 0.2, "diag %.0f vs %.0f", d, n);
    }

    // 4. North wind (blowing to the south) → long head to the south, short backing fire.
    {
        auto L = flat_land(301, 30, "Dry_Grass");
        auto wx = const_wx(6, 0);
        SpreadSolver s(L, wx);
        RunConfig cfg;
        cfg.t_end = 90;
        cfg.sources = {{L.grid.idx(150, 150), 0.f}};
        RunOutput o;
        s.run(cfg, o);
        double south = reach_m(L, o, 1, 0, 30), north = reach_m(L, o, -1, 0, 30), east = reach_m(L, o, 0, 1, 30);
        std::printf("6 m/s north wind, 30 min: head(S) %.0f m, flank(E) %.0f m, back(N) %.0f m\n", south, east, north);
        CHECK(south > 3 * north, "head %.0f back %.0f", south, north);
        CHECK(east >= north && east < south, "flank %.0f", east);
    }

    // 5. Upslope runs faster than downslope.
    {
        auto L = flat_land(201, 30, "Garrigue");
        for (int r = 0; r < L.grid.rows; ++r)
            for (int c = 0; c < L.grid.cols; ++c) L.elev[L.grid.idx(r, c)] = float(100 + (L.grid.rows - r) * 30 * 0.4);
        derive_terrain(L);
        auto wx = const_wx(0, 0);
        SpreadSolver s(L, wx);
        RunConfig cfg;
        cfg.t_end = 120;
        cfg.sources = {{L.grid.idx(100, 100), 0.f}};
        RunOutput o;
        s.run(cfg, o);
        double up = reach_m(L, o, -1, 0, 60), down = reach_m(L, o, 1, 0, 60);
        std::printf("40%% slope, 60 min: upslope %.0f m, downslope %.0f m\n", up, down);
        CHECK(up > 2 * down, "up %.0f down %.0f", up, down);
    }

    // 6. A firebreak completed before the fire arrives stops it; one built too late does not.
    {
        auto L = flat_land(201, 30, "Dry_Grass");
        auto wx = const_wx(6, 0);
        SpreadSolver s(L, wx);
        Grid& g = L.grid;
        Intervention fb;
        fb.kind = IvKind::Firebreak;
        fb.width_m = 30;
        fb.pts = {g.center_of(130, 20), g.center_of(130, 180)};
        fb.t_on = 0;
        InterventionRaster iv;
        iv.build(g, {fb});
        RunConfig cfg;
        cfg.t_end = 600;
        cfg.sources = {{g.idx(100, 100), 0.f}};
        cfg.iv = &iv;
        RunOutput o;
        s.run(cfg, o);
        float behind = o.arrival[g.idx(140, 100)];
        float around = o.arrival[g.idx(140, 195)];
        auto mins = [](float t) { return t >= kInf ? std::string("never") : std::to_string(int(t)) + " min"; };
        std::printf("firebreak: behind it %s, past its end %s (in 600 min)\n", mins(behind).c_str(), mins(around).c_str());
        CHECK(behind > o.arrival[g.idx(125, 100)] + 30, "fire should be delayed behind the break");
        iv.layers[0].t_on = 1e6;  // never finished
        RunOutput o2;
        s.run(cfg, o2);
        CHECK(o2.arrival[g.idx(140, 100)] < behind, "unfinished break must not stop the fire");
    }

    // 7. Performance: 600×600 grid, 6 h of spread in mixed fuel.
    {
        auto L = flat_land(600, 40, "Dry_Grass");
        for (int r = 0; r < L.grid.rows; ++r)  // shrub patches
            for (int c = 0; c < L.grid.cols; ++c)
                if ((r / 40 + c / 40) % 3 == 0) L.fuel[L.grid.idx(r, c)] = uint8_t(fuel_index("Maquis_Dense_Shrub"));
        for (int r = 0; r < L.grid.rows; ++r)
            for (int c = 0; c < L.grid.cols; ++c)
                L.elev[L.grid.idx(r, c)] = float(300 + 150 * std::sin(r * 0.03) * std::cos(c * 0.02));
        derive_terrain(L);
        auto wx = const_wx(8, 300);
        SpreadSolver s(L, wx);
        RunConfig cfg;
        cfg.t_end = 24 * 60;
        cfg.sources = {{L.grid.idx(300, 300), 0.f}};
        RunOutput o;
        auto t0 = std::chrono::steady_clock::now();
        s.run(cfg, o);
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        size_t burned = 0;
        for (float a : o.arrival) burned += a < 1e29f;
        std::printf("perf: 600x600, 24 h → %zu cells (%.0f ha) in %.1f ms\n", burned, burned * 0.16, ms);
        CHECK(ms < 2000, "too slow: %.0f ms", ms);
    }

    // 8. Calibration recovers a known wind-direction and spread-rate bias
    //    from the burned area alone.
    {
        auto L = flat_land(241, 30, "Dry_Grass");
        auto wx = const_wx(5, 0);
        SpreadSolver s(L, wx);
        RunConfig truth_cfg;
        truth_cfg.t_end = 75;
        truth_cfg.sources = {{L.grid.idx(60, 120), 0.f}};
        truth_cfg.p.wind_dir_offset = 25;
        truth_cfg.p.ros_mult = 1.5;
        RunOutput truth;
        s.run(truth_cfg, truth);
        CalibrationInput ci;
        ci.sources = truth_cfg.sources;
        ci.t_obs = 60;
        ci.observed.assign(L.grid.size(), 0);
        size_t n = 0;
        for (size_t i = 0; i < truth.arrival.size(); ++i) n += ci.observed[i] = truth.arrival[i] <= 60;
        CalibrationResult cr = calibrate(s, 0, ci);
        std::printf("calibration: %zu observed cells, agreement %.2f → %.2f in %d runs / %.0f ms; dir %+.1f° (true +25), ros×%.2f (true 1.5), wind×%.2f\n",
                    n, cr.score_before, cr.score_after, cr.evaluations, cr.runtime_ms, cr.params.wind_dir_offset,
                    cr.params.ros_mult, cr.params.wind_mult);
        CHECK(cr.score_after > 0.85, "agreement %.2f", cr.score_after);
        CHECK(cr.score_after > cr.score_before + 0.2, "calibration must improve");
        CHECK(std::abs(cr.params.wind_dir_offset - 25) < 12, "direction %.1f", cr.params.wind_dir_offset);
    }

    // 9. Tactics: with hours to spare, a firebreak is recommended across the
    //    path to the threatened place, and it actually delays the fire there.
    {
        auto L = flat_land(301, 30, "Dry_Grass");
        auto wx = const_wx(4, 0);
        SpreadSolver s(L, wx);
        const Grid& g = L.grid;
        RunConfig cfg;
        cfg.t_end = 600;
        cfg.sources = {{g.idx(40, 150), 0.f}};
        RunOutput best;
        s.run(cfg, best);
        int village = g.idx(260, 150);
        TacticsInput ti;
        ti.L = &L;
        ti.best = &best;
        ti.t_now = 0;
        ti.t_end = 600;
        ti.origin = g.center_of(40, 150);
        ti.head_bearing = 180;
        ti.wind_kmh = 14.4;
        ti.threats = {{"d1", "Village", village, best.arrival[village], 0.9}};
        auto recs = plan_tactics(ti);
        const Recommendation* fb = nullptr;
        for (auto& r : recs)
            if (r.type == "firebreak") fb = &r;
        std::printf("tactics: village ETA %.0f min, %zu actions, firebreak %s\n", best.arrival[village], recs.size(),
                    fb ? "recommended" : "MISSING");
        CHECK(fb != nullptr, "expected a firebreak recommendation");
        if (fb) {
            InterventionRaster iv;
            iv.build(g, {fb->iv});
            RunConfig c2 = cfg;
            c2.iv = &iv;
            c2.t_end = 1200;
            RunOutput with;
            s.run(c2, with);
            std::printf("         with the firebreak: ETA %.0f min (break ready at %.0f min)\n", with.arrival[village], fb->iv.t_on);
            CHECK(with.arrival[village] > best.arrival[village] + 30, "firebreak should delay the village");
        }
        bool has_truck = false;
        for (auto& r : recs) has_truck |= r.type == "truck";
        CHECK(has_truck, "expected a truck at the threatened place");
    }

    // 10. Ensemble: members bracket the best guess; quantiles are ordered.
    {
        auto L = flat_land(201, 30, "Dry_Grass");
        auto wx = const_wx(5, 0);
        SpreadSolver s(L, wx);
        RunConfig cfg;
        cfg.t_end = 120;
        cfg.sources = {{L.grid.idx(60, 100), 0.f}};
        auto E = run_ensemble(s, cfg, make_members({}, Uncertainty{}, 16, 7));
        size_t bad = 0, reached = 0;
        for (size_t i = 0; i < L.grid.size(); ++i) {
            if (E.p10[i] >= kInf) continue;
            ++reached;
            bad += !(E.p10[i] <= E.p50[i] && E.p50[i] <= E.p90[i]);
        }
        std::printf("ensemble: 16 members in %.0f ms, %zu cells possibly reached\n", E.runtime_ms, reached);
        CHECK(bad == 0 && reached > 100, "quantiles out of order: %zu", bad);
    }

    // 11. Planner: with no resources nothing is used; with a bulldozer the
    //     firebreak protecting the village is chosen; budgets are respected.
    {
        auto L = flat_land(301, 30, "Dry_Grass");
        auto wx = const_wx(4, 0);
        SpreadSolver s(L, wx);
        const Grid& g = L.grid;
        RunConfig cfg;
        cfg.t_end = 600;
        cfg.sources = {{g.idx(40, 150), 0.f}};
        auto E = run_ensemble(s, cfg, make_members({}, Uncertainty{}, 6, 3));
        int village = g.idx(260, 150);
        TacticsInput ti;
        ti.L = &L;
        ti.best = &E.best;
        ti.t_end = 600;
        ti.origin = g.center_of(40, 150);
        ti.head_bearing = 180;
        ti.wind_kmh = 14.4;
        ti.threats = {{"d1", "Village", village, E.best.arrival[village], 0.9}};
        PlannerInput pin;
        pin.solver = &s;
        pin.cfg = cfg;
        pin.members = E.members;
        pin.base_arrival = &E.arrival;
        PlanTarget t{"d1", "Village", 4, {}};
        rasterize_disc(g, g.center_of(260, 150), 150, t.cells);
        pin.targets = {t};
        pin.t_end = 600;
        pin.cell_ha = 0.09;
        for (auto [a, tr, dz] : {std::tuple{0, 0, 0}, std::tuple{0, 0, 1}, std::tuple{2, 1, 1}}) {
            auto recs = plan_tactics(ti);
            Resources res{a, tr, dz};
            auto out = choose_plan(pin, recs, res);
            int n = 0, breaks = 0;
            for (auto& r : recs) { n += r.selected; breaks += r.selected && r.type == "firebreak"; }
            std::printf("planner %d/%d/%d: %d actions chosen (%d firebreaks) in %d runs / %.0f ms; used %d/%d/%d\n", a, tr, dz,
                        n, breaks, out.simulations, out.runtime_ms, out.used[0], out.used[1], out.used[2]);
            CHECK(out.used[0] <= a && out.used[1] <= tr && out.used[2] <= dz, "budget exceeded");
            if (a + tr + dz == 0) CHECK(n == 0, "nothing should be chosen without resources");
            if (dz == 1) CHECK(breaks == 1, "the protective firebreak should be chosen");
        }
    }

    // 12. Danger map: the reverse search (time for a fire starting at X to reach
    //     a place) matches a forward run from X; hazard is faster in grass.
    {
        auto L = flat_land(201, 30, "Dry_Grass");
        for (int r = 0; r < 201; ++r)
            for (int c = 0; c < 100; ++c) L.fuel[L.grid.idx(r, c)] = uint8_t(fuel_index("Oak_Forest"));
        auto wx = const_wx(5, 0);
        SpreadSolver s(L, wx);
        const Grid& g = L.grid;
        int place = g.idx(150, 150), start = g.idx(60, 150);
        std::vector<float> mins;
        std::vector<int> lab;
        s.time_to_targets(0, 30, {}, {{place, 0}}, 1e6f, mins, lab);
        RunConfig cfg;
        cfg.t_end = 1e6f;
        cfg.sources = {{start, 0.f}};
        RunOutput fwd;
        s.run(cfg, fwd);
        std::printf("danger: reverse %.0f min vs forward %.0f min from 2.7 km upwind\n", mins[start], fwd.arrival[place]);
        CHECK(std::abs(mins[start] - fwd.arrival[place]) < 0.05 * fwd.arrival[place], "reverse/forward mismatch");
        CHECK(lab[start] == 0, "label");
        std::vector<float> ros, fl, dir;
        s.hazard(0, 0, 120, {}, ros, fl, dir);
        std::printf("        hazard: grass %.2f km/h, oak %.2f km/h, direction %.0f°\n", ros[g.idx(100, 150)], ros[g.idx(100, 50)], dir[g.idx(100, 150)]);
        CHECK(ros[g.idx(100, 150)] > 2 * ros[g.idx(100, 50)], "grass should be more dangerous than oak litter");
        CHECK(std::abs(std::remainder(dir[g.idx(100, 150)] - 180.0, 360.0)) < 10, "north wind → spread south");
    }

    std::printf(g_fail ? "\n%d FAILED\n" : "\nall tests passed\n", g_fail);
    return g_fail ? 1 : 0;
}
