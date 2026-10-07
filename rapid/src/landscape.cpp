#include "landscape.hpp"

#include <algorithm>
#include <cmath>
#include <future>

#include "fetch.hpp"
#include "fuels.hpp"

namespace rapid {

void derive_terrain(Landscape& L) {
    const Grid& g = L.grid;
    const int R = g.rows, C = g.cols;
    L.slope_tan.assign(g.size(), 0.f);
    L.upslope.assign(g.size(), 0.f);
    L.wind_mult.assign(g.size(), 1.f);
    L.moist_adj.assign(g.size(), 0);
    auto z = [&](int r, int c) { return L.elev[g.idx(std::clamp(r, 0, R - 1), std::clamp(c, 0, C - 1))]; };

    // Horn (1981) gradient; x = east, y = north (row 0 is north).
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < C; ++c) {
            double a = z(r - 1, c - 1), b = z(r - 1, c), cc = z(r - 1, c + 1);
            double d = z(r, c - 1), f = z(r, c + 1);
            double gg = z(r + 1, c - 1), h = z(r + 1, c), i = z(r + 1, c + 1);
            double dzdx = ((cc + 2 * f + i) - (a + 2 * d + gg)) / (8 * g.cell_m);
            double dzdy = ((a + 2 * b + cc) - (gg + 2 * h + i)) / (8 * g.cell_m);
            int k = g.idx(r, c);
            L.slope_tan[k] = float(std::min(2.0, std::hypot(dzdx, dzdy)));
            L.upslope[k] = float(std::fmod(std::atan2(dzdx, dzdy) / kDeg + 360.0, 360.0));
            if (L.slope_tan[k] > 0.18f) {
                double facing = std::fmod(L.upslope[k] + 180.0, 360.0);  // aspect
                if (facing >= 135 && facing <= 225) L.moist_adj[k] = 1;       // sun-baked south face
                else if (facing >= 315 || facing <= 45) L.moist_adj[k] = 2;   // shaded north face
            }
        }

    // Terrain exposure: ridges see stronger wind, valleys are sheltered.
    // Topographic position index over a ~500 m box via an integral image.
    std::vector<double> integ(size_t(R + 1) * (C + 1), 0.0);
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < C; ++c)
            integ[(r + 1) * (C + 1) + c + 1] = L.elev[g.idx(r, c)] + integ[r * (C + 1) + c + 1] +
                                               integ[(r + 1) * (C + 1) + c] - integ[r * (C + 1) + c];
    int k = std::max(2, int(std::round(500.0 / g.cell_m)));
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < C; ++c) {
            int r0 = std::max(0, r - k), r1 = std::min(R, r + k + 1), c0 = std::max(0, c - k), c1 = std::min(C, c + k + 1);
            double s = integ[r1 * (C + 1) + c1] - integ[r0 * (C + 1) + c1] - integ[r1 * (C + 1) + c0] + integ[r0 * (C + 1) + c0];
            double mean = s / double((r1 - r0) * (c1 - c0));
            double tpi = L.elev[g.idx(r, c)] - mean;
            L.wind_mult[g.idx(r, c)] = float(std::clamp(1.0 + tpi / 400.0, 0.7, 1.3));
        }
}

Landscape build_landscape(const BuildOptions& opt) {
    Landscape L;
    L.grid = Grid::centred(opt.centre, opt.radius_m, opt.cell_m);
    const Grid& g = L.grid;
    FetchContext ctx{opt.cache_dir, opt.offline};

    std::string terr_src, fuel_src;
    auto f_elev = std::async(std::launch::async, [&] { return fetch_elevation(ctx, g, L.elev, terr_src); });
    std::vector<uint8_t> fuel;
    auto f_fuel = std::async(std::launch::async, [&] { return fetch_fuel_corine(ctx, g, fuel, fuel_src); });
    L.terrain_ok = f_elev.get();
    bool fuel_ok = f_fuel.get();

    if (!L.terrain_ok) {
        if (L.elev.size() != g.size()) L.elev.assign(g.size(), 0.f);
        L.warnings.push_back("Terrain unavailable — assuming flat ground.");
        L.terrain_source = "flat (no data)";
    } else {
        L.terrain_source = terr_src;
    }
    derive_terrain(L);

    const uint8_t water = uint8_t(fuel_index("Water"));
    const uint8_t fallback = uint8_t(fuel_index("Garrigue"));
    if (fuel_ok) {
        // Fill unclassified pixels (map labels, anti-aliased edges) from neighbours.
        for (int pass = 0; pass < 3; ++pass) {
            std::vector<uint8_t> next = fuel;
            for (int r = 0; r < g.rows; ++r)
                for (int c = 0; c < g.cols; ++c) {
                    if (fuel[g.idx(r, c)] != 255) continue;
                    int counts[32] = {0};
                    int best = 255, bc = 0;
                    for (int dr = -1; dr <= 1; ++dr)
                        for (int dc = -1; dc <= 1; ++dc) {
                            if (!g.inside(r + dr, c + dc)) continue;
                            uint8_t v = fuel[g.idx(r + dr, c + dc)];
                            if (v == 255) continue;
                            if (++counts[v] > bc) { bc = counts[v]; best = v; }
                        }
                    next[g.idx(r, c)] = uint8_t(best);
                }
            fuel.swap(next);
        }
        L.fuel_ok = true;
        L.fuel_source = fuel_src;
    } else {
        fuel.assign(g.size(), 255);
        L.warnings.push_back("Land-cover map unavailable — using generic Mediterranean scrub everywhere.");
        L.fuel_source = "generic scrub (no data)";
    }
    for (size_t i = 0; i < fuel.size(); ++i)
        if (fuel[i] == 255) fuel[i] = (L.terrain_ok && L.elev[i] <= 0.5f) ? water : fallback;
    L.fuel = std::move(fuel);
    L.road.assign(g.size(), 0);
    return L;
}

}  // namespace rapid
