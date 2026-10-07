// geo.hpp — grid geometry and small geographic helpers.
//
// The simulation grid is a regular lat/lon raster centred on the incident.
// Row 0 is the northern edge, column 0 the western edge.  Cells are square in
// metres at the centre latitude (dlon is widened by 1/cos(lat)), which is
// accurate to well under one cell over the ≤ 60 km domains used here.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace rapid {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDeg = kPi / 180.0;
constexpr double kMetersPerDegLat = 111320.0;
constexpr float kInf = 1e30f;

struct LatLon {
    double lat = 0, lon = 0;
};

struct Grid {
    int rows = 0, cols = 0;
    double cell_m = 30.0;
    double north = 0, west = 0;  // outer edges
    double dlat = 0, dlon = 0;   // cell size in degrees

    static Grid centred(LatLon c, double radius_m, double cell_m) {
        Grid g;
        g.cell_m = cell_m;
        g.rows = g.cols = std::max(16, int(std::ceil(2.0 * radius_m / cell_m)));
        g.dlat = cell_m / kMetersPerDegLat;
        g.dlon = cell_m / (kMetersPerDegLat * std::cos(c.lat * kDeg));
        g.north = c.lat + g.rows * g.dlat / 2.0;
        g.west = c.lon - g.cols * g.dlon / 2.0;
        return g;
    }

    double south() const { return north - rows * dlat; }
    double east() const { return west + cols * dlon; }
    size_t size() const { return size_t(rows) * size_t(cols); }
    int idx(int r, int c) const { return r * cols + c; }
    bool inside(int r, int c) const { return r >= 0 && c >= 0 && r < rows && c < cols; }

    LatLon center_of(int r, int c) const {
        return {north - (r + 0.5) * dlat, west + (c + 0.5) * dlon};
    }
    // Fractional (row, col) of a coordinate; cell (r,c) spans [r, r+1).
    std::pair<double, double> frac_rc(LatLon p) const {
        return {(north - p.lat) / dlat, (p.lon - west) / dlon};
    }
    bool locate(LatLon p, int& r, int& c) const {
        auto [fr, fc] = frac_rc(p);
        r = int(std::floor(fr));
        c = int(std::floor(fc));
        return inside(r, c);
    }
};

inline double haversine_m(LatLon a, LatLon b) {
    double dlat = (b.lat - a.lat) * kDeg, dlon = (b.lon - a.lon) * kDeg;
    double h = std::sin(dlat / 2) * std::sin(dlat / 2) +
               std::cos(a.lat * kDeg) * std::cos(b.lat * kDeg) * std::sin(dlon / 2) * std::sin(dlon / 2);
    return 2 * 6371000.0 * std::asin(std::sqrt(std::min(1.0, h)));
}

// Compass bearing (degrees clockwise from north) from a to b.
inline double bearing_deg(LatLon a, LatLon b) {
    double dy = (b.lat - a.lat);
    double dx = (b.lon - a.lon) * std::cos(0.5 * (a.lat + b.lat) * kDeg);
    double t = std::atan2(dx, dy) / kDeg;
    return t < 0 ? t + 360 : t;
}

inline LatLon offset_m(LatLon p, double bearing, double dist_m) {
    double dn = dist_m * std::cos(bearing * kDeg), de = dist_m * std::sin(bearing * kDeg);
    return {p.lat + dn / kMetersPerDegLat, p.lon + de / (kMetersPerDegLat * std::cos(p.lat * kDeg))};
}

inline std::string compass_name(double deg) {
    static const char* n[] = {"north", "north-east", "east", "south-east",
                              "south", "south-west", "west", "north-west"};
    int i = int(std::floor(std::fmod(deg + 22.5 + 360.0, 360.0) / 45.0)) % 8;
    return n[i];
}

inline std::string compass_short(double deg) {
    static const char* n[] = {"N", "NE", "E", "SE", "S", "SW", "W", "NW"};
    int i = int(std::floor(std::fmod(deg + 22.5 + 360.0, 360.0) / 45.0)) % 8;
    return n[i];
}

// Scanline rasterisation of a polygon (ring of lat/lon, open or closed) into
// a boolean mask; a cell is inside when its centre is inside the ring.
inline void rasterize_polygon(const Grid& g, const std::vector<LatLon>& ring, std::vector<uint8_t>& mask,
                              uint8_t value = 1) {
    if (ring.size() < 3) return;
    std::vector<std::pair<double, double>> pts;
    for (auto& p : ring) pts.push_back(g.frac_rc(p));
    std::vector<double> xs;
    for (int r = 0; r < g.rows; ++r) {
        double y = r + 0.5;
        xs.clear();
        for (size_t i = 0, j = pts.size() - 1; i < pts.size(); j = i++) {
            double y0 = pts[j].first, y1 = pts[i].first;
            if ((y0 <= y && y1 > y) || (y1 <= y && y0 > y)) {
                double t = (y - y0) / (y1 - y0);
                xs.push_back(pts[j].second + t * (pts[i].second - pts[j].second));
            }
        }
        std::sort(xs.begin(), xs.end());
        for (size_t k = 0; k + 1 < xs.size(); k += 2) {
            int c0 = std::max(0, int(std::ceil(xs[k] - 0.5)));
            int c1 = std::min(g.cols - 1, int(std::floor(xs[k + 1] - 0.5)));
            for (int c = c0; c <= c1; ++c) mask[g.idx(r, c)] = value;
        }
    }
}

// 4-connected ("supercover") rasterisation of a polyline, thickened to
// width_m.  4-connectivity guarantees no diagonal leak through the line.
inline void rasterize_line(const Grid& g, const std::vector<LatLon>& line, double width_m,
                           std::vector<int>& cells_out) {
    std::vector<uint8_t> hit(g.size(), 0);
    int half = std::max(0, int(std::round(0.5 * width_m / g.cell_m - 0.5)));
    auto mark = [&](int r, int c) {
        for (int dr = -half; dr <= half; ++dr)
            for (int dc = -half; dc <= half; ++dc)
                if (g.inside(r + dr, c + dc)) hit[g.idx(r + dr, c + dc)] = 1;
    };
    for (size_t i = 0; i + 1 < line.size(); ++i) {
        auto [r0, c0] = g.frac_rc(line[i]);
        auto [r1, c1] = g.frac_rc(line[i + 1]);
        int n = int(std::ceil(std::max(std::abs(r1 - r0), std::abs(c1 - c0)) * 4)) + 1;
        int pr = int(std::floor(r0)), pc = int(std::floor(c0));
        mark(pr, pc);
        for (int k = 1; k <= n; ++k) {
            double t = double(k) / n;
            int r = int(std::floor(r0 + t * (r1 - r0))), c = int(std::floor(c0 + t * (c1 - c0)));
            if (r != pr && c != pc) mark(pr, c);  // fill the corner → 4-connected
            mark(r, c);
            pr = r;
            pc = c;
        }
    }
    for (size_t i = 0; i < hit.size(); ++i)
        if (hit[i]) cells_out.push_back(int(i));
}

inline void rasterize_disc(const Grid& g, LatLon centre, double radius_m, std::vector<int>& cells_out) {
    int r0, c0;
    g.locate(centre, r0, c0);
    int rad = int(std::ceil(radius_m / g.cell_m));
    for (int dr = -rad; dr <= rad; ++dr)
        for (int dc = -rad; dc <= rad; ++dc) {
            if (!g.inside(r0 + dr, c0 + dc)) continue;
            if (std::hypot(dr, dc) * g.cell_m <= radius_m + 0.5 * g.cell_m) cells_out.push_back(g.idx(r0 + dr, c0 + dc));
        }
}

// Oriented rectangle (length along `bearing`, width across it).
inline void rasterize_strip(const Grid& g, LatLon centre, double bearing, double length_m, double width_m,
                            std::vector<int>& cells_out) {
    int r0, c0;
    g.locate(centre, r0, c0);
    int rad = int(std::ceil(0.5 * std::hypot(length_m, width_m) / g.cell_m)) + 1;
    double ux = std::sin(bearing * kDeg), uy = std::cos(bearing * kDeg);  // east, north
    for (int dr = -rad; dr <= rad; ++dr)
        for (int dc = -rad; dc <= rad; ++dc) {
            if (!g.inside(r0 + dr, c0 + dc)) continue;
            double e = dc * g.cell_m, n = -dr * g.cell_m;
            double along = e * ux + n * uy, across = -e * uy + n * ux;
            if (std::abs(along) <= 0.5 * length_m + 0.5 * g.cell_m &&
                std::abs(across) <= 0.5 * width_m + 0.5 * g.cell_m)
                cells_out.push_back(g.idx(r0 + dr, c0 + dc));
        }
}

}  // namespace rapid
