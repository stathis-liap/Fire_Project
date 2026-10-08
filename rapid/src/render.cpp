#include "render.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <unordered_map>

#include "stb_image_write.h"

namespace rapid {

std::string base64_encode(const std::string& in) {
    static const char* tbl = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((in.size() + 2) / 3 * 4);
    size_t i = 0;
    for (; i + 2 < in.size(); i += 3) {
        uint32_t v = (uint8_t(in[i]) << 16) | (uint8_t(in[i + 1]) << 8) | uint8_t(in[i + 2]);
        out += tbl[v >> 18];
        out += tbl[(v >> 12) & 63];
        out += tbl[(v >> 6) & 63];
        out += tbl[v & 63];
    }
    if (i < in.size()) {
        uint32_t v = uint8_t(in[i]) << 16;
        if (i + 1 < in.size()) v |= uint8_t(in[i + 1]) << 8;
        out += tbl[v >> 18];
        out += tbl[(v >> 12) & 63];
        out += i + 1 < in.size() ? tbl[(v >> 6) & 63] : '=';
        out += '=';
    }
    return out;
}

std::string render_zones_png(const Grid& g, const std::vector<float>& p50, const std::vector<float>& p10, float t_now,
                             float t_end) {
    struct C { uint8_t r, g, b, a; };
    const C burned{45, 25, 25, 175}, h1{225, 30, 30, 170}, h2{245, 115, 20, 160}, h3{250, 185, 35, 150},
        later{250, 232, 90, 135}, possible{250, 232, 90, 60};
    std::vector<uint8_t> px(g.size() * 4, 0);
    for (size_t i = 0; i < g.size(); ++i) {
        C c{0, 0, 0, 0};
        float a = p50[i];
        if (a <= t_now) c = burned;
        else if (a <= t_now + 60) c = h1;
        else if (a <= t_now + 120) c = h2;
        else if (a <= t_now + 180) c = h3;
        else if (a <= t_end) c = later;
        else if (p10[i] <= t_end) c = possible;
        px[4 * i] = c.r;
        px[4 * i + 1] = c.g;
        px[4 * i + 2] = c.b;
        px[4 * i + 3] = c.a;
    }
    std::string png;
    stbi_write_png_to_func(
        [](void* ctx, void* data, int size) { static_cast<std::string*>(ctx)->append(static_cast<char*>(data), size); },
        &png, g.cols, g.rows, 4, px.data(), g.cols * 4);
    return "data:image/png;base64," + base64_encode(png);
}

namespace {
std::string encode_png(const std::vector<uint8_t>& px, int w, int h) {
    std::string png;
    stbi_write_png_to_func(
        [](void* ctx, void* data, int size) { static_cast<std::string*>(ctx)->append(static_cast<char*>(data), size); },
        &png, w, h, 4, px.data(), w * 4);
    return "data:image/png;base64," + base64_encode(png);
}
}  // namespace

std::vector<std::string> render_prob_pngs(const Grid& g, const std::vector<std::vector<float>>& arrivals,
                                          const std::vector<float>& times) {
    const size_t N = g.size(), M = arrivals.size();
    const int F = int(times.size());
    std::vector<std::vector<uint8_t>> px(F, std::vector<uint8_t>(N * 4, 0));
    if (!M) return std::vector<std::string>(F);
    // Colour ramp: pale yellow (unlikely) → orange → red → dark red (certain).
    static const float stops[5][3] = {{255, 236, 120}, {255, 186, 60}, {248, 118, 32}, {222, 44, 30}, {130, 0, 28}};
    std::vector<int> hist(F + 1);
    for (size_t c = 0; c < N; ++c) {
        std::fill(hist.begin(), hist.end(), 0);
        bool any = false;
        for (size_t m = 0; m < M; ++m) {
            float a = arrivals[m][c];
            if (a >= kInf) continue;
            // First frame by which this member has reached the cell.
            int b = int(std::lower_bound(times.begin(), times.end(), a) - times.begin());
            if (b < F) { ++hist[b]; any = true; }
        }
        if (!any) continue;
        int cum = 0;
        for (int h = 0; h < F; ++h) {
            cum += hist[h];
            double p = double(cum) / M;
            if (p < 0.03) continue;
            double x = p * 4.0;
            int i = std::min(3, int(x));
            double f = x - i;
            uint8_t* q = &px[h][4 * c];
            for (int k = 0; k < 3; ++k) q[k] = uint8_t(stops[i][k] + f * (stops[i + 1][k] - stops[i][k]));
            q[3] = uint8_t(60 + 165 * p);
        }
    }
    std::vector<std::string> out;
    for (int h = 0; h < F; ++h) out.push_back(encode_png(px[h], g.cols, g.rows));
    return out;
}

std::string render_danger_png(const Grid& g, const std::vector<uint8_t>& cls) {
    static const uint8_t col[6][4] = {{0, 0, 0, 0},         {56, 170, 92, 120},  {236, 214, 62, 140},
                                      {246, 146, 36, 160},  {226, 42, 34, 175},  {128, 28, 150, 190}};
    std::vector<uint8_t> px(g.size() * 4);
    for (size_t i = 0; i < g.size(); ++i)
        for (int k = 0; k < 4; ++k) px[4 * i + k] = col[std::min<int>(cls[i], 5)][k];
    return encode_png(px, g.cols, g.rows);
}

std::vector<std::vector<LatLon>> contour_lines(const Grid& g, const std::vector<float>& field, float level) {
    // Work on cell centres; values beyond the level (incl. kInf) are "outside".
    struct P { double r, c; };
    std::vector<std::pair<P, P>> segs;
    auto val = [&](int r, int c) { return std::min(field[g.idx(r, c)], 1e7f); };
    auto lerp = [&](double r0, double c0, float v0, double r1, double c1, float v1) {
        double t = (level - v0) / double(v1 - v0);
        return P{r0 + t * (r1 - r0), c0 + t * (c1 - c0)};
    };
    for (int r = 0; r + 1 < g.rows; ++r)
        for (int c = 0; c + 1 < g.cols; ++c) {
            float v[4] = {val(r, c), val(r, c + 1), val(r + 1, c + 1), val(r + 1, c)};
            int code = (v[0] <= level) | (v[1] <= level) << 1 | (v[2] <= level) << 2 | (v[3] <= level) << 3;
            if (code == 0 || code == 15) continue;
            P e[4] = {lerp(r, c, v[0], r, c + 1, v[1]), lerp(r, c + 1, v[1], r + 1, c + 1, v[2]),
                      lerp(r + 1, c + 1, v[2], r + 1, c, v[3]), lerp(r + 1, c, v[3], r, c, v[0])};
            auto add = [&](int a, int b) { segs.push_back({e[a], e[b]}); };
            switch (code) {
                case 1: case 14: add(3, 0); break;
                case 2: case 13: add(0, 1); break;
                case 3: case 12: add(3, 1); break;
                case 4: case 11: add(1, 2); break;
                case 6: case 9: add(0, 2); break;
                case 7: case 8: add(3, 2); break;
                case 5: add(3, 0); add(1, 2); break;
                case 10: add(0, 1); add(2, 3); break;
            }
        }

    // Join segments sharing endpoints into polylines.
    auto key = [](const P& p) { return (long long)std::llround(p.r * 1000) * 100000000LL + std::llround(p.c * 1000); };
    std::unordered_multimap<long long, size_t> at;
    for (size_t i = 0; i < segs.size(); ++i) {
        at.insert({key(segs[i].first), i});
        at.insert({key(segs[i].second), i});
    }
    std::vector<uint8_t> used(segs.size(), 0);
    auto take_next = [&](const P& end) -> int {
        auto range = at.equal_range(key(end));
        for (auto it = range.first; it != range.second; ++it)
            if (!used[it->second]) return int(it->second);
        return -1;
    };
    std::vector<std::vector<LatLon>> lines;
    auto to_ll = [&](const P& p) { return LatLon{g.north - (p.r + 0.5) * g.dlat, g.west + (p.c + 0.5) * g.dlon}; };
    for (size_t s = 0; s < segs.size(); ++s) {
        if (used[s]) continue;
        used[s] = 1;
        std::vector<P> chain = {segs[s].first, segs[s].second};
        for (int dir = 0; dir < 2; ++dir) {
            while (true) {
                const P& end = dir == 0 ? chain.back() : chain.front();
                int n = take_next(end);
                if (n < 0) break;
                used[n] = 1;
                const P& nxt = key(segs[n].first) == key(end) ? segs[n].second : segs[n].first;
                if (dir == 0) chain.push_back(nxt);
                else chain.insert(chain.begin(), nxt);
            }
        }
        if (chain.size() < 4) continue;  // drop specks
        std::vector<LatLon> line;
        for (size_t i = 0; i < chain.size(); ++i)
            if (i % 2 == 0 || i + 1 == chain.size()) line.push_back(to_ll(chain[i]));  // light thinning
        lines.push_back(std::move(line));
    }
    return lines;
}

json geo_point(LatLon p) { return {{"type", "Point"}, {"coordinates", {p.lon, p.lat}}}; }

json geo_line(const std::vector<LatLon>& pts) {
    json c = json::array();
    for (auto& p : pts) c.push_back({std::round(p.lon * 1e6) / 1e6, std::round(p.lat * 1e6) / 1e6});
    return {{"type", "LineString"}, {"coordinates", c}};
}

json feature(const json& geometry, const json& props) {
    return {{"type", "Feature"}, {"geometry", geometry}, {"properties", props}};
}

}  // namespace rapid
