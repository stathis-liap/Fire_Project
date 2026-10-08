#include "windfield.hpp"

#include <chrono>
#include <cmath>
#include <future>
#include <thread>

namespace rapid {

namespace {

// Matrix-free operator  M λ = Σ_faces h_f (λ − λ_nb) / d²  with λ = 0 in the
// ghost cells outside the domain.  Symmetric positive definite.
struct LayerOperator {
    int R, C;
    double inv_d2;
    std::vector<double> he, hs;  // face depths: east face of (r,c), south face of (r,c)
    std::vector<double> hw_b, hn_b;  // boundary faces (west of col 0 / north of row 0)
    std::vector<double> diag;

    LayerOperator(int R_, int C_, double d, const std::vector<double>& h) : R(R_), C(C_), inv_d2(1.0 / (d * d)) {
        he.assign(size_t(R) * C, 0);
        hs.assign(size_t(R) * C, 0);
        diag.assign(size_t(R) * C, 0);
        auto at = [&](int r, int c) { return h[size_t(r) * C + c]; };
        for (int r = 0; r < R; ++r)
            for (int c = 0; c < C; ++c) {
                size_t i = size_t(r) * C + c;
                he[i] = c + 1 < C ? 0.5 * (at(r, c) + at(r, c + 1)) : at(r, c);
                hs[i] = r + 1 < R ? 0.5 * (at(r, c) + at(r + 1, c)) : at(r, c);
            }
        for (int r = 0; r < R; ++r)
            for (int c = 0; c < C; ++c) {
                size_t i = size_t(r) * C + c;
                double hw = c > 0 ? he[i - 1] : at(r, c);
                double hn = r > 0 ? hs[i - C] : at(r, c);
                diag[i] = (he[i] + hs[i] + hw + hn) * inv_d2;
            }
    }

    void apply(const std::vector<double>& x, std::vector<double>& y) const {
        for (int r = 0; r < R; ++r)
            for (int c = 0; c < C; ++c) {
                size_t i = size_t(r) * C + c;
                double s = diag[i] * x[i];
                if (c + 1 < C) s -= he[i] * inv_d2 * x[i + 1];
                if (c > 0) s -= he[i - 1] * inv_d2 * x[i - 1];
                if (r + 1 < R) s -= hs[i] * inv_d2 * x[i + C];
                if (r > 0) s -= hs[i - C] * inv_d2 * x[i - C];
                y[i] = s;
            }
    }
};

// Jacobi-preconditioned conjugate gradients; returns the iteration count.
int solve_pcg(const LayerOperator& A, const std::vector<double>& b, std::vector<double>& x, double rtol, int max_it) {
    const size_t n = b.size();
    x.assign(n, 0.0);
    std::vector<double> r = b, z(n), p(n), q(n);
    double bn = 0;
    for (double v : b) bn += v * v;
    bn = std::sqrt(bn);
    if (bn <= 0) return 0;
    for (size_t i = 0; i < n; ++i) z[i] = r[i] / A.diag[i];
    p = z;
    double rz = 0;
    for (size_t i = 0; i < n; ++i) rz += r[i] * z[i];
    int it = 0;
    for (; it < max_it; ++it) {
        A.apply(p, q);
        double pq = 0;
        for (size_t i = 0; i < n; ++i) pq += p[i] * q[i];
        if (pq <= 0) break;
        double a = rz / pq;
        double rn = 0;
        for (size_t i = 0; i < n; ++i) {
            x[i] += a * p[i];
            r[i] -= a * q[i];
            rn += r[i] * r[i];
        }
        if (std::sqrt(rn) <= rtol * bn) { ++it; break; }
        for (size_t i = 0; i < n; ++i) z[i] = r[i] / A.diag[i];
        double rz2 = 0;
        for (size_t i = 0; i < n; ++i) rz2 += r[i] * z[i];
        double beta = rz2 / rz;
        rz = rz2;
        for (size_t i = 0; i < n; ++i) p[i] = z[i] + beta * p[i];
    }
    return it;
}

}  // namespace

void build_wind_field(const Grid& g, const std::vector<float>& elev, WindField& W) {
    auto tic = std::chrono::steady_clock::now();
    const int R = g.rows, C = g.cols;
    const size_t N = g.size();
    W = WindField{};
    if (N == 0 || elev.size() != N) return;

    // ── 1. Mass-consistent layer flow on a coarse grid ──
    const int f = std::max(1, int(std::ceil(std::max(R, C) / 160.0)));
    const int Rc = (R + f - 1) / f, Cc = (C + f - 1) / f;
    const double d = g.cell_m * f;
    std::vector<double> zc(size_t(Rc) * Cc, 0.0);
    {
        std::vector<int> cnt(zc.size(), 0);
        for (int r = 0; r < R; ++r)
            for (int c = 0; c < C; ++c) {
                size_t k = size_t(r / f) * Cc + c / f;
                zc[k] += elev[g.idx(r, c)];
                ++cnt[k];
            }
        for (size_t k = 0; k < zc.size(); ++k) zc[k] /= std::max(1, cnt[k]);
        // One 3×3 smoothing pass: the layer top follows the broad terrain,
        // not single DEM pixels.
        std::vector<double> s(zc.size());
        for (int r = 0; r < Rc; ++r)
            for (int c = 0; c < Cc; ++c) {
                double a = 0;
                int n = 0;
                for (int dr = -1; dr <= 1; ++dr)
                    for (int dc = -1; dc <= 1; ++dc) {
                        int rr = r + dr, cc = c + dc;
                        if (rr < 0 || cc < 0 || rr >= Rc || cc >= Cc) continue;
                        a += zc[size_t(rr) * Cc + cc];
                        ++n;
                    }
                s[size_t(r) * Cc + c] = a / n;
            }
        zc.swap(s);
    }
    double zmin = 1e30, zmax = -1e30;
    for (double z : zc) { zmin = std::min(zmin, z); zmax = std::max(zmax, z); }
    // Layer top: high enough that the deepest valley is at most twice as deep
    // as the highest ridge (bounds the ridge speed-up to ≈ 2×).
    const double depth_top = std::max(800.0, zmax - zmin);
    std::vector<double> h(zc.size());
    for (size_t k = 0; k < zc.size(); ++k) h[k] = zmax + depth_top - zc[k];

    LayerOperator A(Rc, Cc, d, h);
    // Right-hand side  div(h u0)  for unit east / north winds (x east, y north;
    // row r+1 is south).
    std::vector<double> bx(zc.size()), by(zc.size());
    for (int r = 0; r < Rc; ++r)
        for (int c = 0; c < Cc; ++c) {
            size_t i = size_t(r) * Cc + c;
            double hE = A.he[i], hW = c > 0 ? A.he[i - 1] : h[i];
            double hS = A.hs[i], hN = r > 0 ? A.hs[i - Cc] : h[i];
            bx[i] = (hE - hW) / d;
            by[i] = (hN - hS) / d;
        }
    std::vector<double> lx, ly;
    int itx = 0, ity = 0;
    {
        auto fx = std::async(std::launch::async, [&] { return solve_pcg(A, bx, lx, 1e-6, 4000); });
        ity = solve_pcg(A, by, ly, 1e-6, 4000);
        itx = fx.get();
    }
    W.cg_iterations = std::max(itx, ity);
    W.coarse_factor = f;

    // Coarse velocities: u = u0 + ∇λ (central differences, λ = 0 outside).
    std::vector<float> cxu(zc.size()), cxv(zc.size()), cyu(zc.size()), cyv(zc.size());
    auto grad = [&](const std::vector<double>& l, int r, int c, double& gx, double& gy) {
        auto L = [&](int rr, int cc) { return (rr < 0 || cc < 0 || rr >= Rc || cc >= Cc) ? 0.0 : l[size_t(rr) * Cc + cc]; };
        gx = (L(r, c + 1) - L(r, c - 1)) / (2 * d);
        gy = (L(r - 1, c) - L(r + 1, c)) / (2 * d);  // north = row − 1
    };
    for (int r = 0; r < Rc; ++r)
        for (int c = 0; c < Cc; ++c) {
            size_t i = size_t(r) * Cc + c;
            double gx, gy;
            grad(lx, r, c, gx, gy);
            cxu[i] = float(1.0 + gx);
            cxv[i] = float(gy);
            grad(ly, r, c, gx, gy);
            cyu[i] = float(gx);
            cyv[i] = float(1.0 + gy);
        }
    // Bilinear interpolation back to the full grid (coarse cell centres).
    W.exu.resize(N);
    W.exv.resize(N);
    W.eyu.resize(N);
    W.eyv.resize(N);
    for (int r = 0; r < R; ++r) {
        double fr = std::clamp((r + 0.5) / f - 0.5, 0.0, double(Rc - 1));
        int r0 = int(fr), r1 = std::min(r0 + 1, Rc - 1);
        double wr = fr - r0;
        for (int c = 0; c < C; ++c) {
            double fc = std::clamp((c + 0.5) / f - 0.5, 0.0, double(Cc - 1));
            int c0 = int(fc), c1 = std::min(c0 + 1, Cc - 1);
            double wc = fc - c0;
            auto bl = [&](const std::vector<float>& a) {
                return float((1 - wr) * ((1 - wc) * a[size_t(r0) * Cc + c0] + wc * a[size_t(r0) * Cc + c1]) +
                             wr * ((1 - wc) * a[size_t(r1) * Cc + c0] + wc * a[size_t(r1) * Cc + c1]));
            };
            size_t k = size_t(g.idx(r, c));
            W.exu[k] = bl(cxu);
            W.exv[k] = bl(cxv);
            W.eyu[k] = bl(cyu);
            W.eyv[k] = bl(cyv);
        }
    }

    // ── 2. Lee shelter from the upwind horizon angle, per direction ──
    const int D = WindField::kDirs;
    W.shelter.assign(size_t(D) * N, 255);
    const double dmax = 1000.0;
    const double step = std::max(g.cell_m, dmax / 40.0);
    const int K = std::max(1, int(dmax / step));
    unsigned nt = std::max(1u, std::min<unsigned>(std::thread::hardware_concurrency(), unsigned(D)));
    std::vector<std::thread> pool;
    for (unsigned w = 0; w < nt; ++w)
        pool.emplace_back([&, w] {
            for (int b = int(w); b < D; b += int(nt)) {
                double from = b * 360.0 / D;
                // Upwind is towards where the wind comes FROM.
                double sdc = std::sin(from * kDeg), sdr = -std::cos(from * kDeg);
                std::vector<int> odr(K), odc(K);
                std::vector<double> inv_dist(K);
                for (int k = 1; k <= K; ++k) {
                    odr[k - 1] = int(std::lround(sdr * k * step / g.cell_m));
                    odc[k - 1] = int(std::lround(sdc * k * step / g.cell_m));
                    inv_dist[k - 1] = 1.0 / (k * step);
                }
                uint8_t* out = &W.shelter[size_t(b) * N];
                for (int r = 0; r < R; ++r)
                    for (int c = 0; c < C; ++c) {
                        const double z0 = elev[g.idx(r, c)];
                        double best = -1e9;
                        for (int k = 0; k < K; ++k) {
                            int rr = r + odr[k], cc = c + odc[k];
                            if (rr < 0 || cc < 0 || rr >= R || cc >= C) break;
                            best = std::max(best, (elev[g.idx(rr, cc)] - z0) * inv_dist[k]);
                        }
                        if (best <= 0) continue;
                        double sx = std::atan(best) / kDeg;  // degrees above the horizon
                        double t = std::clamp((sx - 4.0) / 18.0, 0.0, 1.0);
                        double fct = 1.0 - 0.55 * t * t * (3 - 2 * t);  // smoothstep to 0.45
                        out[g.idx(r, c)] = uint8_t(std::lround(255.0 * fct));
                    }
            }
        });
    for (auto& t : pool) t.join();
    W.runtime_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tic).count();
}

ThermalHour thermal_hour(double epoch, double lon, double ambient_ms) {
    ThermalHour th;
    // Local solar time from UTC and longitude.
    double hs = std::fmod(epoch / 3600.0 + lon / 15.0 + 48.0, 24.0);
    double sun = std::sin(kPi * (hs - 6.0) / 12.0);  // > 0 from 06 to 18 solar time
    double az = 90.0 + 15.0 * (hs - 6.0);            // east at sunrise, south at noon, west at sunset
    th.sun_sin = std::sin(az * kDeg);
    th.sun_cos = std::cos(az * kDeg);
    // Anabatic up to ~2 m/s on sunlit slopes, katabatic ~1.2 m/s at night
    // (lagging an hour after sunset).  Both fade once the ambient wind
    // dominates (≈ half strength at 6 m/s).
    double s = sun > 0 ? 2.0 * sun : 0.0;
    double night = std::sin(kPi * (hs - 7.0) / 12.0);
    if (sun <= 0 || night < -0.2) s = night < 0 ? 1.2 * std::max(-1.0, night) : 0.0;
    th.strength = s / (1.0 + (ambient_ms / 6.0) * (ambient_ms / 6.0));
    return th;
}

}  // namespace rapid
