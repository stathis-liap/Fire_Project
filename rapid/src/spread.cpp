#include "spread.hpp"

#include <algorithm>
#include <cmath>
#include <queue>
#include <thread>

namespace rapid {

namespace {

struct Offset {
    int dr, dc;
    double dist;     // in cells
    double bearing;  // compass degrees
    double sb, cb;   // sin/cos of bearing
    int n_via;
    int via_dr[2], via_dc[2];  // cells the segment passes through
};

const std::vector<Offset>& offsets16() {
    static const std::vector<Offset> o = [] {
        std::vector<Offset> v;
        for (int dr = -2; dr <= 2; ++dr)
            for (int dc = -2; dc <= 2; ++dc) {
                int a = std::abs(dr), b = std::abs(dc);
                if ((a == 0 && b == 0) || (a == 2 && b != 1) || (b == 2 && a != 1)) continue;
                double brg = std::fmod(std::atan2(dc, -dr) / kDeg + 360.0, 360.0);
                Offset x{dr, dc, std::hypot(dr, dc), brg, std::sin(brg * kDeg), std::cos(brg * kDeg), 0, {}, {}};
                if (a == 1 && b == 1) {
                    x.n_via = 2;
                    x.via_dr[0] = dr; x.via_dc[0] = 0;
                    x.via_dr[1] = 0;  x.via_dc[1] = dc;
                } else if (a == 1 && b == 2) {
                    x.n_via = 2;
                    x.via_dr[0] = 0;  x.via_dc[0] = dc / 2;
                    x.via_dr[1] = dr; x.via_dc[1] = dc / 2;
                } else if (a == 2 && b == 1) {
                    x.n_via = 2;
                    x.via_dr[0] = dr / 2; x.via_dc[0] = 0;
                    x.via_dr[1] = dr / 2; x.via_dc[1] = dc;
                }
                v.push_back(x);
            }
        return v;
    }();
    return o;
}

constexpr int kMoistClasses = 3;
constexpr double kMoistAdj[kMoistClasses] = {0.0, -0.01, +0.015};

double length_to_breadth(double eff_wind_mph) {
    double u = std::max(0.0, eff_wind_mph);
    double lb = 0.936 * std::exp(0.2566 * u) + 0.461 * std::exp(-0.1548 * u) - 0.397;
    return std::clamp(lb, 1.0, 8.0);
}

// Hourly weather resolved for one run (params applied).
struct HourWx {
    double u10_ftmin;  // 10 m wind speed (ft/min) after wind_mult
    double wind_to;    // direction the wind blows towards (deg)
    std::vector<RothermelBase> base;  // [fuel * kMoistClasses + mclass]
};

struct Ellipse {
    float rmax = 0;    // m/min, head-fire spread rate
    float theta = 0;   // compass direction of max spread (deg)
    float st = 0, ct = 1;  // sin/cos of theta
    float ecc = 0;     // eccentricity
    float k_int = 0;   // Byram intensity per unit ROS: I_B[BTU/ft/s] = k_int * R[ft/min]
    float crown_w = 0; // >0 → crown fire; total fuel consumed (kg/m²) for intensity
};

class RunCtx {
   public:
    RunCtx(const Landscape& L, const WeatherSeries& wx, const std::vector<RothermelFuel>& rf, const SpreadParams& p,
           double t0_epoch, double t_end)
        : L_(L), rf_(rf), p_(p) {
        int K = std::max(1, int(std::ceil(std::max(0.0, t_end) / 60.0)) + 1);
        hours_.resize(K);
        const auto& ft = fuel_table();
        for (int k = 0; k < K; ++k) {
            WxSample s = wx.at(t0_epoch + k * 3600.0 + 1800.0);
            HourWx& h = hours_[k];
            h.u10_ftmin = std::max(0.0, s.wind_ms * p.wind_mult) * kMsToFtMin;
            h.wind_to = std::fmod(s.wind_from + 180.0 + p.wind_dir_offset + 720.0, 360.0);
            double mf0 = emc_fraction(s.temp_c, s.rh) + p.moisture_offset;
            h.base.resize(ft.size() * kMoistClasses);
            for (size_t f = 0; f < ft.size(); ++f)
                for (int m = 0; m < kMoistClasses; ++m)
                    h.base[f * kMoistClasses + m] =
                        rothermel_base(rf[f], std::clamp(mf0 + kMoistAdj[m], 0.02, 0.45));
        }
        cache_.resize(L.grid.size());
        cache_hour_.assign(L.grid.size(), -1);
    }

    int hour_of(double t) const { return std::clamp(int(std::floor(t / 60.0)), 0, int(hours_.size()) - 1); }

    const Ellipse& ellipse(int cell, int k) {
        if (cache_hour_[cell] == k) return cache_[cell];
        cache_hour_[cell] = k;
        Ellipse& e = cache_[cell];
        e = compute(cell, k);
        return e;
    }

    Ellipse compute(int cell, int k) const {
        Ellipse e;
        int f = L_.fuel[cell];
        const RothermelFuel& rf = rf_[f];
        if (!rf.ok) return e;
        const HourWx& h = hours_[k];
        const RothermelBase& b = h.base[f * kMoistClasses + std::clamp<int>(L_.moist_adj[cell], 0, kMoistClasses - 1)];
        if (b.r0_ftmin <= 0) return e;

        double u = h.u10_ftmin * rf.waf * L_.wind_mult[cell];
        u = std::min(u, 96.8 * std::cbrt(b.ir));  // wind limit (Andrews et al. 2013)
        double phi_w = u > 0 ? rf.C * std::pow(u, rf.B) * std::pow(rf.rel_beta, -rf.E) : 0.0;
        double tan_s = L_.slope_tan[cell];
        double phi_s = rf.slope_k * tan_s * tan_s;

        double wx = phi_w * std::sin(h.wind_to * kDeg) + phi_s * std::sin(L_.upslope[cell] * kDeg);
        double wy = phi_w * std::cos(h.wind_to * kDeg) + phi_s * std::cos(L_.upslope[cell] * kDeg);
        double phi_e = std::hypot(wx, wy);
        e.theta = float(phi_e > 1e-9 ? std::fmod(std::atan2(wx, wy) / kDeg + 360.0, 360.0) : 0.0);
        e.st = float(std::sin(e.theta * kDeg));
        e.ct = float(std::cos(e.theta * kDeg));

        double r_ftmin = b.r0_ftmin * (1.0 + phi_e) * p_.ros_mult;
        e.rmax = float(r_ftmin * kFtMinToMMin);
        double ue = phi_e > 0 ? std::pow(phi_e * std::pow(rf.rel_beta, rf.E) / rf.C, 1.0 / rf.B) : 0.0;
        double lb = length_to_breadth(ue / 88.0);
        e.k_int = float(b.ir * rf.t_res_min / 60.0);

        // Crown fire where the surface fire is intense enough to reach the canopy.
        if (rf.cbh > 0) {
            double surface_kwm = 3.4613 * e.k_int * r_ftmin;
            double u10_kmh = h.u10_ftmin / kMsToFtMin * 3.6 * L_.wind_mult[cell];
            CrownResult cr = crown_fire(rf, surface_kwm, u10_kmh, b.mf);
            double r_crown = cr.ros_mmin * p_.ros_mult;
            if (cr.crowning && r_crown > e.rmax) {
                e.rmax = float(r_crown);
                // Rothermel (1991): crown fire length-to-breadth from 20-ft wind.
                lb = std::max(lb, std::min(8.0, 1.0 + 0.125 * (u10_kmh / 1.15 / 1.609)));
                constexpr double canopy_depth_m = 8.0;
                e.crown_w = float(rf.w0_kgm2 + cr.fraction * rf.cbd * canopy_depth_m);
            }
        }
        e.ecc = float(std::sqrt(lb * lb - 1.0) / lb);
        return e;
    }

    static double ros_dir(const Ellipse& e, double bearing) {
        return ros_dir(e, std::sin(bearing * kDeg), std::cos(bearing * kDeg));
    }
    // cos(bearing - theta) from precomputed sines/cosines — no trig per edge.
    static double ros_dir(const Ellipse& e, double sb, double cb) {
        if (e.rmax <= 0) return 0;
        double c = cb * e.ct + sb * e.st;
        return e.rmax * (1.0 - e.ecc) / (1.0 - e.ecc * c);
    }

    static double flame_m(const Ellipse& e, double ros_mmin) {
        if (e.crown_w > 0) {  // Thomas (1963) crown flame length from Byram intensity
            double i_kwm = 18000.0 / 60.0 * e.crown_w * ros_mmin;
            return 0.0266 * std::pow(i_kwm, 2.0 / 3.0);
        }
        double ib = e.k_int * ros_mmin / kFtMinToMMin;  // BTU/ft/s
        return ib > 0 ? 0.3048 * 0.45 * std::pow(ib, 0.46) : 0.0;
    }

   private:
    const Landscape& L_;
    const std::vector<RothermelFuel>& rf_;
    SpreadParams p_;
    std::vector<HourWx> hours_;
    std::vector<Ellipse> cache_;
    std::vector<int16_t> cache_hour_;
};

// Spread-rate factor an intervention layer applies at time t, given the
// flame length the crew/aircraft would face.  Returns 0 for an intact break.
inline double layer_factor(const InterventionRaster::Layer& l, bool breached, double t, double flame) {
    if (t < l.t_on || t > l.t_off) return 1.0;
    switch (l.kind) {
        case IvKind::Firebreak:
            return breached ? 0.5 : 0.0;
        case IvKind::AirDrop:
            // Retardant line: strong effect unless the fire is extreme.
            return flame > 10 ? 0.5 : 0.12;
        case IvKind::Truck:
            // Engines hold the line up to ~3.4 m flames (fireline handbook
            // "hauling chart"), struggle beyond, and are overrun past ~6 m.
            return flame < 3.4 ? 0.15 : flame < 6 ? 0.5 : 0.9;
    }
    return 1.0;
}

}  // namespace

void InterventionRaster::build(const Grid& g, const std::vector<Intervention>& ivs) {
    layers.clear();
    mask.assign(g.size(), 0);
    for (size_t i = 0; i < ivs.size(); ++i) {
        const Intervention& iv = ivs[i];
        if (layers.size() >= 32) break;
        Layer l;
        l.src = int(i);
        l.kind = iv.kind;
        l.t_on = float(iv.t_on);
        l.t_off = float(iv.t_off);
        if (iv.kind == IvKind::Firebreak) {
            rasterize_line(g, iv.pts, std::max(g.cell_m, iv.width_m), l.cells);
        } else if (iv.kind == IvKind::AirDrop && !iv.pts.empty()) {
            rasterize_strip(g, iv.pts[0], iv.bearing, iv.length_m, std::max(iv.width_m, g.cell_m), l.cells);
        } else if (!iv.pts.empty()) {
            rasterize_disc(g, iv.pts[0], iv.radius_m, l.cells);
        }
        if (l.cells.empty()) continue;
        uint32_t bit = 1u << layers.size();
        for (int c : l.cells) mask[c] |= bit;
        layers.push_back(std::move(l));
    }
}

SpreadSolver::SpreadSolver(const Landscape& L, const WeatherSeries& wx) : L_(L), wx_(wx) {
    for (const auto& f : fuel_table()) rf_.push_back(rothermel_prepare_stand(f));
}

double SpreadSolver::head_ros(int cell, double t0_epoch, double t_min, const SpreadParams& p, double* flame,
                              double* dir) const {
    RunCtx ctx(L_, wx_, rf_, p, t0_epoch, t_min + 60);
    Ellipse e = ctx.compute(cell, ctx.hour_of(t_min));
    if (flame) *flame = RunCtx::flame_m(e, e.rmax);
    if (dir) *dir = e.theta;
    return e.rmax;
}

void SpreadSolver::run(const RunConfig& cfg, RunOutput& out) const {
    const Grid& g = L_.grid;
    const size_t N = g.size();
    RunCtx ctx(L_, wx_, rf_, cfg.p, cfg.t0_epoch, cfg.t_end);
    const auto& offs = offsets16();
    const auto& ft = fuel_table();
    const InterventionRaster* iv = (cfg.iv && !cfg.iv->empty()) ? cfg.iv : nullptr;

    out.arrival.assign(N, kInf);
    if (cfg.detail) {
        out.pred.assign(N, -1);
        out.flame_m.assign(N, 0.f);
        out.ros.assign(N, 0.f);
    }
    std::vector<uint8_t> done(N, 0);

    auto factor_at = [&](int cell, double t, double flame) -> double {
        if (!iv) return 1.0;
        uint32_t m = iv->mask[cell];
        double f = 1.0;
        for (int i = 0; m; ++i, m >>= 1)
            if (m & 1u) f *= layer_factor(iv->layers[i], (cfg.breached >> i) & 1u, t, flame);
        return f;
    };
    auto is_barrier = [&](int cell, double t) -> bool {
        if (!ft[L_.fuel[cell]].burnable()) return true;
        if (!iv) return false;
        uint32_t m = iv->mask[cell];
        for (int i = 0; m; ++i, m >>= 1)
            if ((m & 1u) && iv->layers[i].kind == IvKind::Firebreak && !((cfg.breached >> i) & 1u) &&
                t >= iv->layers[i].t_on && t <= iv->layers[i].t_off)
                return true;
        return false;
    };

    using QE = std::pair<float, int>;
    std::priority_queue<QE, std::vector<QE>, std::greater<QE>> pq;
    for (auto [cell, t] : cfg.sources) {
        if (cell < 0 || size_t(cell) >= N) continue;
        if (t < out.arrival[cell]) {
            out.arrival[cell] = t;
            pq.push({t, cell});
        }
    }

    while (!pq.empty()) {
        auto [t, a] = pq.top();
        pq.pop();
        if (done[a] || t > out.arrival[a]) continue;
        if (t > cfg.t_end) break;
        done[a] = 1;
        int k = ctx.hour_of(t);
        const Ellipse ea = ctx.ellipse(a, k);
        if (ea.rmax <= 0) continue;
        int ar = a / g.cols, ac = a % g.cols;
        const bool iv_a = iv && iv->mask[a];
        double fa = iv_a ? factor_at(a, t, RunCtx::flame_m(ea, ea.rmax)) : 1.0;
        if (fa <= 0) continue;

        for (const auto& o : offs) {
            int br = ar + o.dr, bc = ac + o.dc;
            if (!g.inside(br, bc)) continue;
            int b = g.idx(br, bc);
            if (done[b]) continue;
            if (!ft[L_.fuel[b]].burnable()) continue;
            bool blocked = false;
            for (int v = 0; v < o.n_via && !blocked; ++v)
                blocked = is_barrier(g.idx(ar + o.via_dr[v], ac + o.via_dc[v]), t);
            if (blocked) continue;

            const Ellipse& eb = ctx.ellipse(b, k);
            double ra = RunCtx::ros_dir(ea, o.sb, o.cb) * fa;
            double rb = RunCtx::ros_dir(eb, o.sb, o.cb);
            if (iv && iv->mask[b]) rb *= factor_at(b, t, RunCtx::flame_m(eb, rb));
            if (ra <= 1e-4 || rb <= 1e-4) continue;
            double d = o.dist * g.cell_m;
            double dt = 0.5 * d / ra + 0.5 * d / rb;
            float nt = float(t + dt);
            if (nt < out.arrival[b] && nt <= cfg.t_end) {
                out.arrival[b] = nt;
                if (cfg.detail) {
                    out.pred[b] = a;
                    out.ros[b] = float(d / dt);
                    out.flame_m[b] = float(o.bearing);  // placeholder: bearing, converted below
                }
                pq.push({nt, b});
            }
        }
    }
    if (cfg.detail) {
        // Flame length at arrival, evaluated once per cell for the winning path.
        for (size_t b = 0; b < N; ++b) {
            if (out.pred[b] < 0 || out.arrival[b] >= kInf) continue;
            const Ellipse e = ctx.compute(int(b), ctx.hour_of(out.arrival[b]));
            out.flame_m[b] = float(RunCtx::flame_m(e, RunCtx::ros_dir(e, out.flame_m[b])));
        }
    }
    if (cfg.detail)
        for (auto [cell, t] : cfg.sources)
            if (cell >= 0 && size_t(cell) < N && out.pred[cell] < 0) {
                const Ellipse e = ctx.compute(cell, ctx.hour_of(t));
                out.ros[cell] = e.rmax;
                out.flame_m[cell] = float(RunCtx::flame_m(e, e.rmax));
            }
}

int SpreadSolver::hazard(double t0_epoch, double t_from, double t_to, const SpreadParams& p, std::vector<float>& ros_kmh,
                         std::vector<float>& flame_m, std::vector<float>& dir_deg) const {
    const size_t N = L_.grid.size();
    RunCtx ctx(L_, wx_, rf_, p, t0_epoch, t_to + 60);
    ros_kmh.assign(N, 0.f);
    flame_m.assign(N, 0.f);
    dir_deg.assign(N, 0.f);
    int k0 = ctx.hour_of(std::max(0.0, t_from)), k1 = ctx.hour_of(t_to);
    const int H = k1 - k0 + 1;
    std::vector<double> hour_sum(H, 0.0);
    // Cells are independent: split them across threads (compute() is const).
    unsigned nt = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::vector<double>> part(nt, std::vector<double>(H, 0.0));
    std::vector<std::thread> pool;
    for (unsigned w = 0; w < nt; ++w)
        pool.emplace_back([&, w] {
            for (size_t c = w; c < N; c += nt) {
                if (!rf_[L_.fuel[c]].ok) continue;
                for (int k = k0; k <= k1; ++k) {
                    Ellipse e = ctx.compute(int(c), k);
                    double v = e.rmax * 0.06;  // m/min → km/h
                    part[w][k - k0] += v;
                    if (v > ros_kmh[c]) {
                        ros_kmh[c] = float(v);
                        flame_m[c] = float(RunCtx::flame_m(e, e.rmax));
                        dir_deg[c] = e.theta;
                    }
                }
            }
        });
    for (auto& t : pool) t.join();
    for (auto& p : part)
        for (int h = 0; h < H; ++h) hour_sum[h] += p[h];
    int peak = k0;
    for (int h = 0; h < H; ++h)
        if (hour_sum[h] > hour_sum[peak - k0]) peak = k0 + h;
    return peak;
}

void SpreadSolver::time_to_targets(double t0_epoch, double t_at, const SpreadParams& p,
                                   const std::vector<std::pair<int, int>>& targets, float t_max,
                                   std::vector<float>& minutes, std::vector<int>& label) const {
    const Grid& g = L_.grid;
    const size_t N = g.size();
    RunCtx ctx(L_, wx_, rf_, p, t0_epoch, t_at + 60);
    const int k = ctx.hour_of(t_at);
    const auto& offs = offsets16();
    const auto& ft = fuel_table();
    minutes.assign(N, kInf);
    label.assign(N, -1);
    std::vector<uint8_t> is_target(N, 0), done(N, 0);
    using QE = std::pair<float, int>;
    std::priority_queue<QE, std::vector<QE>, std::greater<QE>> pq;
    for (auto [c, l] : targets) {
        if (c < 0 || size_t(c) >= N) continue;
        is_target[c] = 1;
        if (minutes[c] > 0) { minutes[c] = 0; label[c] = l; pq.push({0.f, c}); }
    }
    // Fire moves a → b; we know b's time to the targets and look for a.
    while (!pq.empty()) {
        auto [t, b] = pq.top();
        pq.pop();
        if (done[b] || t > minutes[b]) continue;
        if (t > t_max) break;
        done[b] = 1;
        const int br = b / g.cols, bc = b % g.cols;
        const bool b_burns = ft[L_.fuel[b]].burnable();
        if (!b_burns && !is_target[b]) continue;
        for (const auto& o : offs) {
            int ar = br - o.dr, ac = bc - o.dc;
            if (!g.inside(ar, ac)) continue;
            int a = g.idx(ar, ac);
            if (done[a] || !ft[L_.fuel[a]].burnable()) continue;
            bool blocked = false;
            for (int v = 0; v < o.n_via && !blocked; ++v)
                blocked = !ft[L_.fuel[g.idx(ar + o.via_dr[v], ac + o.via_dc[v])]].burnable();
            if (blocked) continue;
            double ra = RunCtx::ros_dir(ctx.ellipse(a, k), o.sb, o.cb);
            if (ra <= 1e-4) continue;
            double d = o.dist * g.cell_m;
            double rb = b_burns ? RunCtx::ros_dir(ctx.ellipse(b, k), o.sb, o.cb) : 0.0;
            double dt = rb > 1e-4 ? 0.5 * d / ra + 0.5 * d / rb : d / ra;
            float nt = float(t + dt);
            if (nt < minutes[a] && nt <= t_max) {
                minutes[a] = nt;
                label[a] = label[b];
                pq.push({nt, a});
            }
        }
    }
}

int nearest_burnable(const Landscape& L, int r, int c, int max_r) {
    const auto& ft = fuel_table();
    const Grid& g = L.grid;
    int best = -1;
    double bd = 1e9;
    for (int dr = -max_r; dr <= max_r; ++dr)
        for (int dc = -max_r; dc <= max_r; ++dc) {
            if (!g.inside(r + dr, c + dc)) continue;
            int i = g.idx(r + dr, c + dc);
            double d = std::hypot(dr, dc);
            if (ft[L.fuel[i]].burnable() && rothermel_prepare_stand(ft[L.fuel[i]]).ok && d < bd) {
                bd = d;
                best = i;
            }
        }
    return best;
}

}  // namespace rapid
