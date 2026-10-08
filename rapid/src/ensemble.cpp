#include "ensemble.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <random>
#include <thread>

namespace rapid {

namespace {

// Inverse of the standard normal CDF (Acklam 2003, relative error < 1.2e-9).
double norm_inv(double p) {
    static const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
                               1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00};
    static const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
                               6.680131188771972e+01, -1.328068155288572e+01};
    static const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
                               -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00};
    static const double d[] = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
                               3.754408661907416e+00};
    p = std::clamp(p, 1e-9, 1.0 - 1e-9);
    if (p < 0.02425) {
        double q = std::sqrt(-2 * std::log(p));
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    }
    if (p > 1 - 0.02425) return -norm_inv(1 - p);
    double q = p - 0.5, r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

}  // namespace

std::vector<Member> make_members(const SpreadParams& base, const Uncertainty& u, int n, uint32_t seed,
                                 const InterventionRaster* iv) {
    std::vector<Member> out;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    // Latin hypercube over the four perturbations: every member gets its own
    // probability stratum in each input, so 24–64 members already cover the
    // tails evenly instead of clustering by chance.  Member 0 stays the
    // unperturbed best guess.
    const int m = std::max(0, n - 1);
    std::vector<std::array<double, 4>> z(size_t(std::max(n, 1)));
    for (int k = 0; k < 4; ++k) {
        std::vector<int> perm(m);
        for (int i = 0; i < m; ++i) perm[i] = i;
        std::shuffle(perm.begin(), perm.end(), rng);
        for (int i = 0; i < m; ++i) z[i + 1][k] = norm_inv((perm[i] + U01(rng)) / m);
    }
    for (int i = 0; i < n; ++i) {
        Member mb;
        mb.p = base;
        // Always draw the same number of variates so member i is perturbed
        // identically with and without interventions (paired comparison).
        double ub[32];
        for (double& x : ub) x = U01(rng);
        if (i > 0) {
            const auto& zi = z[i];
            mb.p.wind_mult *= std::exp(u.wind_speed_sigma * zi[0] - 0.5 * u.wind_speed_sigma * u.wind_speed_sigma);
            mb.p.wind_dir_offset += u.wind_dir_sigma * zi[1];
            mb.p.moisture_offset += u.moisture_sigma * zi[2];
            mb.p.ros_mult *= std::exp(u.ros_sigma * zi[3] - 0.5 * u.ros_sigma * u.ros_sigma);
            mb.p.patch_seed = seed * 7919u + uint32_t(i);
            mb.p.patch_sigma = float(u.patch_sigma);
            mb.p.patch_m = float(u.patch_m);
            if (iv)
                for (size_t l = 0; l < iv->layers.size(); ++l)
                    if (iv->layers[l].kind == IvKind::Firebreak && ub[l] < iv->layers[l].breach_p)
                        mb.breached |= 1u << l;
        }
        out.push_back(mb);
    }
    return out;
}

EnsembleResult run_ensemble(const SpreadSolver& solver, const RunConfig& base_cfg, const std::vector<Member>& members) {
    auto t0 = std::chrono::steady_clock::now();
    EnsembleResult R;
    R.members = members;
    const size_t n = members.size();
    R.arrival.resize(n);
    std::atomic<size_t> next{0};
    auto worker = [&] {
        for (size_t i; (i = next++) < n;) {
            RunConfig cfg = base_cfg;
            cfg.p = members[i].p;
            cfg.breached = members[i].breached;
            cfg.detail = (i == 0);
            RunOutput out;
            solver.run(cfg, out);
            if (i == 0) {
                R.arrival[0] = out.arrival;
                R.best = std::move(out);
            } else {
                R.arrival[i] = std::move(out.arrival);
            }
        }
    };
    unsigned nt = std::max(1u, std::min<unsigned>(std::thread::hardware_concurrency(), unsigned(n)));
    std::vector<std::thread> pool;
    for (unsigned k = 0; k < nt; ++k) pool.emplace_back(worker);
    for (auto& th : pool) th.join();

    const size_t N = solver.land().grid.size();
    R.p10.assign(N, kInf);
    R.p50.assign(N, kInf);
    R.p90.assign(N, kInf);
    std::vector<float> v(n);
    const size_t i10 = size_t(std::floor(0.1 * (n - 1))), i50 = (n - 1) / 2, i90 = size_t(std::ceil(0.9 * (n - 1)));
    for (size_t c = 0; c < N; ++c) {
        bool any = false;
        for (size_t m = 0; m < n; ++m) {
            v[m] = R.arrival[m][c];
            any |= v[m] < kInf;
        }
        if (!any) continue;
        std::sort(v.begin(), v.end());
        R.p10[c] = v[i10];
        R.p50[c] = v[i50];
        R.p90[c] = v[i90];
    }
    R.runtime_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return R;
}

}  // namespace rapid
