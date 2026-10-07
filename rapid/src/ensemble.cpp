#include "ensemble.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <random>
#include <thread>

namespace rapid {

std::vector<Member> make_members(const SpreadParams& base, const Uncertainty& u, int n, uint32_t seed,
                                 const InterventionRaster* iv) {
    std::vector<Member> out;
    std::mt19937 rng(seed);
    std::normal_distribution<double> N01(0.0, 1.0);
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    for (int i = 0; i < n; ++i) {
        Member m;
        m.p = base;
        // Always draw the same number of variates so member i is perturbed
        // identically with and without interventions (paired comparison).
        double zw = N01(rng), zd = N01(rng), zm = N01(rng), zr = N01(rng);
        double ub[32];
        for (double& x : ub) x = U01(rng);
        if (i > 0) {
            m.p.wind_mult *= std::exp(u.wind_speed_sigma * zw - 0.5 * u.wind_speed_sigma * u.wind_speed_sigma);
            m.p.wind_dir_offset += u.wind_dir_sigma * zd;
            m.p.moisture_offset += u.moisture_sigma * zm;
            m.p.ros_mult *= std::exp(u.ros_sigma * zr - 0.5 * u.ros_sigma * u.ros_sigma);
            if (iv)
                for (size_t l = 0; l < iv->layers.size(); ++l)
                    if (iv->layers[l].kind == IvKind::Firebreak && ub[l] < iv->layers[l].breach_p)
                        m.breached |= 1u << l;
        }
        out.push_back(m);
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
