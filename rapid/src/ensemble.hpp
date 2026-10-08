// ensemble.hpp — many perturbed forecasts run in parallel.  Their spread is
// what turns a single deterministic answer into a confidence level.
#pragma once

#include <cstdint>
#include <vector>

#include "spread.hpp"

namespace rapid {

// How uncertain each input is; set from the data quality of the incident.
struct Uncertainty {
    double wind_speed_sigma = 0.20;  // log-normal σ of wind speed multiplier
    double wind_dir_sigma = 15.0;    // degrees
    double moisture_sigma = 0.012;   // absolute fraction
    double ros_sigma = 0.25;         // log-normal σ of spread-rate multiplier
    double patch_sigma = 0.0;        // log σ of patchy, place-to-place spread-rate errors
    double patch_m = 1000;           // size of those patches (m)
};

struct Member {
    SpreadParams p;
    uint32_t breached = 0;
};

struct EnsembleResult {
    std::vector<Member> members;                 // members[0] is the unperturbed (best-guess) run
    std::vector<std::vector<float>> arrival;     // per member
    RunOutput best;                              // full detail of member 0
    std::vector<float> p10, p50, p90;            // per-cell arrival quantiles
    double runtime_ms = 0;

    // Fraction of members in which the fire reaches `cell` by time t.
    double prob_by(int cell, float t) const {
        if (arrival.empty()) return 0;
        int n = 0;
        for (const auto& a : arrival) n += a[cell] <= t;
        return double(n) / double(arrival.size());
    }
};

std::vector<Member> make_members(const SpreadParams& base, const Uncertainty& u, int n, uint32_t seed,
                                 const InterventionRaster* iv = nullptr);

// Runs all members (member 0 with full detail) on all hardware threads.
EnsembleResult run_ensemble(const SpreadSolver& solver, const RunConfig& base_cfg, const std::vector<Member>& members);

}  // namespace rapid
