// calibrate.hpp — dynamic calibration against the area that has already
// burned.  Given where the fire was at an earlier time (ignition point or an
// earlier perimeter) and the observed burned area now (drawn on the map,
// drone footage, hotspots), it fits the spread parameters so the model
// reproduces what actually happened, then forecasts on from the observed
// perimeter.
#pragma once

#include <cstdint>
#include <vector>

#include "spread.hpp"

namespace rapid {

struct CalibrationInput {
    std::vector<std::pair<int, float>> sources;  // earlier fire state
    float t_obs = 0;                             // minutes since t0 of the observation
    std::vector<uint8_t> observed;               // 1 = observed burned (may be empty if only hotspots)
    std::vector<int> hotspots;                   // cells observed actively burning at t_obs
    SpreadParams start;
};

struct CalibrationResult {
    SpreadParams params;
    double score_before = 0, score_after = 0;  // agreement 0..1 (IoU-based)
    int evaluations = 0;
    double runtime_ms = 0;
};

// Agreement between the forecast burned area at t_obs and the observation.
double agreement_score(const RunOutput& run, const CalibrationInput& in);

CalibrationResult calibrate(const SpreadSolver& solver, double t0_epoch, const CalibrationInput& in);

}  // namespace rapid
