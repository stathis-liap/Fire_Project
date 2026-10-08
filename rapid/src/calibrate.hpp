// calibrate.hpp — dynamic calibration against the area that has already
// burned.  Given where the fire was at an earlier time (ignition point or an
// earlier perimeter) and the observed burned area now (drawn on the map,
// drone footage, hotspots), it fits the spread parameters so the model
// reproduces what actually happened, then forecasts on from the observed
// perimeter.
//
// The fit is tuned for the next 1–2 hours:
//  * it scores the *growth* since the previous fire state (already-burned
//    ground would otherwise inflate the agreement), plus (20 %) how well the
//    simulated front arrives at the observed edge on time — a smooth term
//    that lets the optimiser see "slightly too slow" vs "far too slow";
//  * earlier observations can take part as extra intervals with a small,
//    recency-decaying weight (heavier history was benchmarked to hurt the
//    1–2 h forecast: the fire has moved into other fuel since);
//  * a quasi-random scan of the whole 4-D parameter space seeds several
//    parallel Nelder–Mead searches (no single-basin assumption);
//  * the width of the good-fit region becomes the ensemble spread, so the
//    confidence reflects what the observations actually pin down;
//  * finally, where the real fire ran further or less far than the fitted
//    model in some direction (fuel map errors, local wind), that ratio is
//    carried forward as a per-cell spread correction near the front, fading
//    with distance — it matters most exactly for the next hour or two.
#pragma once

#include <cstdint>
#include <vector>

#include "ensemble.hpp"
#include "spread.hpp"

namespace rapid {

// One observed step of the fire: from `sources` (fire state at its start)
// to the observation at t_obs.
struct CalibrationInterval {
    std::vector<std::pair<int, float>> sources;
    float t_obs = 0;                // minutes since t0
    std::vector<uint8_t> observed;  // 1 = observed burned (may be empty if only hotspots)
    std::vector<int> hotspots;      // cells observed actively burning at t_obs
    double weight = 1.0;            // relative weight in the fit
};

struct CalibrationInput {
    // The newest interval.
    std::vector<std::pair<int, float>> sources;  // earlier fire state
    float t_obs = 0;                             // minutes since t0 of the observation
    std::vector<uint8_t> observed;               // 1 = observed burned (may be empty if only hotspots)
    std::vector<int> hotspots;                   // cells observed actively burning at t_obs
    SpreadParams start;
    // Earlier intervals still worth fitting (weights set by the caller,
    // typically halving every 2 h of age).
    std::vector<CalibrationInterval> history;
    int seeds = 96;  // quasi-random starting points scanned in parallel
    int starts = 4;  // Nelder–Mead searches started from the best seeds
    bool local_correction = true;  // also learn a per-direction spread correction near the front
};

// How well one run matches one observed interval.
struct FitDetail {
    double score = 0;        // combined agreement 0..1 (what the fit maximises)
    double growth_iou = 0;   // overlap of new burned ground (since the interval start)
    double timing = 0;       // 1 = the front reaches the observed edge exactly on time
    double hotspots = 0;     // share of hotspots burning at the observation time
    double front_err_m = 0;  // mean distance between simulated and observed fire edge
};

struct CalibrationResult {
    SpreadParams params;  // includes params.local_ros when a local correction was learned
    double score_before = 0, score_after = 0;  // newest interval agreement 0..1
    FitDetail before, after;                   // newest interval, start vs fitted parameters
    Uncertainty spread;                        // parameter spread the observations still allow
    double local_min = 1, local_max = 1;       // range of the local spread correction (1 = none)
    int evaluations = 0;
    double runtime_ms = 0;
};

FitDetail evaluate_fit(const Grid& g, const RunOutput& run, const CalibrationInterval& iv, bool front_distance = false);

// Agreement between the forecast burned area at t_obs and the newest observation.
double agreement_score(const Grid& g, const RunOutput& run, const CalibrationInput& in);

CalibrationResult calibrate(const SpreadSolver& solver, double t0_epoch, const CalibrationInput& in);

}  // namespace rapid
