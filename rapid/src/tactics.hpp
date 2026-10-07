// tactics.hpp — automatic tactical recommendations.
//
// Uses the forecast's fire paths (the predecessor tree of the
// minimum-travel-time solution) to place resources where they act before
// the fire arrives, following standard wildland doctrine:
//   * Firebreaks / dead zones across the path to a threatened place, far
//     enough ahead that crews finish before the fire arrives, anchored on
//     non-burnable ground where possible.
//   * Aerial water/retardant drops on the head of the fire where it will be
//     when the aircraft arrives.
//   * Fire trucks between the fire and threatened places, on a road, where
//     flames are low enough for engines (hauling-chart limits).
#pragma once

#include <string>
#include <vector>

#include "ensemble.hpp"
#include "landscape.hpp"
#include "spread.hpp"

namespace rapid {

struct Threat {
    std::string dest_id, name;
    int cell = -1;   // cell where the fire first touches the place
    float eta = 0;   // best-guess arrival (minutes since t0)
    double prob = 0; // probability of arrival within the horizon
};

struct TacticsInput {
    const Landscape* L = nullptr;
    const RunOutput* best = nullptr;  // member 0 with predecessor tree
    double t0_epoch = 0;  // wall-clock time of t = 0
    float t_now = 0, t_end = 0;
    LatLon origin;
    double head_bearing = 0;
    double wind_kmh = 0;
    int head_cell = -1;           // furthest point of the head fire (for an untargeted break)
    std::vector<Threat> threats;  // sorted by eta
};

// Lead times (minutes) — conservative defaults for Greek fire services.
struct TacticsTiming {
    double air_eta = 30;          // first aircraft over the fire
    double truck_eta = 25;        // engines on scene
    double dozer_m_per_h = 1200;  // firebreak production rate (dozer + hand crew)
    double safety = 30;           // margin between "line finished" and fire arrival
    double aircraft_hold = 180;   // how long one aircraft keeps re-dropping on its line
    // Size of the candidate pool the planner chooses from.
    int max_breaks = 4, max_drops = 5, max_trucks = 6;
};

struct Recommendation {
    std::string id;
    std::string type;        // firebreak | air_drop | truck | evacuate
    std::string resource;    // dozers | aircraft | trucks | "" (no resource needed)
    bool simulated = true;   // false → advice only (evacuation), not modelled
    Intervention iv;
    // Text may contain {t:EPOCH} tokens; the UI renders them as local clock times.
    std::string title, detail;
    std::string target_id, target_name;
    float act_before = 0;  // minutes since t0
    double breach_p = 0;
    double confidence = 0;  // filled in from the paired ensemble
    std::string effect;     // filled in from the what-if run
    bool enabled = true;     // part of the plan that is simulated
    bool selected = false;   // chosen by the resource planner
    bool forced = false;     // switched on by the user
    double gain = 0;         // planner value added when it was chosen / would be added
    int order = 0;
};

// All sensible candidate actions (more than the resources allow); the
// planner in planner.hpp chooses which to use.  Evacuation warnings are
// always included and need no resources.
std::vector<Recommendation> plan_tactics(const TacticsInput& in, const TacticsTiming& tm = {});

// Fire path from `cell` back towards the ignition, following predecessors.
std::vector<int> trace_path(const RunOutput& run, int cell, int max_len = 100000);
// Direction the fire is moving at `cell` (compass degrees), from its path.
double spread_bearing(const Grid& g, const RunOutput& run, int cell, int lookback = 6);

}  // namespace rapid
