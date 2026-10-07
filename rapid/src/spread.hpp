// spread.hpp — the fire spread engine.
//
// Instead of stepping a cellular automaton through time, the engine computes
// the *arrival time* of the fire at every cell directly with a
// minimum-travel-time (Dijkstra) search over a 16-neighbour stencil, the
// same idea as FlamMap's MTT.  Local spread rate comes from Rothermel (1972)
// with wind + slope combined as vectors and an elliptical fire shape
// (Anderson 1983 length-to-breadth ratio), so a full multi-hour forecast on a
// 600×600 grid costs tens of milliseconds and every destination gets a
// time-to-impact for free.
//
// Weather is time dependent: the ellipse of each cell is evaluated with the
// weather of the hour in which the fire front reaches it.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "fuels.hpp"
#include "landscape.hpp"
#include "weather.hpp"

namespace rapid {

// Tunable model parameters — what calibration fits and the ensemble perturbs.
struct SpreadParams {
    double ros_mult = 1.0;         // overall spread-rate multiplier
    double wind_mult = 1.0;        // forecast wind speed bias
    double wind_dir_offset = 0.0;  // forecast wind direction bias (deg)
    double moisture_offset = 0.0;  // fine dead fuel moisture bias (fraction)
};

enum class IvKind { Firebreak = 0, AirDrop = 1, Truck = 2 };

inline const char* iv_kind_name(IvKind k) {
    switch (k) {
        case IvKind::Firebreak: return "firebreak";
        case IvKind::AirDrop: return "air_drop";
        default: return "truck";
    }
}

struct Intervention {
    std::string id;
    IvKind kind = IvKind::Firebreak;
    std::vector<LatLon> pts;  // firebreak: polyline; drop/truck: single centre point
    double bearing = 0;       // air drop: orientation of the drop line
    double width_m = 0;       // firebreak width / drop width
    double length_m = 0;      // drop length
    double radius_m = 0;      // truck reach
    double t_on = 0, t_off = 1e9;  // minutes since ignition reference
};

// Interventions rasterised onto a grid as time-windowed layers.
struct InterventionRaster {
    struct Layer {
        IvKind kind;
        float t_on, t_off;
        float breach_p = 0;  // firebreaks only: chance the fire jumps it
        int src = 0;         // index of the Intervention it came from
        std::vector<int> cells;
    };
    std::vector<Layer> layers;   // at most 32
    std::vector<uint32_t> mask;  // per cell: bit i set → layer i covers it

    void build(const Grid& g, const std::vector<Intervention>& ivs);
    bool empty() const { return layers.empty(); }
};

struct RunConfig {
    SpreadParams p;
    double t0_epoch = 0;  // reference time for t = 0 (minutes axis)
    double t_end = 360;   // stop when the front passes this time (minutes)
    std::vector<std::pair<int, float>> sources;  // (cell, time it is burning)
    const InterventionRaster* iv = nullptr;
    uint32_t breached = 0;  // bit i → firebreak layer i is breached in this run
    bool detail = true;     // also record predecessor, flame length and ROS
};

struct RunOutput {
    std::vector<float> arrival;  // minutes since t0, kInf if not reached
    std::vector<int32_t> pred;   // cell the fire came from (-1 for sources)
    std::vector<float> flame_m;  // flame length when the front arrived
    std::vector<float> ros;      // local spread rate when the front arrived (m/min)
};

class SpreadSolver {
   public:
    SpreadSolver(const Landscape& L, const WeatherSeries& wx);
    // Stateless and const → one solver can serve many threads at once.
    void run(const RunConfig& cfg, RunOutput& out) const;

    // Head-fire spread rate (m/min) of a cell at time t under params p —
    // used for sizing the domain and for plain-language summaries.
    double head_ros(int cell, double t0_epoch, double t_min, const SpreadParams& p, double* flame_m = nullptr,
                    double* dir_deg = nullptr) const;

    // Fire potential everywhere: head-fire spread rate (km/h), flame length
    // and spread direction at the worst hour between t_from and t_to
    // (minutes since t0).  Returns the index of the hour with the highest
    // average spread rate (the "peak danger" hour).
    int hazard(double t0_epoch, double t_from, double t_to, const SpreadParams& p, std::vector<float>& ros_kmh,
               std::vector<float>& flame_m, std::vector<float>& dir_deg) const;

    // Reverse minimum travel time: for every cell, how long a fire starting
    // there would take to reach any target cell (weather of hour `t_at`), and
    // which target (label) it reaches first.  Stops at t_max minutes.
    void time_to_targets(double t0_epoch, double t_at, const SpreadParams& p,
                         const std::vector<std::pair<int, int>>& targets, float t_max, std::vector<float>& minutes,
                         std::vector<int>& label) const;

    const Landscape& land() const { return L_; }
    const WeatherSeries& weather() const { return wx_; }

   private:
    const Landscape& L_;
    const WeatherSeries& wx_;
    std::vector<RothermelFuel> rf_;
};

// Picks the burnable cell nearest to (r, c) within `max_r` cells, or -1.
int nearest_burnable(const Landscape& L, int r, int c, int max_r = 4);

}  // namespace rapid
