// landscape.hpp — terrain, fuel and map features on the simulation grid.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "geo.hpp"
#include "windfield.hpp"

namespace rapid {

struct Place {
    std::string name;
    std::string kind;  // village, town, hospital, school, ...
    LatLon pos;
};

struct Way {
    std::string name;
    std::string kind;  // road class or "river"
    bool road = false;
    bool major = false;  // shown as a named boundary / destination
    std::vector<LatLon> pts;
};

struct MapFeatures {
    std::vector<Place> places;
    std::vector<Way> ways;
};

struct Landscape {
    Grid grid;
    std::vector<float> elev;        // m
    std::vector<float> slope_tan;   // rise/run
    std::vector<float> upslope;     // compass bearing pointing uphill (deg)
    WindField wind;                 // terrain-steered wind response (see windfield.hpp)
    std::vector<int8_t> moist_adj;  // aspect moisture class: 0 normal, 1 sunny (drier), 2 shaded (moister)
    std::vector<uint8_t> fuel;      // index into fuel_table()
    std::vector<uint8_t> road;      // 1 = a vehicle-accessible road crosses the cell
    MapFeatures features;

    // Provenance, surfaced in the UI and used for confidence scoring.
    std::string terrain_source = "none", fuel_source = "none", features_source = "none";
    bool terrain_ok = false, fuel_ok = false;
    std::vector<std::string> warnings;
};

struct BuildOptions {
    LatLon centre;
    double radius_m = 12000;
    double cell_m = 40;
    std::string cache_dir = "cache";
    bool offline = false;  // only use cached data
};

// Fetches (or loads from cache) DEM, land cover and map features, and
// derives slope, aspect, the terrain wind field and accessibility.
Landscape build_landscape(const BuildOptions& opt);

// Recomputes slope/aspect/wind field from elev (used by build and tests).
void derive_terrain(Landscape& L);

}  // namespace rapid
