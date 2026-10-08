// render.hpp — turns arrival-time grids into things a map can show:
// colour-coded risk zones (PNG), hourly fire-front lines and spread arrows.
#pragma once

#include <string>
#include <vector>

#include "geo.hpp"
#include "json.hpp"

namespace rapid {

using json = nlohmann::json;

// Zone image: burned / <1 h / 1–2 h / 2–3 h / later / "possible" (only the
// pessimistic ensemble members reach it).  Returns a data: URL.
std::string render_zones_png(const Grid& g, const std::vector<float>& p50, const std::vector<float>& p10, float t_now,
                             float t_end);

// Probability heat maps: for each time in `times` (minutes since t0,
// ascending), the share of scenarios in which the fire has reached each cell
// by then.  Returns data: URLs.
std::vector<std::string> render_prob_pngs(const Grid& g, const std::vector<std::vector<float>>& arrivals,
                                          const std::vector<float>& times);

// Danger map: class 0 = cannot burn (transparent), 1 low … 5 extreme.
std::string render_danger_png(const Grid& g, const std::vector<uint8_t>& danger_class);

// Marching-squares iso-lines of `field` at `level`, joined into polylines.
std::vector<std::vector<LatLon>> contour_lines(const Grid& g, const std::vector<float>& field, float level);

std::string base64_encode(const std::string& in);

// GeoJSON helpers.
json geo_point(LatLon p);
json geo_line(const std::vector<LatLon>& pts);
json feature(const json& geometry, const json& props);

}  // namespace rapid
