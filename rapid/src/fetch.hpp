// fetch.hpp — internet data sources with a disk cache, so a laptop that has
// already loaded an area keeps working when the network drops.
//
//   Terrain : AWS Terrain Tiles (Terrarium PNG, global, no key)
//   Fuel    : EEA CORINE Land Cover 2018 MapServer export (Europe)
//   Weather : Open-Meteo forecast API (global, no key)
//   Places  : OpenStreetMap via Overpass (villages, hospitals, roads, rivers)
#pragma once

#include <optional>
#include <string>
#include <vector>

#include "geo.hpp"
#include "landscape.hpp"
#include "weather.hpp"

namespace rapid {

struct FetchContext {
    std::string cache_dir = "cache";
    bool offline = false;
};

// GET with a disk cache.  A cached copy younger than max_age_s is returned
// without touching the network; an older one is used when the network fails.
std::optional<std::string> http_get_cached(const FetchContext& ctx, const std::string& url,
                                           const std::string& cache_key, double max_age_s, int timeout_s,
                                           const std::string& post_body = "");

bool fetch_elevation(const FetchContext& ctx, const Grid& g, std::vector<float>& elev, std::string& source);
bool fetch_fuel_corine(const FetchContext& ctx, const Grid& g, std::vector<uint8_t>& fuel, std::string& source);
bool fetch_weather(const FetchContext& ctx, LatLon p, WeatherSeries& out);
bool fetch_features(const FetchContext& ctx, const Grid& g, MapFeatures& out, std::string& source);

}  // namespace rapid
