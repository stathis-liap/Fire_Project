#include "fetch.hpp"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <future>
#include <map>
#include <sstream>

#include "fuels.hpp"
#include "httplib.h"
#include "json.hpp"
#include "stb_image.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace rapid {

namespace {

const char* kUserAgent = "WILSON-rapid/1.0 (wildfire rapid response; github.com/stathis-liap/Fire_Project)";

std::optional<std::string> read_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return std::nullopt;
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

void write_file(const fs::path& p, const std::string& data) {
    std::error_code ec;
    fs::create_directories(p.parent_path(), ec);
    fs::path tmp = p;
    tmp += ".part";
    {
        std::ofstream f(tmp, std::ios::binary);
        f.write(data.data(), std::streamsize(data.size()));
    }
    fs::rename(tmp, p, ec);
}

double age_seconds(const fs::path& p) {
    std::error_code ec;
    auto t = fs::last_write_time(p, ec);
    if (ec) return 1e18;
    auto now = fs::file_time_type::clock::now();
    return std::chrono::duration<double>(now - t).count();
}

std::optional<std::string> http_request(const std::string& url, int timeout_s, const std::string& post_body) {
    auto scheme_end = url.find("://");
    if (scheme_end == std::string::npos) return std::nullopt;
    auto path_start = url.find('/', scheme_end + 3);
    std::string origin = url.substr(0, path_start);
    std::string path = path_start == std::string::npos ? "/" : url.substr(path_start);
    httplib::Client cli(origin);
    cli.set_follow_location(true);
    cli.set_connection_timeout(std::min(timeout_s, 8), 0);
    cli.set_read_timeout(timeout_s, 0);
    cli.set_write_timeout(timeout_s, 0);
    httplib::Headers h = {{"User-Agent", kUserAgent}};
    httplib::Result res = post_body.empty()
                              ? cli.Get(path, h)
                              : cli.Post(path, h, post_body, "application/x-www-form-urlencoded");
    if (!res || res->status != 200) return std::nullopt;
    return res->body;
}

std::string url_encode(const std::string& s) {
    std::ostringstream o;
    for (unsigned char c : s) {
        if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') o << c;
        else o << '%' << std::uppercase << std::hex << ((c >> 4) & 0xF) << (c & 0xF) << std::nouppercase << std::dec;
    }
    return o.str();
}

std::string fmt(double v, int prec) {
    std::ostringstream o;
    o.setf(std::ios::fixed);
    o.precision(prec);
    o << v;
    return o.str();
}

}  // namespace

std::optional<std::string> http_get_cached(const FetchContext& ctx, const std::string& url,
                                           const std::string& cache_key, double max_age_s, int timeout_s,
                                           const std::string& post_body) {
    fs::path p = fs::path(ctx.cache_dir) / cache_key;
    bool have = fs::exists(p);
    if (have && (ctx.offline || age_seconds(p) <= max_age_s)) return read_file(p);
    if (!ctx.offline) {
        if (auto body = http_request(url, timeout_s, post_body)) {
            write_file(p, *body);
            return body;
        }
    }
    if (have) return read_file(p);  // stale beats nothing in an emergency
    return std::nullopt;
}

// ── Terrain: AWS Terrarium tiles ─────────────────────────────────────────────
bool fetch_elevation(const FetchContext& ctx, const Grid& g, std::vector<float>& elev, std::string& source) {
    const double lat0 = 0.5 * (g.north + g.south());
    int z = int(std::ceil(std::log2(156543.03392 * std::cos(lat0 * kDeg) / g.cell_m)));
    z = std::clamp(z, 6, 13);
    const double n = std::pow(2.0, z);
    auto px_x = [&](double lon) { return (lon + 180.0) / 360.0 * n * 256.0; };
    auto px_y = [&](double lat) {
        double s = std::sin(lat * kDeg);
        return (0.5 - std::log((1 + s) / (1 - s)) / (4 * kPi)) * n * 256.0;
    };
    int tx0 = int(std::floor(px_x(g.west) / 256)) , tx1 = int(std::floor(px_x(g.east()) / 256));
    int ty0 = int(std::floor(px_y(g.north) / 256)), ty1 = int(std::floor(px_y(g.south()) / 256));

    std::map<std::pair<int, int>, std::vector<float>> tiles;
    std::vector<std::pair<int, int>> keys;
    for (int tx = tx0; tx <= tx1; ++tx)
        for (int ty = ty0; ty <= ty1; ++ty) keys.push_back({tx, ty});

    // Download in parallel, 8 at a time.
    std::mutex mu;
    for (size_t i = 0; i < keys.size(); i += 8) {
        std::vector<std::future<void>> jobs;
        for (size_t j = i; j < std::min(keys.size(), i + 8); ++j) {
            jobs.push_back(std::async(std::launch::async, [&, j] {
                auto [tx, ty] = keys[j];
                std::string key = "terrain/" + std::to_string(z) + "/" + std::to_string(tx) + "/" + std::to_string(ty) + ".png";
                std::string url = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/" + std::to_string(z) + "/" +
                                  std::to_string(tx) + "/" + std::to_string(ty) + ".png";
                auto body = http_get_cached(ctx, url, key, 1e12, 20);
                if (!body) return;
                int w, h, ch;
                unsigned char* px = stbi_load_from_memory(reinterpret_cast<const unsigned char*>(body->data()),
                                                          int(body->size()), &w, &h, &ch, 3);
                if (!px || w != 256 || h != 256) {
                    if (px) stbi_image_free(px);
                    return;
                }
                std::vector<float> e(256 * 256);
                for (int k = 0; k < 256 * 256; ++k)
                    e[k] = float(px[3 * k] * 256.0 + px[3 * k + 1] + px[3 * k + 2] / 256.0 - 32768.0);
                stbi_image_free(px);
                std::lock_guard<std::mutex> lk(mu);
                tiles[{tx, ty}] = std::move(e);
            }));
        }
        for (auto& f : jobs) f.get();
    }
    if (tiles.empty()) return false;

    auto sample_px = [&](long gx, long gy, bool& ok) -> float {
        auto it = tiles.find({int(gx >> 8), int(gy >> 8)});
        if (it == tiles.end()) { ok = false; return 0.f; }
        return it->second[(gy & 255) * 256 + (gx & 255)];
    };
    elev.assign(g.size(), 0.f);
    size_t missing = 0;
    for (int r = 0; r < g.rows; ++r)
        for (int c = 0; c < g.cols; ++c) {
            LatLon p = g.center_of(r, c);
            double fx = px_x(p.lon) - 0.5, fy = px_y(p.lat) - 0.5;
            long x0 = long(std::floor(fx)), y0 = long(std::floor(fy));
            double ax = fx - x0, ay = fy - y0;
            bool ok = true;
            float v = float((1 - ax) * (1 - ay) * sample_px(x0, y0, ok) + ax * (1 - ay) * sample_px(x0 + 1, y0, ok) +
                            (1 - ax) * ay * sample_px(x0, y0 + 1, ok) + ax * ay * sample_px(x0 + 1, y0 + 1, ok));
            if (!ok) ++missing;
            elev[g.idx(r, c)] = v;
        }
    source = "AWS Terrain Tiles (zoom " + std::to_string(z) + ", " + std::to_string(tiles.size()) + "/" +
             std::to_string(keys.size()) + " tiles)";
    return missing < g.size() / 10;
}

// ── Fuel: CORINE Land Cover 2018 ─────────────────────────────────────────────
bool fetch_fuel_corine(const FetchContext& ctx, const Grid& g, std::vector<uint8_t>& fuel, std::string& source) {
    int w = std::min(g.cols, 2048), h = std::min(g.rows, 2048);
    std::string bbox = fmt(g.west, 5) + "," + fmt(g.south(), 5) + "," + fmt(g.east(), 5) + "," + fmt(g.north, 5);
    std::string url =
        "https://image.discomap.eea.europa.eu/arcgis/rest/services/Corine/CLC2018_WM/MapServer/export?bbox=" + bbox +
        "&bboxSR=4326&imageSR=4326&size=" + std::to_string(w) + "," + std::to_string(h) +
        "&format=png&transparent=false&f=image";
    std::string key = "corine/" + bbox + "_" + std::to_string(w) + "x" + std::to_string(h) + ".png";
    auto body = http_get_cached(ctx, url, key, 1e12, 45);
    if (!body || body->size() < 8 || (*body)[1] != 'P') return false;
    int iw, ih, ch;
    unsigned char* px =
        stbi_load_from_memory(reinterpret_cast<const unsigned char*>(body->data()), int(body->size()), &iw, &ih, &ch, 3);
    if (!px) return false;

    constexpr uint8_t kUnknown = 255;
    fuel.assign(g.size(), kUnknown);
    size_t known = 0;
    for (int r = 0; r < g.rows; ++r)
        for (int c = 0; c < g.cols; ++c) {
            int sx = std::min(iw - 1, int((c + 0.5) * iw / g.cols)), sy = std::min(ih - 1, int((r + 0.5) * ih / g.rows));
            const unsigned char* p = px + 3 * (size_t(sy) * iw + sx);
            int code = clc_from_rgb(p[0], p[1], p[2]);
            const char* name = code ? clc_to_fuel(code) : nullptr;
            if (name) {
                fuel[g.idx(r, c)] = uint8_t(fuel_index(name));
                ++known;
            }
        }
    stbi_image_free(px);
    source = "CORINE Land Cover 2018 (EEA)";
    return known > g.size() / 4;
}

// ── Weather: Open-Meteo ──────────────────────────────────────────────────────
bool fetch_weather(const FetchContext& ctx, LatLon p, WeatherSeries& out) {
    std::string lat = fmt(p.lat, 2), lon = fmt(p.lon, 2);
    std::string url = "https://api.open-meteo.com/v1/forecast?latitude=" + lat + "&longitude=" + lon +
                      "&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m"
                      "&wind_speed_unit=ms&timeformat=unixtime&past_days=1&forecast_days=2&timezone=GMT";
    auto body = http_get_cached(ctx, url, "weather/" + lat + "_" + lon + ".json", 1800, 20);
    if (!body) return false;
    try {
        json j = json::parse(*body);
        const auto& hr = j.at("hourly");
        const auto &t = hr.at("time"), &T = hr.at("temperature_2m"), &RH = hr.at("relative_humidity_2m"),
                   &W = hr.at("wind_speed_10m"), &D = hr.at("wind_direction_10m");
        out.hourly.clear();
        for (size_t i = 0; i < t.size(); ++i) {
            if (T[i].is_null() || W[i].is_null() || D[i].is_null() || RH[i].is_null()) continue;
            out.hourly.push_back({t[i].get<double>(), W[i].get<double>(), D[i].get<double>(), T[i].get<double>(),
                                  RH[i].get<double>()});
        }
        out.source = "Open-Meteo forecast";
        return !out.hourly.empty();
    } catch (...) {
        return false;
    }
}

// ── Places, roads and rivers: OpenStreetMap / Overpass ───────────────────────
bool fetch_features(const FetchContext& ctx, const Grid& g, MapFeatures& out, std::string& source) {
    std::string bb = fmt(g.south(), 4) + "," + fmt(g.west, 4) + "," + fmt(g.north, 4) + "," + fmt(g.east(), 4);
    std::string q =
        "[out:json][timeout:40];("
        "node[\"place\"~\"^(city|town|village|hamlet|suburb)$\"](" + bb + ");"
        "nwr[\"amenity\"~\"^(hospital|clinic|school|kindergarten|fire_station|nursing_home)$\"](" + bb + ");"
        "way[\"highway\"~\"^(motorway|trunk|primary|secondary|tertiary|unclassified)$\"](" + bb + ");"
        "way[\"waterway\"=\"river\"][\"name\"](" + bb + ");"
        ");out geom qt;";
    std::string body_param = "data=" + url_encode(q);
    std::optional<std::string> body;
    for (const char* ep : {"https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter",
                           "https://maps.mail.ru/osm/tools/overpass/api/interpreter"}) {
        body = http_get_cached(ctx, ep, "osm/" + bb + ".json", 30 * 86400.0, 45, body_param);
        if (body && !body->empty() && (*body)[0] == '{') break;
        body.reset();
    }
    if (!body) return false;
    try {
        json j = json::parse(*body);
        for (const auto& e : j.at("elements")) {
            const auto& tags = e.value("tags", json::object());
            std::string type = e.value("type", "");
            std::string name = tags.value("name:en", tags.value("name", ""));
            if (tags.contains("place") || tags.contains("amenity")) {
                std::string kind = tags.contains("place") ? tags["place"].get<std::string>() : tags["amenity"].get<std::string>();
                LatLon pos;
                if (type == "node") pos = {e.value("lat", 0.0), e.value("lon", 0.0)};
                else if (e.contains("bounds"))
                    pos = {0.5 * (e["bounds"]["minlat"].get<double>() + e["bounds"]["maxlat"].get<double>()),
                           0.5 * (e["bounds"]["minlon"].get<double>() + e["bounds"]["maxlon"].get<double>())};
                else continue;
                if (name.empty()) {
                    if (tags.contains("place")) continue;
                    name = kind;
                    name[0] = char(toupper(name[0]));
                    std::replace(name.begin(), name.end(), '_', ' ');
                }
                out.places.push_back({name, kind, pos});
            } else if (type == "way" && e.contains("geometry")) {
                Way w;
                w.name = name.empty() ? tags.value("ref", "") : name;
                if (tags.contains("highway")) {
                    w.kind = tags["highway"].get<std::string>();
                    w.road = true;
                    w.major = (w.kind == "motorway" || w.kind == "trunk" || w.kind == "primary") && !w.name.empty();
                } else {
                    w.kind = "river";
                    w.major = !w.name.empty();
                }
                for (const auto& pt : e["geometry"]) w.pts.push_back({pt.value("lat", 0.0), pt.value("lon", 0.0)});
                out.ways.push_back(std::move(w));
            }
        }
        source = "OpenStreetMap (Overpass)";
        return true;
    } catch (...) {
        return false;
    }
}

}  // namespace rapid
