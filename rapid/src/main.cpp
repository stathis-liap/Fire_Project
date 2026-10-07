// main.cpp — WILSON Rapid: lightweight rapid-response fire spread estimator.
//
//   wilson_rapid                       start the map UI on http://localhost:8090
//   wilson_rapid --forecast 38.5,22.0  print a plain-text forecast and exit
//
// See rapid/README.md for the HTTP API and the drop-in folder format.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <thread>

#include "httplib.h"
#include "incident.hpp"
#include "json.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace rapid;

namespace {

struct Args {
    std::string host = "127.0.0.1";
    int port = 8090;
    std::string web, cache = "cache", dropin = "dropin";
    bool offline = false;
    int members = 24;
    std::string forecast;  // "lat,lon" → CLI mode
    double ago = 0, hours = 6;
};

void usage() {
    std::puts(
        "WILSON Rapid — rapid-response wildfire spread estimator\n\n"
        "  wilson_rapid [options]\n"
        "  wilson_rapid --forecast LAT,LON [--ago MIN] [--hours H]\n\n"
        "  --host ADDR      listen address (default 127.0.0.1; use 0.0.0.0 to serve tablets on the LAN)\n"
        "  --port N         HTTP port (default 8090)\n"
        "  --web DIR        UI directory (default: auto-detect rapid/web)\n"
        "  --cache DIR      map/weather cache (default ./cache)\n"
        "  --dropin DIR     folder watched for drone/crew observation files (default ./dropin)\n"
        "  --offline        never touch the network; use cached data only\n"
        "  --members N      ensemble size for confidence (default 24)\n");
}

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage(); std::exit(2); }
            return argv[++i];
        };
        if (s == "--host") a.host = next();
        else if (s == "--port") a.port = std::stoi(next());
        else if (s == "--web") a.web = next();
        else if (s == "--cache") a.cache = next();
        else if (s == "--dropin") a.dropin = next();
        else if (s == "--offline") a.offline = true;
        else if (s == "--members") a.members = std::clamp(std::stoi(next()), 4, 128);
        else if (s == "--forecast") a.forecast = next();
        else if (s == "--ago") a.ago = std::stod(next());
        else if (s == "--hours") a.hours = std::stod(next());
        else if (s == "-h" || s == "--help") { usage(); std::exit(0); }
        else { std::cerr << "Unknown option " << s << "\n"; usage(); std::exit(2); }
    }
    return a;
}

std::string find_web_dir(const std::string& given, const char* argv0) {
    if (!given.empty()) return given;
    std::error_code ec;
    fs::path exe = fs::canonical(fs::path(argv0), ec).parent_path();
    for (fs::path p : {fs::path("web"), fs::path("rapid/web"), exe / "../web", exe / "web", exe / "../../web"})
        if (fs::exists(p / "index.html", ec)) return fs::weakly_canonical(p, ec).string();
    return "web";
}

std::string replace_tokens(std::string s) {
    static const std::regex tok(R"(\{t:(\d+)\})");
    std::string out;
    auto it = std::sregex_iterator(s.begin(), s.end(), tok);
    size_t last = 0;
    for (; it != std::sregex_iterator(); ++it) {
        out += s.substr(last, it->position() - last);
        std::time_t t = std::stoll((*it)[1].str());
        char buf[16];
        std::strftime(buf, sizeof buf, "%H:%M", std::localtime(&t));
        out += buf;
        last = it->position() + it->length();
    }
    return out + s.substr(last);
}

std::string clock_of(const json& v) {
    if (!v.is_number()) return "--:--";
    return replace_tokens("{t:" + std::to_string(v.get<long long>()) + "}");
}

// ── Drop-in folder: JSON / GeoJSON observations, or the drone module's CSV ──
double parse_iso_utc(const std::string& s) {
    std::tm tm{};
    std::istringstream ss(s.substr(0, 19));
    ss >> std::get_time(&tm, "%Y-%m-%dT%H:%M:%S");
    if (ss.fail()) return 0;
    return double(timegm(&tm));
}

std::vector<json> parse_dropin(const fs::path& p) {
    std::vector<json> out;
    std::ifstream f(p);
    std::stringstream ss;
    ss << f.rdbuf();
    std::string text = ss.str();
    std::string ext = p.extension().string();
    if (ext == ".csv") {
        // drone/src/main.py FireEventLogger: timestamp_utc, ..., fire_lat, fire_lon, ..., confidence
        std::istringstream in(text);
        std::string line;
        std::vector<std::string> head;
        std::vector<std::tuple<double, double, double, double>> rows;  // t, lat, lon, conf
        while (std::getline(in, line)) {
            std::vector<std::string> cols;
            std::stringstream ls(line);
            std::string c;
            while (std::getline(ls, c, ',')) cols.push_back(c);
            if (head.empty()) { head = cols; continue; }
            auto col = [&](const char* name) -> std::string {
                for (size_t i = 0; i < head.size() && i < cols.size(); ++i)
                    if (head[i] == name) return cols[i];
                return "";
            };
            try {
                rows.emplace_back(parse_iso_utc(col("timestamp_utc")), std::stod(col("fire_lat")),
                                  std::stod(col("fire_lon")), col("confidence").empty() ? 1.0 : std::stod(col("confidence")));
            } catch (...) {}
        }
        if (rows.empty()) return out;
        double tmax = 0;
        for (auto& r : rows) tmax = std::max(tmax, std::get<0>(r));
        json pts = json::array();
        for (auto& [t, la, lo, cf] : rows)
            if (cf >= 0.3 && t >= tmax - 20 * 60) pts.push_back({lo, la});
        if (!pts.empty())
            out.push_back({{"type", "hotspots"}, {"points", pts}, {"time", tmax > 0 ? tmax : double(std::time(nullptr))},
                           {"source", "drone " + p.filename().string()}});
        return out;
    }
    json j = json::parse(text, nullptr, false);
    if (j.is_discarded()) return out;
    if (j.is_array()) {
        for (auto& e : j) out.push_back(e);
    } else if (j.value("type", "") == "FeatureCollection") {
        for (auto& ftr : j["features"]) {
            json props = ftr.value("properties", json::object());
            json o = props;
            o["type"] = props.value("type", props.value("kind", ""));
            o["geometry"] = ftr["geometry"];
            if (ftr["geometry"].value("type", "") == "Point" || ftr["geometry"].value("type", "") == "MultiPoint")
                o["points"] = ftr["geometry"];
            if (o["type"] == "") o["type"] = ftr["geometry"].value("type", "") == "Polygon" ? "perimeter" : "hotspots";
            out.push_back(o);
        }
    } else if (j.value("type", "") == "Feature") {
        json props = j.value("properties", json::object());
        json o = props;
        o["type"] = props.value("type", props.value("kind", "perimeter"));
        o["geometry"] = j["geometry"];
        out.push_back(o);
    } else {
        out.push_back(j);
    }
    if (!out.empty())
        for (auto& o : out)
            if (!o.contains("source")) o["source"] = "drop-in " + p.filename().string();
    return out;
}

void watch_dropin(Incident& inc, const std::string& dir, std::atomic<bool>& stop) {
    std::error_code ec;
    fs::create_directories(fs::path(dir) / "processed", ec);
    while (!stop) {
        // Files wait in the folder until there is a fire to apply them to.
        if (!inc.active()) {
            std::this_thread::sleep_for(std::chrono::seconds(2));
            continue;
        }
        for (auto& e : fs::directory_iterator(dir, ec)) {
            if (!e.is_regular_file()) continue;
            std::string ext = e.path().extension().string();
            if (ext != ".json" && ext != ".geojson" && ext != ".csv") continue;
            // Skip files still being written.
            auto age = fs::file_time_type::clock::now() - fs::last_write_time(e.path(), ec);
            if (age < std::chrono::seconds(1)) continue;
            bool failed = false;
            try {
                auto items = parse_dropin(e.path());
                if (items.empty()) {
                    failed = true;
                    std::cout << "[dropin] " << e.path().filename().string() << " → unreadable (not JSON/GeoJSON/drone CSV)" << std::endl;
                }
                for (auto& obs : items) {
                    json r = obs.value("type", "") == "destination" ? inc.add_destination(obs) : inc.observe(obs);
                    failed |= r.contains("error");
                    std::cout << "[dropin] " << e.path().filename().string() << " → "
                              << (r.contains("error") ? r["error"].get<std::string>() : "applied") << std::endl;
                }
            } catch (const std::exception& ex) {
                failed = true;
                std::cout << "[dropin] " << e.path().filename().string() << " → rejected: " << ex.what() << std::endl;
            }
            fs::path dst = fs::path(dir) / "processed" /
                           (std::to_string(std::time(nullptr)) + (failed ? "_FAILED_" : "_") + e.path().filename().string());
            fs::rename(e.path(), dst, ec);
        }
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
}

int run_cli(const Args& a) {
    auto comma = a.forecast.find(',');
    if (comma == std::string::npos) { usage(); return 2; }
    IncidentOptions io{a.cache, a.offline, a.members};
    Incident inc(io);
    json req = {{"lat", std::stod(a.forecast.substr(0, comma))}, {"lon", std::stod(a.forecast.substr(comma + 1))},
                {"started_min_ago", a.ago}, {"horizon_h", a.hours}};
    json r = inc.start(req);
    if (r.contains("error")) { std::cerr << r["error"].get<std::string>() << "\n"; return 1; }
    // Places and roads load in the background; give them a moment.
    for (int i = 0; i < 60 && r["sources"].value("places", "none") == "none"; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        r = inc.snapshot();
    }
    const auto& s = r["summary"];
    const auto& w = r["weather"]["now"];
    std::printf("\nFIRE FORECAST  (next %.0f h, confidence %s %d%%)\n", r["incident"]["horizon_h"].get<double>(),
                r["confidence"]["grade"].get<std::string>().c_str(), r["confidence"]["score"].get<int>());
    std::printf("  Spreading towards the %s at up to %.1f km/h; flames up to %.1f m.\n",
                s["direction_text"].get<std::string>().c_str(), s["head_speed_kmh"].get<double>(), s["max_flame_m"].get<double>());
    std::printf("  %s\n", s["attack_advice"].get<std::string>().c_str());
    std::printf("  Wind %s km/h from the %s, %s °C, %s %% humidity (%s)\n", w["wind_kmh"].dump().c_str(),
                w["dir_text"].get<std::string>().c_str(), w["temp_c"].dump().c_str(), w["rh"].dump().c_str(),
                w["source"].get<std::string>().c_str());
    std::printf("  Burned area: %s ha now → %s ha by %s\n", s["area_ha_now"].dump().c_str(), s["area_ha_end"].dump().c_str(),
                clock_of(r["incident"]["end_epoch"]).c_str());
    std::printf("\nPLACES AT RISK\n");
    int n = 0;
    for (auto& d : r["destinations"]) {
        if (++n > 12) break;
        std::printf("  %-28s %s  (%s%% likely, confidence %s)\n", d["name"].get<std::string>().substr(0, 28).c_str(),
                    d["eta_epoch"].is_null() ? "not reached     " : ("fire ~" + clock_of(d["eta_epoch"]) + "     ").c_str(),
                    d["prob"].dump().c_str(), d["grade"].get<std::string>().c_str());
    }
    if (!n) std::printf("  none within the forecast period\n");
    std::printf("\nRECOMMENDED ACTIONS\n");
    for (auto& rec : r["recommendations"]) {
        std::printf("  %d. %s  [act before %s]\n     %s\n", rec["order"].get<int>(), rec["title"].get<std::string>().c_str(),
                    clock_of(rec["act_before_epoch"]).c_str(), replace_tokens(rec["detail"].get<std::string>()).c_str());
        const auto& g = rec["geometry"];
        if (g["type"] == "Point")
            std::printf("     at %.5f, %.5f\n", g["coordinates"][1].get<double>(), g["coordinates"][0].get<double>());
        else
            std::printf("     from %.5f, %.5f to %.5f, %.5f\n", g["coordinates"][0][1].get<double>(),
                        g["coordinates"][0][0].get<double>(), g["coordinates"].back()[1].get<double>(),
                        g["coordinates"].back()[0].get<double>());
        if (!rec["effect"].get<std::string>().empty())
            std::printf("     Effect: %s (confidence %s)\n", replace_tokens(rec["effect"].get<std::string>()).c_str(),
                        rec["grade"].get<std::string>().c_str());
    }
    for (auto& wmsg : r["warnings"]) std::printf("\n! %s", wmsg.get<std::string>().c_str());
    std::printf("\n\n(computed %d-member ensemble in %s ms, total %s ms)\n", r["timing_ms"]["members"].get<int>(),
                r["timing_ms"]["ensemble"].dump().c_str(), r["timing_ms"]["recompute"].dump().c_str());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    Args a = parse(argc, argv);
    if (!a.forecast.empty()) return run_cli(a);

    IncidentOptions io{a.cache, a.offline, a.members};
    Incident inc(io);
    httplib::Server srv;
    const std::string web = find_web_dir(a.web, argv[0]);
    if (!srv.set_mount_point("/", web)) std::cerr << "WARNING: UI directory not found: " << web << "\n";

    auto send = [](httplib::Response& res, const json& j) {
        res.status = j.contains("error") ? 400 : 200;
        res.set_content(j.dump(), "application/json");
    };
    auto body = [](const httplib::Request& req) {
        json j = json::parse(req.body, nullptr, false);
        return j.is_discarded() ? json::object() : j;
    };

    srv.Get("/api/status", [&](const httplib::Request&, httplib::Response& res) { send(res, inc.status()); });
    srv.Get("/api/result", [&](const httplib::Request&, httplib::Response& res) { send(res, inc.snapshot()); });
    srv.Get("/api/point", [&](const httplib::Request& req, httplib::Response& res) {
        try {
            send(res, inc.point_info(std::stod(req.get_param_value("lat")), std::stod(req.get_param_value("lon")),
                                     req.get_param_value("plan") == "1"));
        } catch (...) { send(res, {{"error", "lat and lon required"}}); }
    });
    srv.Get("/api/fuels", [&](const httplib::Request&, httplib::Response& res) {
        json j = json::array();
        for (auto& f : fuel_table()) j.push_back({{"name", f.name}, {"label", f.label}});
        send(res, j);
    });
    auto guarded = [&](auto fn) {
        return [&, fn](const httplib::Request& req, httplib::Response& res) {
            try { send(res, fn(body(req), req)); }
            catch (const std::exception& e) { send(res, {{"error", std::string("Bad request: ") + e.what()}}); }
        };
    };
    srv.Post("/api/fire", guarded([&](const json& b, const httplib::Request&) { return inc.start(b); }));
    srv.Post("/api/observe", guarded([&](const json& b, const httplib::Request&) { return inc.observe(b); }));
    srv.Post(R"(/api/local/(weather|perimeter|hotspots|fuel))", guarded([&](json b, const httplib::Request& req) {
                 b["type"] = req.matches[1].str();
                 return inc.observe(b);
             }));
    srv.Post("/api/destination", guarded([&](const json& b, const httplib::Request&) { return inc.add_destination(b); }));
    srv.Post("/api/plan", guarded([&](const json& b, const httplib::Request&) { return inc.set_plan(b); }));
    srv.Post("/api/refresh", guarded([&](const json& b, const httplib::Request&) { return inc.refresh(b); }));
    srv.Post("/api/resources", guarded([&](const json& b, const httplib::Request&) { return inc.set_resources(b); }));
    srv.Post("/api/settings", guarded([&](const json& b, const httplib::Request&) { return inc.set_settings(b); }));
    srv.Post("/api/reset", guarded([&](const json&, const httplib::Request&) { return inc.reset(); }));

    std::atomic<bool> stop{false};
    std::thread watcher([&] { watch_dropin(inc, a.dropin, stop); });

    std::printf("WILSON Rapid running → http://%s:%d   (UI: %s, cache: %s, drop-in: %s)\n",
                a.host == "0.0.0.0" ? "localhost" : a.host.c_str(), a.port, web.c_str(), a.cache.c_str(), a.dropin.c_str());
    std::fflush(stdout);
    bool ok = srv.listen(a.host, a.port);
    stop = true;
    watcher.join();
    if (!ok) {
        std::cerr << "Could not listen on " << a.host << ":" << a.port << "\n";
        return 1;
    }
    return 0;
}
