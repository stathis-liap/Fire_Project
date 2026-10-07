#include "incident.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <future>
#include <map>
#include <sstream>

#include "fetch.hpp"
#include "render.hpp"

namespace rapid {

namespace {

double now_epoch() { return double(std::time(nullptr)); }

double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

std::string clock_tok(double epoch) { return "{t:" + std::to_string(static_cast<long long>(epoch)) + "}"; }

std::string fmt_duration(double minutes) {
    int m = int(std::round(std::max(0.0, minutes) / 5.0) * 5);
    if (m < 60) return std::to_string(std::max(m, 5)) + " min";
    int h = m / 60, r = m % 60;
    return std::to_string(h) + " h" + (r ? " " + std::to_string(r) + " min" : "");
}

const char* kDangerName[6] = {"none", "low", "moderate", "high", "very high", "extreme"};

// Fire potential: 0.05 km/h → 0, 5 km/h → 1 (log scale).
double hazard_score(double ros_kmh) {
    if (ros_kmh <= 0.05) return 0;
    return std::clamp(std::log10(ros_kmh / 0.05) / 2.0, 0.0, 1.0);
}
// Exposure: an important place within 30 min of fire travel → 1, 3 h or more → 0.
double exposure_score(double minutes) { return std::clamp(1.0 - (minutes - 30.0) / 150.0, 0.0, 1.0); }

uint8_t danger_class(double ros_kmh, double minutes_to_place) {
    double h = hazard_score(ros_kmh);
    if (h <= 0) return ros_kmh > 0 ? 1 : 0;
    double risk = 0.6 * h + 0.4 * exposure_score(minutes_to_place) * std::min(1.0, 3.0 * h);
    return risk < 0.2 ? 1 : risk < 0.4 ? 2 : risk < 0.6 ? 3 : risk < 0.8 ? 4 : 5;
}

std::string grade_of(double score) { return score >= 70 ? "high" : score >= 45 ? "medium" : "low"; }

double round_to(double v, double q) { return std::round(v / q) * q; }

// Accepts [[lon,lat],...], a GeoJSON geometry, or a Feature.
std::vector<LatLon> parse_coords(const json& j) {
    std::vector<LatLon> out;
    if (j.is_object()) {
        if (j.contains("geometry")) return parse_coords(j["geometry"]);
        if (j.contains("coordinates")) {
            std::string t = j.value("type", "");
            if (t == "Polygon") return parse_coords(j["coordinates"][0]);
            if (t == "Point") return {{j["coordinates"][1].get<double>(), j["coordinates"][0].get<double>()}};
            return parse_coords(j["coordinates"]);
        }
        if (j.contains("lat") && j.contains("lon")) return {{j["lat"].get<double>(), j["lon"].get<double>()}};
        return out;
    }
    if (j.is_array()) {
        if (j.size() >= 2 && j[0].is_number()) return {{j[1].get<double>(), j[0].get<double>()}};
        for (const auto& e : j) {
            auto v = parse_coords(e);
            out.insert(out.end(), v.begin(), v.end());
        }
    }
    return out;
}

double place_radius_m(const std::string& kind) {
    if (kind == "city") return 1500;
    if (kind == "town") return 700;
    if (kind == "village") return 300;
    if (kind == "hamlet" || kind == "suburb") return 200;
    return 100;  // single buildings: hospitals, schools, ...
}

struct DestStat {
    float p10 = kInf, p25 = kInf, p50 = kInf, p75 = kInf, p90 = kInf;
    double prob = 0;
    std::vector<float> per_member;
};

DestStat dest_stat(const EnsembleResult& E, const Destination& d, float t_end) {
    DestStat s;
    const size_t n = E.arrival.size();
    if (!n || d.cells.empty()) return s;
    s.per_member.resize(n, kInf);
    for (size_t m = 0; m < n; ++m)
        for (int c : d.cells) s.per_member[m] = std::min(s.per_member[m], E.arrival[m][c]);
    std::vector<float> v = s.per_member;
    std::sort(v.begin(), v.end());
    s.p10 = v[size_t(std::floor(0.1 * (n - 1)))];
    s.p25 = v[size_t(std::floor(0.25 * (n - 1)))];
    s.p50 = v[(n - 1) / 2];
    s.p75 = v[size_t(std::ceil(0.75 * (n - 1)))];
    s.p90 = v[size_t(std::ceil(0.9 * (n - 1)))];
    int hit = 0;
    for (float a : v) hit += a <= t_end;
    s.prob = double(hit) / double(n);
    return s;
}

WeatherSeries default_weather(double from_epoch) {
    WeatherSeries w;
    for (int h = -24; h < 48; ++h) w.hourly.push_back({from_epoch + h * 3600.0, 5.0, 0.0, 30.0, 25.0});
    w.source = "default summer conditions (no forecast)";
    return w;
}

}  // namespace

Incident::Incident(IncidentOptions opt) : opt_(std::move(opt)), members_(opt_.members) {
    publish({{"status", "idle"}});
}

Incident::~Incident() {
    for (auto& t : bg_)
        if (t.joinable()) t.join();
}

void Incident::set_status(const std::string& s) {
    std::lock_guard<std::mutex> lk(snap_mu_);
    status_ = s;
}

void Incident::publish(json j) {
    std::lock_guard<std::mutex> lk(snap_mu_);
    j["version"] = ++version_;
    snapshot_ = std::move(j);
    status_ = snapshot_.value("status", "ready");
}

json Incident::snapshot() const {
    std::lock_guard<std::mutex> lk(snap_mu_);
    return snapshot_;
}

bool Incident::active() const {
    std::lock_guard<std::mutex> lk(snap_mu_);
    return snapshot_.value("status", "") == "ready";
}

json Incident::status() const {
    std::lock_guard<std::mutex> lk(snap_mu_);
    return {{"status", status_}, {"version", version_.load()}};
}

// ──────────────────────────────────────────────────────────────────────────
// Start a new incident
// ──────────────────────────────────────────────────────────────────────────
json Incident::start(const json& req) {
    std::lock_guard<std::mutex> lk(work_mu_);
    auto tic = std::chrono::steady_clock::now();
    const uint64_t gen = ++generation_;

    std::vector<LatLon> poly = req.contains("polygon") ? parse_coords(req["polygon"]) : std::vector<LatLon>{};
    LatLon pos;
    if (req.contains("lat") && req.contains("lon")) pos = {req["lat"].get<double>(), req["lon"].get<double>()};
    else if (!poly.empty()) {
        for (auto& p : poly) { pos.lat += p.lat / poly.size(); pos.lon += p.lon / poly.size(); }
    } else {
        return {{"error", "Give a position (lat, lon) or a polygon."}};
    }
    if (std::abs(pos.lat) > 85 || std::abs(pos.lon) > 180) return {{"error", "Coordinates out of range."}};

    const double now = now_epoch();
    const double ago = std::clamp(req.value("started_min_ago", 0.0), 0.0, 24 * 60.0);
    horizon_h_ = std::clamp(req.value("horizon_h", 6.0), 1.0, 24.0);
    t0_epoch_ = now - ago * 60.0;
    ignition_ = pos;
    params_ = SpreadParams{};
    calibrated_ = false;
    calibration_ = nullptr;
    obs_.clear();
    dests_.clear();
    recs_.clear();
    manual_.clear();
    disabled_.clear();
    forced_.clear();
    next_dest_id_ = next_manual_id_ = 1;
    active_ = false;

    // 1. Weather first: it sizes the domain.
    set_status("Getting weather…");
    FetchContext ctx{opt_.cache_dir, opt_.offline};
    WeatherSeries wx;
    std::vector<std::string> warnings;
    if (!fetch_weather(ctx, pos, wx)) {
        wx = default_weather(now);
        warnings.push_back("No weather forecast available — using typical summer conditions (30 °C, 25 % humidity, "
                           "18 km/h north wind). Report the local wind to improve the forecast.");
    }
    wx_ = std::move(wx);

    // 2. Domain: far enough for a fast grass fire over the whole period.
    double total_min = ago + horizon_h_ * 60.0;
    double dist = 0;
    {
        Landscape probe;
        probe.grid = Grid::centred(pos, 50, 100);
        probe.grid.rows = probe.grid.cols = 1;
        probe.elev = {0};
        derive_terrain(probe);
        probe.fuel = {uint8_t(fuel_index("Dry_Grass"))};
        probe.road = {0};
        SpreadSolver s(probe, wx_);
        for (double t = 0; t < total_min; t += 60) dist += s.head_ros(0, t0_epoch_, t, params_) * std::min(60.0, total_min - t);
    }
    double radius = std::clamp(dist * 0.9 + 2000.0, 4000.0, 30000.0);
    double cell = std::clamp(round_to(2.0 * radius / 450.0, 5.0), 25.0, 130.0);
    if (req.contains("radius_km")) radius = std::clamp(req["radius_km"].get<double>() * 1000.0, 2000.0, 40000.0);

    // 3. Terrain + land cover in parallel (cached on disk).
    set_status("Loading terrain and vegetation maps…");
    BuildOptions bo;
    bo.centre = pos;
    bo.radius_m = radius;
    bo.cell_m = cell;
    bo.cache_dir = opt_.cache_dir;
    bo.offline = opt_.offline;
    radius_m_ = radius;
    L_ = std::make_unique<Landscape>(build_landscape(bo));
    for (auto& w : warnings) L_->warnings.push_back(w);
    solver_ = std::make_unique<SpreadSolver>(*L_, wx_);
    const Grid& g = L_->grid;

    // 4. Ignition state.
    fc_sources_.clear();
    float t_ign = 0;
    if (poly.size() >= 3) {
        // A drawn area is "burning now".
        t_ign = float(ago);
        std::vector<uint8_t> m(g.size(), 0);
        rasterize_polygon(g, poly, m);
        for (size_t i = 0; i < m.size(); ++i)
            if (m[i]) fc_sources_.push_back({int(i), t_ign});
        Observation o;
        o.kind = "perimeter";
        o.epoch = now;
        o.source = "initial area";
        o.pts = poly;
        obs_.push_back(o);
    }
    if (fc_sources_.empty()) {
        int r, c;
        g.locate(pos, r, c);
        int k = nearest_burnable(*L_, r, c, 6);
        if (k < 0) return {{"error", "No vegetation that can burn at this location (water, rock or built-up)."}};
        fc_sources_.push_back({k, 0.f});
    }
    fc_time_ = t_ign;
    active_ = true;

    // 5. Places, roads and rivers arrive in the background (Overpass can be slow).
    (void)gen;
    start_feature_fetch_locked();

    set_status("Computing forecast…");
    recompute_locked();
    json s = snapshot();
    s["timing_ms"]["total_start"] = ms_since(tic);
    return s;
}

void Incident::start_feature_fetch_locked() {
    const uint64_t gen = generation_.load();
    const Grid g = L_->grid;
    bg_.emplace_back([this, gen, g] {
        MapFeatures f;
        std::string src;
        FetchContext c{opt_.cache_dir, opt_.offline};
        if (fetch_features(c, g, f, src)) attach_features(gen, std::move(f), src);
        else attach_features(gen, {}, "");
    });
}

bool Incident::fire_touches_edge_locked() const {
    const Grid& g = L_->grid;
    const int band = 3;
    for (int r = 0; r < g.rows; ++r)
        for (int c = 0; c < g.cols; ++c) {
            if (r >= band && c >= band && r < g.rows - band && c < g.cols - band) {
                c = g.cols - band - 1;  // jump over the interior
                continue;
            }
            if (base_.p10[g.idx(r, c)] <= t_end_) return true;
        }
    return false;
}

// Grows the forecast area (coarser cells) when the fire would run off the map.
void Incident::rebuild_domain_locked(double radius_m) {
    set_status("Fire may leave the map — enlarging the forecast area…");
    const Grid old = L_->grid;
    std::vector<std::pair<LatLon, float>> src_ll;
    for (auto [c, t] : fc_sources_) src_ll.push_back({old.center_of(c / old.cols, c % old.cols), t});
    MapFeatures keep_features = L_->features;
    std::string keep_src = L_->features_source;

    BuildOptions bo;
    bo.centre = ignition_;
    bo.radius_m = radius_m;
    bo.cell_m = std::clamp(round_to(2.0 * radius_m / 450.0, 5.0), 25.0, 130.0);
    bo.cache_dir = opt_.cache_dir;
    bo.offline = opt_.offline;
    radius_m_ = radius_m;
    auto L = std::make_unique<Landscape>(build_landscape(bo));
    const Grid& g = L->grid;
    for (const auto& o : obs_)  // re-apply vegetation corrections from the field
        if (o.kind == "fuel") {
            std::vector<uint8_t> m(g.size(), 0);
            rasterize_polygon(g, o.pts, m);
            for (size_t i = 0; i < m.size(); ++i)
                if (m[i]) L->fuel[i] = uint8_t(fuel_index(o.fuel));
        }
    std::map<int, float> cells;
    for (auto& [p, t] : src_ll) {
        int r, c;
        if (!g.locate(p, r, c)) continue;
        int k = g.idx(r, c);
        auto it = cells.find(k);
        if (it == cells.end() || t < it->second) cells[k] = t;
    }
    fc_sources_.assign(cells.begin(), cells.end());
    for (auto& d : dests_)
        if (d.user) {
            d.cells.clear();
            rasterize_disc(g, d.pos, 150, d.cells);
        }
    L->features = std::move(keep_features);
    L->features_source = keep_src;
    for (const auto& w : L->features.ways)
        if (w.road) {
            std::vector<int> rc;
            rasterize_line(g, w.pts, g.cell_m, rc);
            for (int c : rc) L->road[c] = 1;
        }
    L_ = std::move(L);
    solver_ = std::make_unique<SpreadSolver>(*L_, wx_);
    ++generation_;  // discard features still loading for the old area
    build_destinations_locked();
    start_feature_fetch_locked();
}

void Incident::attach_features(uint64_t generation, MapFeatures f, std::string source) {
    std::lock_guard<std::mutex> lk(work_mu_);
    if (generation != generation_.load() || !L_) return;
    if (source.empty()) {
        L_->features_source = "unavailable";
        L_->warnings.push_back("Villages and roads could not be loaded (OpenStreetMap busy or offline). "
                               "Use “Protect a place” to add places by hand.");
    } else {
        L_->features = std::move(f);
        L_->features_source = source;
        const Grid& g = L_->grid;
        for (const auto& w : L_->features.ways) {
            if (!w.road) continue;
            std::vector<int> cells;
            rasterize_line(g, w.pts, g.cell_m, cells);
            for (int c : cells) L_->road[c] = 1;
        }
    }
    build_destinations_locked();
    recompute_locked();
}

void Incident::build_destinations_locked() {
    std::vector<Destination> keep;
    for (auto& d : dests_)
        if (d.user) keep.push_back(d);
    dests_ = std::move(keep);
    const Grid& g = L_->grid;
    for (const auto& p : L_->features.places) {
        int r, c;
        if (!g.locate(p.pos, r, c)) continue;
        Destination d;
        d.id = "d" + std::to_string(next_dest_id_++);
        d.name = p.name;
        d.kind = p.kind;
        d.pos = p.pos;
        rasterize_disc(g, p.pos, place_radius_m(p.kind), d.cells);
        dests_.push_back(std::move(d));
    }
    std::map<std::string, size_t> by_name;
    for (const auto& w : L_->features.ways) {
        if (!w.major) continue;
        std::string key = w.kind + "|" + w.name;
        auto it = by_name.find(key);
        if (it == by_name.end()) {
            Destination d;
            d.id = "d" + std::to_string(next_dest_id_++);
            d.name = w.name;
            d.kind = w.road ? "road" : "river";
            d.boundary = true;
            dests_.push_back(std::move(d));
            it = by_name.emplace(key, dests_.size() - 1).first;
        }
        Destination& d = dests_[it->second];
        d.line.insert(d.line.end(), w.pts.begin(), w.pts.end());
        rasterize_line(g, w.pts, g.cell_m, d.cells);
    }
    for (auto& d : dests_) {
        std::sort(d.cells.begin(), d.cells.end());
        d.cells.erase(std::unique(d.cells.begin(), d.cells.end()), d.cells.end());
    }
}

Uncertainty Incident::uncertainty_locked(double now) const {
    Uncertainty u;
    bool local_recent = false;
    for (const auto& l : wx_.local) local_recent |= std::abs(l.s.epoch - now) < 3 * 3600;
    if (local_recent) { u.wind_speed_sigma = 0.12; u.wind_dir_sigma = 8; u.moisture_sigma = 0.008; }
    else if (wx_.source.rfind("default", 0) == 0) { u.wind_speed_sigma = 0.4; u.wind_dir_sigma = 40; u.moisture_sigma = 0.02; }
    else { u.wind_speed_sigma = 0.22; u.wind_dir_sigma = 18; }
    double score = calibrated_ ? calibration_.value("score_after", 0.0) : 0.0;
    u.ros_sigma = calibrated_ ? 0.10 + 0.25 * (1.0 - score) : 0.30;
    if (wide_) {  // "not sure about the inputs": explore a much wider range
        u.wind_speed_sigma *= 1.7;
        u.wind_dir_sigma *= 1.7;
        u.moisture_sigma *= 1.7;
        u.ros_sigma *= 1.7;
    }
    return u;
}

// ──────────────────────────────────────────────────────────────────────────
// The full forecast → destinations → tactics → what-if → JSON pipeline.
// ──────────────────────────────────────────────────────────────────────────
void Incident::recompute_locked() {
    if (!active_ || !L_) return;
    auto tic = std::chrono::steady_clock::now();
    const double now = now_epoch();
    t_now_ = float((now - t0_epoch_) / 60.0);
    t_end_ = float(t_now_ + horizon_h_ * 60.0);

    RunConfig cfg;
    cfg.t0_epoch = t0_epoch_;
    cfg.t_end = t_end_;
    cfg.sources = fc_sources_;
    cfg.p = params_;
    const Uncertainty unc = uncertainty_locked(now);
    const uint32_t seed = 20240611;

    base_ = run_ensemble(*solver_, cfg, make_members(params_, unc, members_, seed));
    for (int pass = 0; pass < 2 && radius_m_ < 30000 && fire_touches_edge_locked(); ++pass) {
        rebuild_domain_locked(std::min(30000.0, radius_m_ * 1.8));
        cfg.sources = fc_sources_;
        base_ = run_ensemble(*solver_, cfg, make_members(params_, unc, members_, seed));
    }
    const Grid& g = L_->grid;
    const RunOutput& B = base_.best;

    // ── Danger map (whole area, independent of where the fire is now) ──
    auto tic_d = std::chrono::steady_clock::now();
    {
        std::ostringstream key;
        key << g.rows << 'x' << g.cols << '@' << g.north << ',' << g.west << '|' << params_.ros_mult << ','
            << params_.wind_mult << ',' << params_.wind_dir_offset << ',' << params_.moisture_offset << '|'
            << wx_.local.size() << '|' << dests_.size() << ':' << (dests_.empty() ? "" : dests_.back().id) << '|'
            << int(t_now_ / 60) << '-' << int(t_end_ / 60) << '|' << obs_.size();
        if (key.str() != danger_key_) {
            danger_key_ = key.str();
            danger_peak_hour_ = solver_->hazard(t0_epoch_, t_now_, t_end_, params_, hz_ros_, hz_flame_, hz_dir_);
            std::vector<std::pair<int, int>> targets;
            rev_names_.clear();
            for (const auto& d : dests_) {
                if (d.boundary || place_weight(d.kind) < 2) continue;
                for (int c : d.cells) targets.push_back({c, int(rev_names_.size())});
                rev_names_.push_back(d.name);
            }
            solver_->time_to_targets(t0_epoch_, danger_peak_hour_ * 60.0 + 30.0, params_, targets, 360.f, rev_min_,
                                     rev_label_);
            danger_.assign(g.size(), 0);
            for (size_t c = 0; c < g.size(); ++c) danger_[c] = danger_class(hz_ros_[c], rev_min_[c]);
        }
    }
    const int peak_hour = danger_peak_hour_;
    double ms_danger = ms_since(tic_d);
    double ms_base = base_.runtime_ms;

    // ── Fire behaviour summary ──
    LatLon origin{0, 0};
    for (auto [c, t] : fc_sources_) {
        LatLon p = g.center_of(c / g.cols, c % g.cols);
        origin.lat += p.lat / fc_sources_.size();
        origin.lon += p.lon / fc_sources_.size();
    }
    WxSample wnow = wx_.at(now);
    double wind_to = std::fmod(wnow.wind_from + 180.0 + params_.wind_dir_offset, 360.0);
    double far_d = 0, head_bearing = wind_to;
    int far_cell = -1;
    for (size_t k = 0; k < g.size(); ++k)
        if (base_.p50[k] <= t_end_) {
            double d = haversine_m(origin, g.center_of(int(k) / g.cols, int(k) % g.cols));
            if (d > far_d) { far_d = d; far_cell = int(k); }
        }
    if (far_cell >= 0 && far_d > 2 * g.cell_m) head_bearing = bearing_deg(origin, g.center_of(far_cell / g.cols, far_cell % g.cols));
    double head_ros = 0, max_flame = 0;
    float f0 = std::max(t_now_, fc_time_);
    for (size_t k = 0; k < g.size(); ++k)
        if (B.arrival[k] >= f0 && B.arrival[k] <= f0 + 60) {
            head_ros = std::max(head_ros, double(B.ros[k]));
            max_flame = std::max(max_flame, double(B.flame_m[k]));
        }
    if (head_ros == 0)
        for (auto [c, t] : fc_sources_) {
            head_ros = std::max(head_ros, double(B.ros[c]));
            max_flame = std::max(max_flame, double(B.flame_m[c]));
        }

    const double cell_ha = g.cell_m * g.cell_m / 1e4;
    auto area_by = [&](const std::vector<float>& a, float t) {
        size_t n = 0;
        for (float v : a) n += v <= t;
        return n * cell_ha;
    };

    // ── Confidence ──
    double q = 1.0;
    json reasons = json::array();
    bool local_recent = false;
    for (const auto& l : wx_.local) local_recent |= std::abs(l.s.epoch - now) < 3 * 3600;
    if (local_recent) reasons.push_back("Wind measured on site.");
    else if (wx_.source.rfind("default", 0) == 0) { q *= 0.55; reasons.push_back("No weather forecast — report the local wind."); }
    else { q *= 0.9; reasons.push_back("Wind from the weather forecast (not measured on site)."); }
    if (!L_->fuel_ok) { q *= 0.75; reasons.push_back("Vegetation map missing — generic fuel assumed."); }
    if (!L_->terrain_ok) { q *= 0.85; reasons.push_back("Terrain missing — flat ground assumed."); }
    if (calibrated_) {
        double s = calibration_.value("score_after", 0.0);
        q *= 0.75 + 0.25 * s;
        reasons.push_back("Adjusted to the burned area seen on the ground (" + std::to_string(int(std::round(s * 100))) + "% match).");
    } else {
        q *= 0.85;
        reasons.push_back("Not yet checked against the real fire — mark the burned area to improve accuracy.");
    }
    size_t possible = 0, certain = 0;
    for (size_t k = 0; k < g.size(); ++k) {
        possible += base_.p10[k] <= t_end_;
        certain += base_.p90[k] <= t_end_;
    }
    double agree = possible ? double(certain) / double(possible) : 1.0;
    double conf_score = std::clamp(100.0 * q * (0.45 + 0.55 * agree), 5.0, 99.0);
    if (agree < 0.3) reasons.push_back("Forecasts disagree a lot on how far the fire gets.");

    // Wind shift within the horizon.
    json wind_shift = nullptr;
    json hourly = json::array();
    for (int h = 0; h <= int(horizon_h_); ++h) {
        WxSample s = wx_.at(now + h * 3600.0);
        hourly.push_back({{"epoch", now + h * 3600.0}, {"wind_kmh", std::round(s.wind_ms * 3.6)},
                          {"dir_from", std::round(s.wind_from)}, {"temp_c", std::round(s.temp_c)}, {"rh", std::round(s.rh)}});
        double turn = std::abs(std::remainder(s.wind_from - wnow.wind_from, 360.0));
        if (wind_shift.is_null() && h > 0 && turn > 45 && s.wind_ms > 2.0) {
            wind_shift = {{"epoch", now + h * 3600.0}, {"dir_from", std::round(s.wind_from)},
                          {"text", "Wind turns to blow from the " + compass_name(s.wind_from) + " around " +
                                       clock_tok(now + h * 3600.0) + " — the fire will change direction."}};
            reasons.push_back("Wind change expected during the forecast.");
            conf_score *= 0.9;
        }
    }

    // ── Destinations ──
    struct DS { const Destination* d; DestStat s; };
    std::vector<DS> dstats;
    for (const auto& d : dests_) dstats.push_back({&d, dest_stat(base_, d, t_end_)});

    // ── Threats & tactics ──
    // Plan for a reasonable worst case: action geometry comes from the
    // scenario at the 70th percentile of burned area (with fire paths), while
    // the planner still scores every action on the paired ensemble.
    RunOutput W;
    {
        std::vector<std::pair<double, size_t>> areas;
        for (size_t m = 0; m < base_.arrival.size(); ++m) areas.push_back({area_by(base_.arrival[m], t_end_), m});
        std::sort(areas.begin(), areas.end());
        size_t pick = areas[size_t(std::floor(0.7 * (areas.size() - 1)))].second;
        RunConfig wc = cfg;
        wc.p = base_.members[pick].p;
        solver_->run(wc, W);
    }
    std::vector<Threat> threats;
    for (const auto& x : dstats) {
        if (x.d->boundary || x.s.prob < 0.2) continue;
        Threat th;
        th.dest_id = x.d->id;
        th.name = x.d->name;
        th.prob = x.s.prob;
        float best = kInf;
        for (int c : x.d->cells)
            if (W.arrival[c] < best) { best = W.arrival[c]; th.cell = c; }
        if (th.cell < 0 || best >= kInf) continue;
        th.eta = best;
        threats.push_back(th);
    }
    std::sort(threats.begin(), threats.end(), [](const Threat& a, const Threat& b) { return a.eta < b.eta; });
    int head_w = -1;
    {
        double fdw = 0;
        for (size_t k = 0; k < g.size(); ++k)
            if (W.arrival[k] <= t_end_) {
                double d = haversine_m(origin, g.center_of(int(k) / g.cols, int(k) % g.cols));
                if (d > fdw) { fdw = d; head_w = int(k); }
            }
    }
    TacticsInput ti;
    ti.L = L_.get();
    ti.best = &W;
    ti.t0_epoch = t0_epoch_;
    ti.t_now = t_now_;
    ti.t_end = t_end_;
    ti.origin = origin;
    ti.head_bearing = head_bearing;
    ti.wind_kmh = wnow.wind_ms * params_.wind_mult * 3.6;
    ti.threats = threats;
    ti.head_cell = head_w;
    recs_ = plan_tactics(ti);
    std::map<std::string, int> key_count;
    std::vector<std::string> keys;
    for (auto& r : recs_) {
        std::string base = r.type + ":" + (r.target_name.empty() ? "-" : r.target_name);
        keys.push_back(base + ":" + std::to_string(key_count[base]++));
        r.forced = forced_.count(keys.back()) > 0;
        r.enabled = !disabled_.count(keys.back());  // eligible for the planner
    }

    // The user's own actions, always in the plan.
    std::vector<double> manual_breach;
    for (auto& m : manual_) {
        double fl = 0;
        if (!m.pts.empty()) {
            int rr, cc;
            if (g.locate(m.pts[m.pts.size() / 2], rr, cc)) fl = B.flame_m[g.idx(rr, cc)];
        }
        manual_breach.push_back(m.kind == IvKind::Firebreak ? std::clamp(0.05 + 0.07 * std::max(0.0, fl - 1.5), 0.05, 0.85) : 0.0);
    }

    // ── Resource planner: the best plan for what the commander has ──
    PlannerInput pin;
    pin.solver = solver_.get();
    pin.cfg = cfg;
    const int K = std::min<int>(6, int(base_.members.size()));
    pin.members.assign(base_.members.begin(), base_.members.begin() + K);
    pin.base_arrival = &base_.arrival;
    for (auto& x : dstats) {
        bool threatened = false;
        for (int m = 0; m < K && !threatened; ++m)
            for (int c : x.d->cells)
                if (base_.arrival[m][c] <= t_end_) { threatened = true; break; }
        if (threatened) pin.targets.push_back({x.d->id, x.d->name, place_weight(x.d->kind), x.d->cells});
    }
    pin.fixed = manual_;
    pin.fixed_breach = manual_breach;
    pin.t_now = t_now_;
    pin.t_end = t_end_;
    pin.cell_ha = cell_ha;
    set_status("Choosing the best plan for your resources…");
    PlannerOutput pout = choose_plan(pin, recs_, resources_);

    // ── What-if: run the full ensemble with the chosen plan ──
    std::vector<Intervention> ivs;
    std::vector<double> breach;
    for (auto& r : recs_) {
        r.enabled = r.simulated && (r.selected || r.forced);  // now: "in the plan"
        if (r.enabled) { ivs.push_back(r.iv); breach.push_back(r.breach_p); }
    }
    for (size_t i = 0; i < manual_.size(); ++i) { ivs.push_back(manual_[i]); breach.push_back(manual_breach[i]); }
    InterventionRaster raster;
    raster.build(g, ivs);
    for (auto& l : raster.layers) l.breach_p = float(breach[l.src]);
    plan_active_ = !raster.empty();
    double ms_plan = 0;
    if (plan_active_) {
        RunConfig pc = cfg;
        pc.iv = &raster;
        plan_ = run_ensemble(*solver_, pc, make_members(params_, unc, members_, seed, &raster));
        ms_plan = plan_.runtime_ms;
    }
    for (auto& r : recs_)
        if (!r.simulated) {
            r.enabled = true;
            for (auto& x : dstats)
                if (x.d->id == r.target_id) r.confidence = 100.0 * x.s.prob;
        }

    // ── Assemble JSON ──
    json out;
    out["status"] = "ready";
    out["incident"] = {{"t0_epoch", t0_epoch_}, {"now_epoch", now}, {"horizon_h", horizon_h_},
                       {"t_now_min", t_now_}, {"end_epoch", t0_epoch_ + t_end_ * 60.0},
                       {"ignition", {{"lat", ignition_.lat}, {"lon", ignition_.lon}}}};
    out["grid"] = {{"west", g.west}, {"east", g.east()}, {"north", g.north}, {"south", g.south()},
                   {"rows", g.rows}, {"cols", g.cols}, {"cell_m", g.cell_m}};
    json coords = {{g.west, g.north}, {g.east(), g.north}, {g.east(), g.south()}, {g.west, g.south()}};

    out["weather"] = {{"now", {{"wind_kmh", std::round(wnow.wind_ms * 3.6)}, {"dir_from", std::round(wnow.wind_from)},
                               {"dir_text", compass_name(wnow.wind_from)}, {"temp_c", std::round(wnow.temp_c)},
                               {"rh", std::round(wnow.rh)}, {"source", local_recent ? "measured on site" : wx_.source}}},
                      {"shift", wind_shift}, {"hourly", hourly}};

    std::string attack;
    if (max_flame < 1.2) attack = "Low intensity: hand crews can attack directly.";
    else if (max_flame < 2.4) attack = "Moderate: engines and hand crews can attack the edge.";
    else if (max_flame < 3.4) attack = "High: use heavy equipment and aircraft at the head.";
    else attack = "Very high: do not attack the head directly — work the flanks, cut breaks ahead, protect people.";
    json area_h = json::array();
    for (int h = 0; h <= int(horizon_h_); ++h) area_h.push_back(std::round(area_by(base_.p50, t_now_ + h * 60.f)));
    out["summary"] = {{"direction_deg", std::round(head_bearing)}, {"direction_text", compass_name(head_bearing)},
                      {"head_speed_kmh", std::round(head_ros * 0.06 * 10) / 10}, {"max_flame_m", std::round(max_flame * 10) / 10},
                      {"attack_advice", attack}, {"area_ha_now", std::round(area_by(base_.p50, t_now_))},
                      {"area_ha_end", std::round(area_by(base_.p50, t_end_))}, {"area_ha_by_hour", area_h},
                      {"head_reach_km", std::round(far_d / 100) / 10}};
    out["confidence"] = {{"score", std::round(conf_score)}, {"grade", grade_of(conf_score)}, {"reasons", reasons}};
    out["calibration"] = calibration_;

    auto layer_json = [&](const EnsembleResult& E) {
        json iso = {{"type", "FeatureCollection"}, {"features", json::array()}};
        if (t_now_ > fc_time_ + 5 || fc_time_ > 0)
            for (auto& line : contour_lines(g, E.p50, t_now_))
                iso["features"].push_back(feature(geo_line(line), {{"hours", 0}, {"label", "now"}}));
        for (int h = 1; h <= int(horizon_h_); ++h)
            for (auto& line : contour_lines(g, E.p50, t_now_ + h * 60.f))
                iso["features"].push_back(feature(geo_line(line), {{"hours", h}, {"label", "+" + std::to_string(h) + " h"},
                                                                   {"clock", clock_tok(now + h * 3600.0)}}));
        json prob = json::array();
        for (auto& u : render_prob_pngs(g, E.arrival, t_now_, int(horizon_h_))) prob.push_back(u);
        return json{{"overlay", {{"url", render_zones_png(g, E.p50, E.p10, t_now_, t_end_)}, {"coordinates", coords}}},
                    {"isochrones", iso}, {"probability", prob}};
    };
    out["base"] = layer_json(base_);
    {
        // Per place: the 1-hour danger zone — ground from which a fire would
        // reach it within an hour — and the direction the threat comes from.
        struct Zone { double cells = 0, far = 0, brg = 0, ros = 0; };
        std::vector<Zone> zones(rev_names_.size());
        std::vector<LatLon> anchor(rev_names_.size());
        {
            size_t li = 0;
            for (const auto& d : dests_)
                if (!d.boundary && place_weight(d.kind) >= 2) anchor[li++] = d.pos;
        }
        for (size_t c = 0; c < g.size(); ++c) {
            int l = rev_label_[c];
            if (l < 0 || rev_min_[c] > 60) continue;
            Zone& z = zones[l];
            z.cells += 1;
            z.ros = std::max(z.ros, double(hz_ros_[c]));
            LatLon p = g.center_of(int(c) / g.cols, int(c) % g.cols);
            double d = haversine_m(anchor[l], p);
            if (d > z.far) { z.far = d; z.brg = bearing_deg(anchor[l], p); }
        }
        json places = json::array();
        std::vector<size_t> ord(zones.size());
        for (size_t i = 0; i < ord.size(); ++i) ord[i] = i;
        std::sort(ord.begin(), ord.end(), [&](size_t a, size_t b) { return zones[a].far > zones[b].far; });
        for (size_t i : ord) {
            if (zones[i].far < 2 * g.cell_m || places.size() >= 10) continue;
            places.push_back({{"name", rev_names_[i]}, {"lat", anchor[i].lat}, {"lon", anchor[i].lon},
                              {"zone_ha", std::round(zones[i].cells * cell_ha)},
                              {"reach_km", std::round(zones[i].far / 100) / 10},
                              {"from_dir", compass_name(zones[i].brg)},
                              {"max_ros_kmh", std::round(zones[i].ros * 10) / 10}});
        }
        json area_by_class = json::array();
        std::vector<double> cnt(6, 0);
        for (uint8_t c : danger_) cnt[c] += cell_ha;
        for (int k = 1; k <= 5; ++k) area_by_class.push_back({{"class", kDangerName[k]}, {"ha", std::round(cnt[k])}});
        uint8_t here = 0;
        for (auto [c, t] : fc_sources_) here = std::max(here, danger_[c]);
        out["danger"] = {{"overlay", render_danger_png(g, danger_)},
                         {"peak_epoch", t0_epoch_ + (peak_hour * 60.0 + 30.0) * 60.0},
                         {"fire_area_class", kDangerName[here]}, {"fire_area_level", here},
                         {"places", places}, {"area_by_class", area_by_class}};
    }
    {   // Area worth looking at: everything the fire might reach, plus the ignition.
        int r0 = g.rows, r1 = -1, c0 = g.cols, c1 = -1;
        for (int r = 0; r < g.rows; ++r)
            for (int c = 0; c < g.cols; ++c)
                if (base_.p10[g.idx(r, c)] <= t_end_) { r0 = std::min(r0, r); r1 = std::max(r1, r); c0 = std::min(c0, c); c1 = std::max(c1, c); }
        if (r1 < 0) { r0 = r1 = g.rows / 2; c0 = c1 = g.cols / 2; }
        LatLon nw = g.center_of(r0, c0), se = g.center_of(r1, c1);
        out["extent"] = {std::min(nw.lon, ignition_.lon), std::min(se.lat, ignition_.lat),
                         std::max(se.lon, ignition_.lon), std::max(nw.lat, ignition_.lat)};
    }

    // Spread arrows: per 30° sector, the furthest point the fire reaches.
    {
        json arrows = {{"type", "FeatureCollection"}, {"features", json::array()}};
        const int S = 12;
        std::vector<int> far(S, -1);
        std::vector<double> fd(S, 0);
        for (size_t k = 0; k < g.size(); ++k) {
            if (B.arrival[k] > t_end_ || B.pred[k] < 0) continue;
            LatLon p = g.center_of(int(k) / g.cols, int(k) % g.cols);
            double d = haversine_m(origin, p);
            int s = int(bearing_deg(origin, p) / (360.0 / S)) % S;
            if (d > fd[s]) { fd[s] = d; far[s] = int(k); }
        }
        double dmax = *std::max_element(fd.begin(), fd.end());
        float t_from = std::max(fc_time_, t_end_ - float(horizon_h_ * 60.0 * 0.6));
        for (int s = 0; s < S; ++s) {
            if (far[s] < 0 || fd[s] < 0.35 * dmax || fd[s] < 4 * g.cell_m) continue;
            std::vector<int> path = trace_path(B, far[s]);
            std::vector<LatLon> pts;
            for (int k : path) {
                pts.push_back(g.center_of(k / g.cols, k % g.cols));
                if (B.arrival[k] <= t_from) break;
            }
            if (pts.size() < 3) continue;
            std::reverse(pts.begin(), pts.end());
            std::vector<LatLon> simp;
            size_t stride = std::max<size_t>(1, pts.size() / 6);
            for (size_t i = 0; i < pts.size(); i += stride) simp.push_back(pts[i]);
            if (simp.back().lat != pts.back().lat || simp.back().lon != pts.back().lon) simp.push_back(pts.back());
            double brg = bearing_deg(simp[simp.size() - 2], simp.back());
            bool head = fd[s] == dmax;
            arrows["features"].push_back(feature(geo_line(simp), {{"kind", head ? "head" : "front"},
                                                                  {"bearing", std::round(brg)},
                                                                  {"ros_kmh", std::round(B.ros[far[s]] * 0.6) / 10}}));
            arrows["features"].push_back(feature(geo_point(simp.back()), {{"kind", head ? "head" : "front"}, {"bearing", std::round(brg)}}));
        }
        out["base"]["arrows"] = arrows;
    }

    if (plan_active_) {
        out["plan"] = layer_json(plan_);
        double a0 = area_by(base_.p50, t_end_), a1 = area_by(plan_.p50, t_end_);
        out["plan"]["area_ha_end"] = std::round(a1);
        out["plan"]["saved_ha"] = std::round(std::max(0.0, a0 - a1));
    } else {
        out["plan"] = nullptr;
    }

    // Observed fire: latest perimeter + hotspots.
    json burned = {{"type", "FeatureCollection"}, {"features", json::array()}};
    for (auto it = obs_.rbegin(); it != obs_.rend(); ++it)
        if (it->kind == "perimeter") {
            json ring = json::array();
            for (auto& p : it->pts) ring.push_back({p.lon, p.lat});
            if (!it->pts.empty()) ring.push_back({it->pts[0].lon, it->pts[0].lat});
            burned["features"].push_back(feature({{"type", "Polygon"}, {"coordinates", {ring}}},
                                                 {{"kind", "perimeter"}, {"epoch", it->epoch}, {"source", it->source}}));
            break;
        }
    for (auto& o : obs_)
        if (o.kind == "hotspots")
            for (auto& p : o.pts)
                burned["features"].push_back(feature(geo_point(p), {{"kind", "hotspot"}, {"epoch", o.epoch}, {"source", o.source}}));
    out["observed"] = burned;

    // Destinations, most urgent first.
    json dj = json::array();
    std::sort(dstats.begin(), dstats.end(), [](const DS& a, const DS& b) {
        if (a.s.p50 != b.s.p50) return a.s.p50 < b.s.p50;
        return a.s.prob > b.s.prob;
    });
    std::map<std::string, DestStat> plan_stats;
    if (plan_active_)
        for (auto& x : dstats) plan_stats[x.d->id] = dest_stat(plan_, *x.d, t_end_);
    int listed = 0;
    for (auto& x : dstats) {
        if (!x.d->user && x.s.prob < 0.05) continue;
        if (!x.d->user && ++listed > 30) continue;
        const DestStat& s = x.s;
        double qd;
        if (s.prob >= 0.5) {
            // Timing spread from the inter-quartile range: the slowest members
            // often never arrive, which the probability already accounts for.
            double rel = (std::min(s.p75, t_end_) - s.p25) / std::max(45.0, double(s.p50 - t_now_));
            qd = q * (0.4 + 0.6 * std::abs(2 * s.prob - 1)) / (1.0 + 0.6 * rel);
        } else {
            qd = q * (0.4 + 0.6 * std::abs(2 * s.prob - 1));
        }
        double dconf = std::clamp(100.0 * qd, 5.0, 99.0);
        json e = {{"id", x.d->id}, {"name", x.d->name}, {"kind", x.d->kind}, {"user", x.d->user},
                  {"boundary", x.d->boundary}, {"lat", x.d->pos.lat}, {"lon", x.d->pos.lon},
                  {"prob", std::round(s.prob * 100)}, {"confidence", std::round(dconf)}, {"grade", grade_of(dconf)}};
        if (x.d->boundary) {
            float best = kInf;
            int bc = -1;
            for (int c : x.d->cells)
                if (base_.p50[c] < best) { best = base_.p50[c]; bc = c; }
            if (bc < 0) bc = x.d->cells.empty() ? -1 : x.d->cells.front();
            if (bc >= 0) {
                LatLon p = g.center_of(bc / g.cols, bc % g.cols);
                e["lat"] = p.lat;
                e["lon"] = p.lon;
            }
        }
        auto put_eta = [&](json& o, const DestStat& st) {
            if (st.p50 < kInf) {
                o["eta_epoch"] = t0_epoch_ + st.p50 * 60.0;
                o["eta_min"] = std::round(st.p50 - t_now_);
            } else {
                o["eta_epoch"] = nullptr;
                o["eta_min"] = nullptr;
            }
            o["earliest_epoch"] = st.p10 < kInf ? json(t0_epoch_ + st.p10 * 60.0) : json(nullptr);
            o["latest_epoch"] = st.p90 < kInf ? json(t0_epoch_ + st.p90 * 60.0) : json(nullptr);
            o["burning_now"] = st.p50 <= t_now_;
        };
        put_eta(e, s);
        if (plan_active_) {
            json pj;
            const DestStat& ps = plan_stats[x.d->id];
            put_eta(pj, ps);
            pj["prob"] = std::round(ps.prob * 100);
            if (s.p50 < kInf && ps.p50 >= kInf) pj["delay_text"] = "fire kept out";
            else if (s.p50 < kInf && ps.p50 - s.p50 >= 5) pj["delay_text"] = "+" + fmt_duration(ps.p50 - s.p50) + " later";
            e["with_plan"] = pj;
        }
        dj.push_back(e);
    }
    out["destinations"] = dj;

    json rj = json::array();
    for (size_t i = 0; i < recs_.size(); ++i) {
        const auto& r = recs_[i];
        json geom;
        if (r.type == "firebreak") geom = geo_line(r.iv.pts);
        else if (r.type == "air_drop") {
            LatLon c = r.iv.pts[0];
            geom = geo_line({offset_m(c, r.iv.bearing, -r.iv.length_m / 2), offset_m(c, r.iv.bearing, r.iv.length_m / 2)});
        } else geom = geo_point(r.iv.pts.empty() ? origin : r.iv.pts[0]);
        rj.push_back({{"id", r.id}, {"key", keys[i]}, {"type", r.type}, {"order", r.order}, {"title", r.title},
                      {"detail", r.detail}, {"target_id", r.target_id}, {"target_name", r.target_name},
                      {"act_before_epoch", t0_epoch_ + r.act_before * 60.0}, {"enabled", r.enabled},
                      {"selected", r.selected}, {"forced", r.forced}, {"resource", r.resource},
                      {"gain", std::round(r.gain * 100) / 100},
                      {"simulated", r.simulated}, {"confidence", std::round(r.confidence)},
                      {"grade", grade_of(r.confidence)}, {"effect", r.effect}, {"geometry", geom}});
    }
    out["recommendations"] = rj;
    const char* rnames[3] = {"aircraft", "trucks", "dozers"};
    json rs;
    for (int k = 0; k < 3; ++k) rs[rnames[k]] = {{"available", pout.available[k]}, {"used", pout.used[k]}};
    out["resources"] = rs;
    out["plan_notes"] = pout.notes;
    out["settings"] = {{"members", members_}, {"wide", wide_}};

    json mj = json::array();
    for (const auto& m : manual_) {
        json geom = m.kind == IvKind::Firebreak ? geo_line(m.pts)
                    : m.kind == IvKind::AirDrop
                        ? geo_line({offset_m(m.pts[0], m.bearing, -m.length_m / 2), offset_m(m.pts[0], m.bearing, m.length_m / 2)})
                        : geo_point(m.pts[0]);
        mj.push_back({{"id", m.id}, {"type", iv_kind_name(m.kind)}, {"geometry", geom}, {"active_from_epoch", t0_epoch_ + m.t_on * 60.0}});
    }
    out["manual"] = mj;

    json local = json::array();
    for (auto& o : obs_) local.push_back({{"kind", o.kind}, {"epoch", o.epoch}, {"source", o.source}});
    out["sources"] = {{"terrain", L_->terrain_source}, {"fuel", L_->fuel_source}, {"weather", wx_.source},
                      {"places", L_->features_source}, {"local", local}};
    out["warnings"] = L_->warnings;
    out["params"] = {{"ros_mult", params_.ros_mult}, {"wind_mult", params_.wind_mult},
                     {"wind_dir_offset", params_.wind_dir_offset}, {"moisture_offset", params_.moisture_offset}};
    out["timing_ms"] = {{"ensemble", std::round(ms_base)}, {"plan", std::round(ms_plan)}, {"recompute", std::round(ms_since(tic))},
                        {"members", members_}, {"danger", std::round(ms_danger)}, {"planner", std::round(pout.runtime_ms)}, {"planner_runs", pout.simulations}};
    publish(std::move(out));
}

// ──────────────────────────────────────────────────────────────────────────
// Local measurements and drone / crew observations
// ──────────────────────────────────────────────────────────────────────────
json Incident::observe(const json& obs) {
    std::lock_guard<std::mutex> lk(work_mu_);
    if (!active_) return {{"error", "No active fire. Mark the fire on the map first."}};
    json r = observe_locked(obs);
    if (r.contains("error")) return r;
    set_status("Updating forecast…");
    recompute_locked();
    json s = snapshot();
    s["observation_result"] = r;
    return s;
}

json Incident::observe_locked(const json& obs) {
    const Grid& g = L_->grid;
    const std::string kind = obs.value("type", obs.value("kind", ""));
    const double now = now_epoch();
    double epoch = obs.value("time", now);
    if (epoch > 1e11) epoch /= 1000.0;  // accept JS milliseconds
    const std::string source = obs.value("source", "field report");
    const float t_obs = float((epoch - t0_epoch_) / 60.0);

    if (kind == "weather") {
        LocalWx l;
        WxSample f = wx_.at(epoch);
        l.s = f;
        l.s.epoch = epoch;
        if (obs.contains("wind_kmh")) l.s.wind_ms = obs["wind_kmh"].get<double>() / 3.6;
        if (obs.contains("wind_speed_ms")) l.s.wind_ms = obs["wind_speed_ms"].get<double>();
        if (obs.contains("wind_dir_deg")) l.s.wind_from = obs["wind_dir_deg"].get<double>();
        if (obs.contains("temp_c")) l.s.temp_c = obs["temp_c"].get<double>();
        if (obs.contains("rh")) l.s.rh = obs["rh"].get<double>();
        l.source = source;
        wx_.local.push_back(l);
        // On-site wind replaces any wind bias the calibration had learned.
        params_.wind_mult = 1.0;
        params_.wind_dir_offset = 0.0;
        obs_.push_back({"weather", epoch, source, {}, {}});
        return {{"ok", true}, {"message", "Local weather applied."}};
    }

    if (kind == "fuel") {
        auto ring = parse_coords(obs.contains("polygon") ? obs["polygon"] : obs["geometry"]);
        std::string fname = obs.value("fuel", "Non_Combustible");
        if (fname == "burned" || fname == "cleared") fname = "Non_Combustible";
        int fi = fuel_index(fname);
        if (fi < 0 || ring.size() < 3) return {{"error", "Unknown fuel or bad polygon."}};
        std::vector<uint8_t> m(g.size(), 0);
        rasterize_polygon(g, ring, m);
        for (size_t i = 0; i < m.size(); ++i)
            if (m[i]) L_->fuel[i] = uint8_t(fi);
        obs_.push_back({"fuel", epoch, source, ring, fname});
        return {{"ok", true}, {"message", "Vegetation updated."}};
    }

    if (kind != "perimeter" && kind != "hotspots")
        return {{"error", "Unknown observation type '" + kind + "' (use weather, perimeter, hotspots or fuel)."}};
    if (t_obs < fc_time_ - 1) return {{"error", "Observation is older than the latest known fire state."}};

    CalibrationInput ci;
    ci.sources = fc_sources_;
    ci.t_obs = t_obs;
    ci.start = params_;
    Observation o{kind, epoch, source, {}, {}};
    std::vector<std::pair<int, float>> new_sources;

    if (kind == "perimeter") {
        o.pts = parse_coords(obs.contains("polygon") ? obs["polygon"] : obs["geometry"]);
        if (o.pts.size() < 3) return {{"error", "A burned area needs at least 3 points."}};
        ci.observed.assign(g.size(), 0);
        rasterize_polygon(g, o.pts, ci.observed);
        size_t n = 0;
        for (size_t i = 0; i < g.size(); ++i)
            if (ci.observed[i]) {
                ++n;
                new_sources.push_back({int(i), t_obs});
            }
        if (n == 0) return {{"error", "The burned area is outside the map area."}};
    } else {
        o.pts = parse_coords(obs.contains("points") ? obs["points"] : obs["geometry"]);
        for (auto& p : o.pts) {
            int r, c;
            if (!g.locate(p, r, c)) continue;
            int k = nearest_burnable(*L_, r, c, 2);
            if (k < 0) continue;
            ci.hotspots.push_back(k);
            new_sources.push_back({k, t_obs});
        }
        if (ci.hotspots.empty()) return {{"error", "No hotspot falls on burnable ground inside the map."}};
        // Hotspots alone only say "burning here", which any too-fast model
        // also satisfies.  Treat the hull of the hotspots plus the previous
        // fire state as the approximate burned area so over-prediction costs.
        std::vector<std::pair<double, double>> pts;
        for (auto [c, t] : fc_sources_) pts.push_back({double(c % g.cols), double(c / g.cols)});
        for (int c : ci.hotspots) pts.push_back({double(c % g.cols), double(c / g.cols)});
        std::sort(pts.begin(), pts.end());
        pts.erase(std::unique(pts.begin(), pts.end()), pts.end());
        if (pts.size() >= 3) {
            auto cross = [](auto o, auto a, auto b) {
                return (a.first - o.first) * (b.second - o.second) - (a.second - o.second) * (b.first - o.first);
            };
            std::vector<std::pair<double, double>> hull(2 * pts.size());
            size_t k = 0;
            for (size_t i = 0; i < pts.size(); ++i) {
                while (k >= 2 && cross(hull[k - 2], hull[k - 1], pts[i]) <= 0) --k;
                hull[k++] = pts[i];
            }
            for (size_t i = pts.size() - 1, t = k + 1; i-- > 0;) {
                while (k >= t && cross(hull[k - 2], hull[k - 1], pts[i]) <= 0) --k;
                hull[k++] = pts[i];
            }
            hull.resize(k - 1);
            if (hull.size() >= 3) {
                // Centroid-relative 1-cell buffer, then rasterise.
                double cx = 0, cy = 0;
                for (auto& h : hull) { cx += h.first / hull.size(); cy += h.second / hull.size(); }
                std::vector<LatLon> ring;
                for (auto& h : hull) {
                    double dx = h.first - cx, dy = h.second - cy, d = std::max(1e-6, std::hypot(dx, dy));
                    ring.push_back(g.center_of(int(std::round(h.second + dy / d * 1.5)), int(std::round(h.first + dx / d * 1.5))));
                }
                ci.observed.assign(g.size(), 0);
                rasterize_polygon(g, ring, ci.observed);
                for (int c : ci.hotspots) ci.observed[c] = 1;
            }
        }
    }

    json result = {{"ok", true}};
    // Calibrate only when the fire had time to move since the last known state.
    if (t_obs - fc_time_ >= 10) {
        set_status("Matching the model to the real fire…");
        CalibrationResult cr = calibrate(*solver_, t0_epoch_, ci);
        params_ = cr.params;
        calibrated_ = true;
        calibration_ = {{"score_before", std::round(cr.score_before * 100) / 100},
                        {"score_after", std::round(cr.score_after * 100) / 100},
                        {"ros_mult", cr.params.ros_mult}, {"wind_mult", cr.params.wind_mult},
                        {"wind_dir_offset", cr.params.wind_dir_offset}, {"moisture_offset", cr.params.moisture_offset},
                        {"evaluations", cr.evaluations}, {"runtime_ms", std::round(cr.runtime_ms)}, {"epoch", epoch},
                        {"text", "Model matched to the observed fire: " + std::to_string(int(std::round(cr.score_before * 100))) +
                                     "% → " + std::to_string(int(std::round(cr.score_after * 100))) + "% agreement."}};
        result["calibration"] = calibration_;
    }
    // Forecast continues from what was observed.
    if (kind == "perimeter") {
        fc_sources_ = std::move(new_sources);
    } else {
        for (auto& s : new_sources) fc_sources_.push_back(s);
    }
    fc_time_ = t_obs;
    obs_.push_back(o);
    return result;
}

json Incident::add_destination(const json& req) {
    std::lock_guard<std::mutex> lk(work_mu_);
    if (!active_) return {{"error", "No active fire."}};
    if (req.contains("remove")) {
        std::string id = req["remove"].get<std::string>();
        dests_.erase(std::remove_if(dests_.begin(), dests_.end(), [&](const Destination& d) { return d.user && d.id == id; }), dests_.end());
    } else {
        Destination d;
        d.id = "u" + std::to_string(next_dest_id_++);
        d.name = req.value("name", "Protected place");
        d.kind = "user";
        d.user = true;
        auto pts = parse_coords(req.contains("geometry") ? req["geometry"] : req);
        if (pts.empty()) return {{"error", "A place needs lat/lon or a GeoJSON point."}};
        d.pos = pts[0];
        int r, c;
        if (!L_->grid.locate(d.pos, r, c)) return {{"error", "That place is outside the forecast area."}};
        rasterize_disc(L_->grid, d.pos, req.value("radius_m", 150.0), d.cells);
        dests_.push_back(std::move(d));
    }
    recompute_locked();
    return snapshot();
}

json Incident::set_plan(const json& req) {
    std::lock_guard<std::mutex> lk(work_mu_);
    if (!active_) return {{"error", "No active fire."}};
    const double t_now = (now_epoch() - t0_epoch_) / 60.0;
    if (req.contains("toggle")) {
        // On = force into the plan (even beyond the budget); off = never use it.
        std::string key = req["toggle"].value("key", "");
        if (req["toggle"].value("enabled", true)) { disabled_.erase(key); forced_.insert(key); }
        else { forced_.erase(key); disabled_.insert(key); }
    }
    if (req.value("reset_choices", false)) { disabled_.clear(); forced_.clear(); }
    if (req.contains("add")) {
        const json& a = req["add"];
        std::string type = a.value("type", "");
        Intervention iv;
        iv.id = "m" + std::to_string(next_manual_id_++);
        iv.pts = parse_coords(a.contains("points") ? a["points"] : a["geometry"]);
        if (iv.pts.empty()) return {{"error", "No position given."}};
        if (type == "firebreak") {
            if (iv.pts.size() < 2) return {{"error", "A firebreak needs at least two points."}};
            double len = 0;
            for (size_t i = 1; i < iv.pts.size(); ++i) len += haversine_m(iv.pts[i - 1], iv.pts[i]);
            iv.kind = IvKind::Firebreak;
            iv.width_m = a.value("width_m", 15.0);
            iv.t_on = t_now + a.value("ready_in_min", std::max(20.0, len / 1200.0 * 60.0));
        } else if (type == "air_drop") {
            iv.kind = IvKind::AirDrop;
            if (iv.pts.size() >= 2) {
                LatLon c{(iv.pts[0].lat + iv.pts[1].lat) / 2, (iv.pts[0].lon + iv.pts[1].lon) / 2};
                iv.bearing = bearing_deg(iv.pts[0], iv.pts[1]);
                iv.length_m = std::clamp(haversine_m(iv.pts[0], iv.pts[1]), 100.0, 1500.0);
                iv.pts = {c};
            } else {
                json snap = snapshot();
                double brg = snap.contains("summary") ? snap["summary"].value("direction_deg", 0.0) : 0.0;
                iv.bearing = std::fmod(brg + 90.0, 360.0);
                iv.length_m = 400;
            }
            iv.width_m = 40;
            iv.t_on = t_now + a.value("ready_in_min", 30.0);
            iv.t_off = iv.t_on + 90;
        } else if (type == "truck") {
            iv.kind = IvKind::Truck;
            iv.pts.resize(1);
            iv.radius_m = a.value("radius_m", 150.0);
            iv.t_on = t_now + a.value("ready_in_min", 25.0);
            iv.t_off = iv.t_on + 360;
        } else {
            return {{"error", "Unknown action type (firebreak, air_drop, truck)."}};
        }
        manual_.push_back(iv);
    }
    if (req.contains("remove")) {
        std::string id = req["remove"].get<std::string>();
        manual_.erase(std::remove_if(manual_.begin(), manual_.end(), [&](const Intervention& m) { return m.id == id; }), manual_.end());
    }
    if (req.value("clear_manual", false)) manual_.clear();
    recompute_locked();
    return snapshot();
}

json Incident::refresh(const json& req) {
    std::lock_guard<std::mutex> lk(work_mu_);
    if (!active_) return {{"error", "No active fire."}};
    if (req.contains("horizon_h")) horizon_h_ = std::clamp(req["horizon_h"].get<double>(), 1.0, 24.0);
    if (req.value("refetch_weather", false)) {
        FetchContext ctx{opt_.cache_dir, opt_.offline};
        WeatherSeries w;
        if (fetch_weather(ctx, ignition_, w)) {
            w.local = wx_.local;
            wx_.hourly = std::move(w.hourly);
            wx_.source = w.source;
        }
    }
    recompute_locked();
    return snapshot();
}

json Incident::set_resources(const json& req) {
    std::lock_guard<std::mutex> lk(work_mu_);
    resources_.aircraft = std::clamp(req.value("aircraft", resources_.aircraft), 0, 20);
    resources_.trucks = std::clamp(req.value("trucks", resources_.trucks), 0, 50);
    resources_.dozers = std::clamp(req.value("dozers", resources_.dozers), 0, 10);
    if (!active_) return {{"ok", true}};
    recompute_locked();
    return snapshot();
}

json Incident::set_settings(const json& req) {
    std::lock_guard<std::mutex> lk(work_mu_);
    members_ = std::clamp(req.value("members", members_), 8, 200);
    wide_ = req.value("wide", wide_);
    if (!active_) return {{"ok", true}};
    set_status("Running " + std::to_string(members_) + " scenarios…");
    recompute_locked();
    return snapshot();
}

json Incident::reset() {
    std::lock_guard<std::mutex> lk(work_mu_);
    ++generation_;
    active_ = false;
    solver_.reset();
    L_.reset();
    obs_.clear();
    dests_.clear();
    manual_.clear();
    recs_.clear();
    publish({{"status", "idle"}});
    return snapshot();
}

json Incident::point_info(double lat, double lon, bool plan) const {
    std::lock_guard<std::mutex> lk(work_mu_);
    if (!active_ || !L_) return {{"error", "No active fire."}};
    const Grid& g = L_->grid;
    int r, c;
    if (!g.locate({lat, lon}, r, c)) return {{"inside", false}};
    const int k = g.idx(r, c);
    const EnsembleResult& E = (plan && plan_active_) ? plan_ : base_;
    std::vector<float> v;
    for (auto& a : E.arrival) v.push_back(a[k]);
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    int hit = 0;
    for (float a : v) hit += a <= t_end_;
    const Fuel& f = fuel_table()[L_->fuel[k]];
    json j = {{"inside", true}, {"lat", lat}, {"lon", lon}, {"fuel", f.label}, {"burnable", f.burnable()},
              {"elevation_m", std::round(L_->elev[k])}, {"slope_pct", std::round(L_->slope_tan[k] * 100)},
              {"prob", n ? std::round(100.0 * hit / n) : 0}};
    auto ep = [&](float t) { return t < kInf ? json(t0_epoch_ + t * 60.0) : json(nullptr); };
    if (n) {
        j["eta_epoch"] = ep(v[(n - 1) / 2]);
        j["earliest_epoch"] = ep(v[size_t(std::floor(0.1 * (n - 1)))]);
        j["latest_epoch"] = ep(v[size_t(std::ceil(0.9 * (n - 1)))]);
        j["burning_now"] = v[(n - 1) / 2] <= t_now_;
    }
    if (danger_.size() == g.size()) {
        json dj = {{"level", danger_[k]}, {"class", kDangerName[danger_[k]]},
                   {"spread_kmh", std::round(double(hz_ros_[k]) * 10) / 10}, {"flame_m", std::round(double(hz_flame_[k]) * 10) / 10},
                   {"toward", compass_name(hz_dir_[k])}};
        if (rev_label_[k] >= 0 && rev_min_[k] < kInf) {
            dj["reaches"] = rev_names_[rev_label_[k]];
            dj["reaches_min"] = std::round(rev_min_[k]);
        }
        j["danger"] = dj;
    }
    if (E.best.arrival.size() == g.size() && E.best.arrival[k] < kInf) {
        j["flame_m"] = std::round(double(E.best.flame_m[k]) * 10) / 10;
        j["ros_kmh"] = std::round(double(E.best.ros[k]) * 0.6) / 10;
    }
    return j;
}

}  // namespace rapid
