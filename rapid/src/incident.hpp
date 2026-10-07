// incident.hpp — one active fire incident: data, forecast, calibration,
// tactics and the JSON the UI renders.  Thread-safe; heavy work is
// serialised while readers always get the latest finished snapshot.
#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "calibrate.hpp"
#include "ensemble.hpp"
#include "json.hpp"
#include "landscape.hpp"
#include "planner.hpp"
#include "spread.hpp"
#include "tactics.hpp"
#include "weather.hpp"

namespace rapid {

using json = nlohmann::json;

struct Destination {
    std::string id, name, kind;
    bool user = false;      // added by responders ("protect this")
    bool boundary = false;  // named road / river rather than a place
    LatLon pos;
    std::vector<LatLon> line;  // boundaries only
    std::vector<int> cells;
};

struct Observation {
    std::string kind;  // perimeter | hotspots | weather | fuel
    double epoch = 0;
    std::string source;
    std::vector<LatLon> pts;
    std::string fuel;  // fuel observations: the fuel name applied
};

struct IncidentOptions {
    std::string cache_dir = "cache";
    bool offline = false;
    int members = 24;
};

class Incident {
   public:
    explicit Incident(IncidentOptions opt);
    ~Incident();

    json start(const json& req);           // new fire from a pin / polygon
    json observe(const json& obs);         // on-site data: weather, perimeter, hotspots, fuel
    json add_destination(const json& req);
    json set_plan(const json& req);        // toggle recommendations, add/remove own interventions
    json refresh(const json& req);         // re-run for the current time / new horizon
    json set_resources(const json& req);   // {aircraft, trucks, dozers} the commander can give
    json set_settings(const json& req);    // {members, wide}: scenarios for the probability map
    json reset();

    json snapshot() const;
    bool active() const;
    uint64_t version() const { return version_.load(); }
    json point_info(double lat, double lon, bool plan) const;
    json status() const;

   private:
    void recompute_locked();
    bool fire_touches_edge_locked() const;
    void rebuild_domain_locked(double radius_m);
    void start_feature_fetch_locked();
    void build_destinations_locked();
    void attach_features(uint64_t generation, MapFeatures f, std::string source);
    Uncertainty uncertainty_locked(double now_epoch) const;
    json observe_locked(const json& obs);
    void set_status(const std::string& s);
    void publish(json j);

    IncidentOptions opt_;
    mutable std::mutex work_mu_;
    mutable std::mutex snap_mu_;
    json snapshot_;
    std::string status_ = "idle";
    std::atomic<uint64_t> version_{0};
    std::atomic<uint64_t> generation_{0};
    std::vector<std::thread> bg_;

    // ── incident state (guarded by work_mu_) ──
    bool active_ = false;
    std::unique_ptr<Landscape> L_;
    WeatherSeries wx_;
    std::unique_ptr<SpreadSolver> solver_;
    double t0_epoch_ = 0, horizon_h_ = 6;
    double radius_m_ = 0;
    LatLon ignition_;
    std::vector<std::pair<int, float>> fc_sources_;  // latest known fire state
    float fc_time_ = 0;
    SpreadParams params_;
    bool calibrated_ = false;
    json calibration_;
    std::vector<Observation> obs_;
    std::vector<Destination> dests_;
    int next_dest_id_ = 1;
    std::vector<Recommendation> recs_;
    std::vector<Intervention> manual_;
    int next_manual_id_ = 1;
    std::set<std::string> disabled_;  // recommendation keys switched off by the user
    std::set<std::string> forced_;    // recommendation keys switched on by the user
    Resources resources_;
    int members_ = 24;
    bool wide_ = false;
    EnsembleResult base_, plan_;
    // Danger map: fire potential × how quickly it would reach important places.
    std::vector<float> hz_ros_, hz_flame_, hz_dir_, rev_min_;
    std::vector<int> rev_label_;
    std::vector<std::string> rev_names_;
    std::vector<uint8_t> danger_;
    std::string danger_key_;  // inputs the cached danger map was computed for
    int danger_peak_hour_ = 0;
    bool plan_active_ = false;
    float t_now_ = 0, t_end_ = 0;
};

}  // namespace rapid
