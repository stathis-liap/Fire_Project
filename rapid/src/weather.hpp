// weather.hpp — hourly weather series with optional on-site overrides.
#pragma once

#include <cmath>
#include <string>
#include <vector>

#include "fuels.hpp"
#include "geo.hpp"

namespace rapid {

struct WxSample {
    double epoch = 0;      // unix seconds (UTC)
    double wind_ms = 0;    // 10 m wind speed
    double wind_from = 0;  // meteorological direction (deg, wind blows FROM)
    double temp_c = 25;
    double rh = 40;
};

// A measurement taken on site (weather station, handheld anemometer, crew report).
struct LocalWx {
    WxSample s;
    std::string source;  // free text, e.g. "Engine 12 kestrel"
};

class WeatherSeries {
   public:
    std::vector<WxSample> hourly;  // sorted by epoch
    std::string source = "none";   // "Open-Meteo forecast", "default", ...
    std::vector<LocalWx> local;    // on-site measurements (latest wins)

    bool empty() const { return hourly.empty(); }

    // Weather at time `epoch`.  Forecast is interpolated through u/v wind
    // components; an on-site measurement fully overrides within ±1 h of when
    // it was taken and then fades linearly back to the forecast over 3 h.
    WxSample at(double epoch) const {
        WxSample f = interp(epoch);
        if (local.empty()) return f;
        const LocalWx* best = &local.front();
        for (auto& l : local)
            if (l.s.epoch > best->s.epoch) best = &l;
        double dt_h = std::abs(epoch - best->s.epoch) / 3600.0;
        double w = dt_h <= 1.0 ? 1.0 : std::max(0.0, 1.0 - (dt_h - 1.0) / 3.0);
        if (w <= 0) return f;
        return blend(f, best->s, w);
    }

    static WxSample blend(const WxSample& a, const WxSample& b, double w) {
        WxSample o;
        o.epoch = a.epoch;
        double ua = a.wind_ms * std::sin(a.wind_from * kDeg), va = a.wind_ms * std::cos(a.wind_from * kDeg);
        double ub = b.wind_ms * std::sin(b.wind_from * kDeg), vb = b.wind_ms * std::cos(b.wind_from * kDeg);
        double u = ua + w * (ub - ua), v = va + w * (vb - va);
        // Keep speed as the blended scalar speed: averaging vectors through a
        // direction change would otherwise create an artificial lull.
        o.wind_ms = a.wind_ms + w * (b.wind_ms - a.wind_ms);
        o.wind_from = std::fmod(std::atan2(u, v) / kDeg + 360.0, 360.0);
        o.temp_c = a.temp_c + w * (b.temp_c - a.temp_c);
        o.rh = a.rh + w * (b.rh - a.rh);
        return o;
    }

   private:
    WxSample interp(double epoch) const {
        if (hourly.empty()) {
            WxSample d;
            d.epoch = epoch;
            return d;
        }
        if (epoch <= hourly.front().epoch) return hourly.front();
        if (epoch >= hourly.back().epoch) return hourly.back();
        size_t i = 1;
        while (hourly[i].epoch < epoch) ++i;
        const auto &a = hourly[i - 1], &b = hourly[i];
        double w = (epoch - a.epoch) / std::max(1.0, b.epoch - a.epoch);
        WxSample o = blend(a, b, w);
        o.epoch = epoch;
        return o;
    }
};

}  // namespace rapid
