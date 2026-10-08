// windfield.hpp — diagnostic terrain wind solver.
//
// The forecast gives one wind per hour for the whole area.  On the ground the
// wind is steered around hills, squeezed and sped up over ridges, sheltered
// in their lee, and joined by slope winds that blow uphill on sunny slopes by
// day and downhill at night.  This module turns the single forecast wind into
// a wind for every cell:
//
//  1. Mass-consistent flow (Sherman 1978; the 2-D layer form used by
//     diagnostic models such as WindNinja's conservation-of-mass solver).
//     Air in a layer of depth h(x, y) = H − z(x, y) has to satisfy
//     ∇·(h u) = 0.  The minimum change to the forecast wind u₀ that does so
//     is u = u₀ + ∇λ with ∇·(h∇λ) = −∇·(h u₀) (Dirichlet λ = 0 on the open
//     boundary), solved with preconditioned conjugate gradients on a grid of
//     ≤ 160×160 cells.  Because the problem is linear in u₀, two solves (for
//     a unit east and a unit north wind) give the field for *every* hour and
//     every ensemble member: u = U·e_x(x) + V·e_y(x).
//  2. Lee sheltering from the upwind horizon angle Sx (Winstral et al. 2002),
//     precomputed for 24 wind directions.  Flow separates behind ridges, which
//     a potential-flow solution cannot represent on its own.
//  3. Thermal slope winds (anabatic / katabatic), computed per hour from sun
//     position, slope and aspect, and damped when the ambient wind is strong.
#pragma once

#include <cstdint>
#include <vector>

#include "geo.hpp"

namespace rapid {

struct WindField {
    // Response of the local wind (east, north components) to a unit ambient
    // wind blowing towards the east (x) and towards the north (y).
    std::vector<float> exu, exv, eyu, eyv;
    // Lee shelter factor (0..255 → 0..1) for kDirs wind-FROM directions,
    // laid out [dir * N + cell]; direction i is i * (360 / kDirs) degrees.
    static constexpr int kDirs = 24;
    std::vector<uint8_t> shelter;
    int coarse_factor = 1;
    int cg_iterations = 0;
    double runtime_ms = 0;

    bool active() const { return !exu.empty(); }

    // Local 10 m wind (m/s; u east, v north) for an ambient wind (U, V)
    // (components it blows towards) with its lee-shelter bins from
    // shelter_bins().  The terrain-modified speed is kept within
    // [0.25, 2] × ambient so DEM artefacts cannot create gales or dead calm.
    void local(int cell, double U, double V, int bin0, int bin1, float wbin, double& u, double& v) const {
        u = U * exu[cell] + V * eyu[cell];
        v = U * exv[cell] + V * eyv[cell];
        const size_t N = exu.size();
        double sh = ((1.f - wbin) * shelter[size_t(bin0) * N + cell] + wbin * shelter[size_t(bin1) * N + cell]) / 255.0;
        double amb = std::hypot(U, V), loc = std::hypot(u, v);
        double k = sh;
        if (amb > 1e-6 && loc > 1e-9) {
            double r = loc / amb;
            double rc = std::clamp(r, 0.25, 2.0);
            k *= rc / r;
        }
        u *= k;
        v *= k;
    }
};

// Builds the field for a grid from its elevation (m).  Flat ground gives the
// identity (local wind = ambient wind everywhere).
void build_wind_field(const Grid& g, const std::vector<float>& elev, WindField& W);

// Interpolation weights for a wind-FROM direction between the precomputed bins.
inline void shelter_bins(double from_deg, int& b0, int& b1, float& w) {
    double x = std::fmod(from_deg + 720.0, 360.0) / (360.0 / WindField::kDirs);
    b0 = int(std::floor(x)) % WindField::kDirs;
    b1 = (b0 + 1) % WindField::kDirs;
    w = float(x - std::floor(x));
}

// Thermal slope wind for one hour: signed strength (m/s; > 0 upslope by day,
// < 0 downslope at night) before slope/aspect weighting, and the sun azimuth.
struct ThermalHour {
    double strength = 0;
    double sun_sin = 0, sun_cos = -1;  // sin/cos of the sun azimuth
};
ThermalHour thermal_hour(double epoch, double lon, double ambient_ms);

// Thermal wind speed (m/s, signed along the upslope direction) at a cell.
inline double thermal_speed(const ThermalHour& th, double slope_tan, double up_sin, double up_cos) {
    if (th.strength == 0 || slope_tan < 0.03) return 0;
    double slope_fac = std::min(1.0, slope_tan / 0.35);
    if (th.strength < 0) return th.strength * slope_fac;  // katabatic: aspect does not matter
    // Slopes facing the sun heat most (facing = upslope + 180°).
    double face_sun = -(up_sin * th.sun_sin + up_cos * th.sun_cos);
    return th.strength * slope_fac * (0.55 + 0.45 * face_sun);
}

}  // namespace rapid
