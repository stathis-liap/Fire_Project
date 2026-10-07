// fuels.hpp — Greek / Mediterranean fuel models and Rothermel (1972) surface
// spread.  Fuel parameters are ported 1:1 from core/fuels.py so the rapid
// estimator and the full WILSON engine agree on fuel behaviour.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <vector>

namespace rapid {

struct Fuel {
    const char* name;   // matches core/fuels.py keys
    const char* label;  // plain-language name shown to responders
    double heat;        // BTU/lb
    double sigma;       // ft²/ft³
    double w0;          // lb/ft²
    double depth;       // ft
    double mx;          // moisture of extinction (fraction)
    double waf;         // wind adjustment factor 10 m → midflame (Andrews 2012)
    std::array<unsigned char, 3> rgb;
    double cbh = 0;     // canopy base height (m); 0 → no canopy, surface fire only
    double cbd = 0;     // canopy bulk density (kg/m³)
    // Surface fuel under the canopy.  Greek pine and oak stands usually carry
    // a shrub understory that drives surface spread and crown initiation;
    // the litter values above are used where no understory is given.
    const char* understory = nullptr;

    bool burnable() const { return w0 > 0 && depth > 0 && sigma > 0; }
};

// Order matters: indices are stored in the landscape fuel grid.
// Canopy (cbh, cbd) for forests: Mitsopoulos & Dimitrakopoulos (2007) for
// Aleppo pine; Scott & Reinhardt (2001) typical values for the others.
inline const std::vector<Fuel>& fuel_table() {
    static const std::vector<Fuel> t = {
        {"Aleppo_Pine", "Pine forest", 8000, 1500, 0.041, 0.20, 0.25, 0.20, {34, 120, 34}, 2.5, 0.15, "Maquis_Dense_Shrub"},
        {"Black_Pine", "Mountain pine forest", 8000, 1400, 0.051, 0.25, 0.25, 0.15, {20, 100, 20}, 4.0, 0.12, "Garrigue"},
        {"Maritime_Pine", "Maritime pine forest", 8500, 1600, 0.061, 0.25, 0.22, 0.20, {25, 110, 30}, 3.0, 0.15, "Maquis_Dense_Shrub"},
        {"Stone_Pine", "Stone pine", 8000, 1200, 0.035, 0.18, 0.25, 0.20, {50, 140, 50}, 5.0, 0.08, "Dry_Grass"},
        {"Cypress", "Cypress", 9000, 2800, 0.051, 0.50, 0.20, 0.20, {10, 80, 30}, 1.5, 0.20},
        {"Greek_Fir", "Fir forest", 7800, 1500, 0.041, 0.25, 0.30, 0.12, {30, 130, 60}, 3.0, 0.15},
        {"Oak_Forest", "Oak / broadleaf forest", 8000, 1200, 0.030, 0.20, 0.30, 0.15, {80, 160, 40}, 5.0, 0.06, "Garrigue"},
        {"Chestnut_Forest", "Chestnut forest", 7800, 1000, 0.035, 0.25, 0.30, 0.15, {90, 150, 30}, 6.0, 0.05},
        {"Beech_Forest", "Beech forest", 7800, 1000, 0.030, 0.25, 0.35, 0.12, {100, 170, 50}, 8.0, 0.04},
        {"Maquis_Dense_Shrub", "Dense shrubland", 8500, 1800, 0.092, 1.30, 0.25, 0.40, {120, 155, 45}},
        {"Tall_Maquis", "Tall shrubland", 8500, 1600, 0.115, 1.80, 0.25, 0.40, {110, 145, 40}},
        {"Phrygana_Low_Scrub", "Low scrub", 8000, 2500, 0.023, 0.50, 0.15, 0.40, {180, 200, 100}},
        {"Garrigue", "Scrub / garrigue", 8500, 2000, 0.046, 0.80, 0.18, 0.40, {170, 185, 75}},
        {"Dry_Grass", "Dry grass", 8000, 2000, 0.046, 1.00, 0.15, 0.40, {210, 200, 75}},
        {"Annual_Crops", "Fields / stubble", 7500, 3500, 0.034, 1.00, 0.12, 0.40, {225, 215, 95}},
        {"Abandoned_Agricultural", "Overgrown farmland", 7800, 2200, 0.028, 0.80, 0.15, 0.40, {185, 185, 105}},
        {"Eucalyptus", "Eucalyptus", 9500, 1800, 0.092, 0.30, 0.20, 0.20, {50, 160, 70}, 4.0, 0.10, "Garrigue"},
        {"Olive_Grove", "Olive grove", 8000, 1400, 0.020, 0.30, 0.20, 0.30, {150, 168, 58}},
        {"Vineyard", "Vineyard", 7800, 1200, 0.010, 0.20, 0.20, 0.40, {195, 158, 38}},
        {"Riparian_Vegetation", "River vegetation", 7800, 1600, 0.025, 0.30, 0.35, 0.20, {50, 170, 100}},
        {"Water", "Water", 0, 0, 0, 0, 1.0, 0, {30, 80, 200}},
        {"Urban_Fabric", "Built-up area", 8000, 800, 0.010, 0.50, 0.40, 0.30, {158, 148, 148}},
        {"Urban_Road", "Road", 7500, 2000, 0.003, 0.10, 0.12, 0.40, {100, 92, 90}},
        {"Non_Combustible", "Bare / rock", 0, 0, 0, 0, 1.0, 0, {120, 115, 108}},
    };
    return t;
}

inline int fuel_index(const std::string& name) {
    const auto& t = fuel_table();
    for (size_t i = 0; i < t.size(); ++i)
        if (name == t[i].name) return int(i);
    return -1;
}

// CORINE Land Cover 2018 class → fuel name (same mapping as core/landscape.py).
const char* clc_to_fuel(int clc_code);
// Official CLC 2018 legend colours (verified against the EEA MapServer legend).
// Returns the CLC code for an exact or near (≤ tol) colour match, else 0.
int clc_from_rgb(int r, int g, int b, int tol = 24);

// Equilibrium moisture content of fine dead fuel (Simard 1968), fraction.
inline double emc_fraction(double temp_c, double rh) {
    double tf = temp_c * 9.0 / 5.0 + 32.0, e;
    if (rh < 10) e = 0.03229 + 0.281073 * rh - 0.000578 * rh * tf;
    else if (rh < 50) e = 2.22749 + 0.160107 * rh - 0.01478 * tf;
    else e = 21.0606 + 0.005565 * rh * rh - 0.00035 * rh * tf - 0.483199 * rh;
    return std::clamp(e / 100.0, 0.02, 0.35);
}

// Moisture-independent Rothermel terms for one fuel, precomputed once.
struct RothermelFuel {
    bool ok = false;
    double beta, rel_beta, gamma, xi, eps, rho_b, wn_h_etas, sigma, mx;
    double C, B, E;   // wind-factor coefficients
    double slope_k;   // 5.275 β^-0.3
    double t_res_min; // flame residence time 384/σ
    double waf;
    double w0_kgm2 = 0;  // surface fuel load (kg/m²)
    double cbh = 0, cbd = 0;
    double crown_i0 = 0;  // critical surface intensity for crowning (kW/m), Van Wagner (1977)
};

// Crown fire: Van Wagner (1977) initiation, Cruz et al. (2005) active spread
// and Van Wagner's 3/CBD criterion for active vs passive crowning.
struct CrownResult {
    bool crowning = false;
    double ros_mmin = 0;  // head-fire crown spread rate (m/min)
    double fraction = 0;  // 1 = active crown fire
};
CrownResult crown_fire(const RothermelFuel& rf, double surface_int_kwm, double u10_kmh, double fine_moisture);

RothermelFuel rothermel_prepare(const Fuel& f);
// As above, but for a forest with an understory: surface terms from the
// understory fuel, wind sheltering and canopy from the forest itself.
RothermelFuel rothermel_prepare_stand(const Fuel& f);

// No-wind, no-slope spread and reaction intensity for fuel moisture mf.
struct RothermelBase {
    double r0_ftmin = 0;  // ft/min
    double ir = 0;        // reaction intensity BTU/ft²/min
    double mf = 0;        // fine dead fuel moisture used (fraction)
};
RothermelBase rothermel_base(const RothermelFuel& rf, double mf);

constexpr double kFtPerM = 3.28084;
constexpr double kMsToFtMin = 196.8504;
constexpr double kFtMinToMMin = 0.3048;

}  // namespace rapid
