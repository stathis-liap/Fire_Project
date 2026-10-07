#include "fuels.hpp"

#include <algorithm>
#include <climits>

namespace rapid {

const char* clc_to_fuel(int code) {
    switch (code) {
        case 111: case 112: return "Urban_Fabric";
        case 121: case 123: case 124: case 131: case 132: return "Non_Combustible";
        case 122: return "Urban_Road";
        case 133: return "Abandoned_Agricultural";
        case 141: case 142: return "Dry_Grass";
        case 211: case 212: case 241: return "Annual_Crops";
        case 213: return "Riparian_Vegetation";
        case 221: return "Vineyard";
        case 222: case 223: return "Olive_Grove";
        case 231: return "Dry_Grass";
        case 242: case 243: return "Abandoned_Agricultural";
        case 244: case 311: return "Oak_Forest";
        case 312: case 313: return "Aleppo_Pine";
        case 321: return "Dry_Grass";
        case 322: case 333: return "Phrygana_Low_Scrub";
        case 323: return "Maquis_Dense_Shrub";
        case 324: return "Garrigue";
        case 331: case 332: case 334: case 335: return "Non_Combustible";
        case 411: case 412: return "Riparian_Vegetation";
        case 421: case 422: case 423: case 511: case 512: case 521: case 522: case 523: return "Water";
        default: return nullptr;
    }
}

namespace {
struct ClcColor { int code, r, g, b; };
const ClcColor kClc[] = {
    {111, 230, 0, 77},    {112, 255, 0, 0},     {121, 204, 77, 242},  {122, 204, 0, 0},
    {123, 230, 204, 204}, {124, 230, 204, 230}, {131, 166, 0, 204},   {132, 166, 77, 0},
    {133, 255, 77, 255},  {141, 255, 166, 255}, {142, 255, 230, 255}, {211, 255, 255, 168},
    {212, 255, 255, 0},   {213, 230, 230, 0},   {221, 230, 128, 0},   {222, 242, 166, 77},
    {223, 230, 166, 0},   {231, 230, 230, 77},  {241, 255, 230, 166}, {242, 255, 230, 77},
    {243, 230, 204, 77},  {244, 242, 204, 166}, {311, 128, 255, 0},   {312, 0, 166, 0},
    {313, 77, 255, 0},    {321, 204, 242, 77},  {322, 166, 255, 128}, {323, 166, 230, 77},
    {324, 166, 242, 0},   {331, 230, 230, 230}, {332, 204, 204, 204}, {333, 204, 255, 204},
    {334, 0, 0, 0},       {335, 166, 230, 204}, {411, 166, 166, 255}, {412, 77, 77, 255},
    {421, 204, 204, 255}, {422, 230, 230, 255}, {423, 166, 166, 230}, {511, 0, 204, 242},
    {512, 128, 242, 230}, {521, 0, 255, 166},   {522, 166, 255, 230}, {523, 230, 242, 255},
};
}  // namespace

int clc_from_rgb(int r, int g, int b, int tol) {
    if (r == 255 && g == 255 && b == 255) return 0;  // map background = no data
    int best = 0, bd = INT_MAX;
    for (const auto& c : kClc) {
        int d = std::abs(c.r - r) + std::abs(c.g - g) + std::abs(c.b - b);
        if (d < bd) { bd = d; best = c.code; }
    }
    return bd <= tol ? best : 0;
}

RothermelFuel rothermel_prepare(const Fuel& f) {
    RothermelFuel rf;
    rf.waf = f.waf;
    rf.mx = f.mx;
    if (!f.burnable()) return rf;
    constexpr double particle_density = 32.0, mineral_total = 0.0555, mineral_eff = 0.01;
    const double s = f.sigma;
    rf.sigma = s;
    rf.rho_b = f.w0 / f.depth;
    rf.beta = rf.rho_b / particle_density;
    double beta_op = 3.348 * std::pow(s, -0.8189);
    rf.rel_beta = rf.beta / beta_op;
    double gamma_max = std::pow(s, 1.5) / (495.0 + 0.0594 * std::pow(s, 1.5));
    double A = 133.0 * std::pow(s, -0.7913);  // Albini (1976) correction of Rothermel's A
    rf.gamma = gamma_max * std::pow(rf.rel_beta, A) * std::exp(A * (1.0 - rf.rel_beta));
    double eta_s = std::min(1.0, 0.174 * std::pow(mineral_eff, -0.19));
    rf.wn_h_etas = f.w0 * (1.0 - mineral_total) * f.heat * eta_s;
    rf.xi = std::exp((0.792 + 0.681 * std::sqrt(s)) * (rf.beta + 0.1)) / (192.0 + 0.2595 * s);
    rf.eps = std::exp(-138.0 / s);
    rf.C = 7.47 * std::exp(-0.133 * std::pow(s, 0.55));
    rf.B = 0.02526 * std::pow(s, 0.54);
    rf.E = 0.715 * std::exp(-3.59e-4 * s);
    rf.slope_k = 5.275 * std::pow(rf.beta, -0.3);
    rf.t_res_min = 384.0 / s;
    rf.w0_kgm2 = f.w0 * 4.8824;
    rf.cbh = f.cbh;
    rf.cbd = f.cbd;
    if (f.cbh > 0) {
        constexpr double foliar_moisture = 100.0;  // % — summer conifer foliage
        rf.crown_i0 = std::pow(0.01 * f.cbh * (460.0 + 25.9 * foliar_moisture), 1.5);
    }
    rf.ok = true;
    return rf;
}

RothermelFuel rothermel_prepare_stand(const Fuel& f) {
    if (!f.understory) return rothermel_prepare(f);
    int u = fuel_index(f.understory);
    if (u < 0) return rothermel_prepare(f);
    RothermelFuel rf = rothermel_prepare(fuel_table()[u]);
    RothermelFuel own = rothermel_prepare(f);
    rf.waf = own.waf;
    rf.cbh = own.cbh;
    rf.cbd = own.cbd;
    rf.crown_i0 = own.crown_i0;
    return rf;
}

CrownResult crown_fire(const RothermelFuel& rf, double surface_int_kwm, double u10_kmh, double fine_moisture) {
    CrownResult c;
    if (rf.cbh <= 0 || rf.cbd <= 0 || surface_int_kwm < rf.crown_i0) return c;
    // +1.5 %: fine fuels under a canopy are shaded (Rothermel 1983 tables).
    double effm = std::clamp(fine_moisture * 100.0 + 1.5, 2.0, 30.0);
    double active = 11.02 * std::pow(std::max(0.0, u10_kmh), 0.90) * std::pow(rf.cbd, 0.19) * std::exp(-0.17 * effm);
    double critical = 3.0 / rf.cbd;  // m/min needed to sustain an active crown fire
    c.crowning = true;
    if (active >= critical) {
        c.ros_mmin = active;
        c.fraction = 1.0;
    } else {
        double cac = active / critical;  // Cruz et al. (2005) passive crown fire
        c.ros_mmin = active * std::exp(-cac);
        c.fraction = cac;
    }
    return c;
}

RothermelBase rothermel_base(const RothermelFuel& rf, double mf) {
    RothermelBase out;
    out.mf = mf;
    if (!rf.ok) return out;
    double rm = std::clamp(mf / rf.mx, 0.0, 1.0);
    double eta_m = rm >= 1.0 ? 0.0 : std::max(0.0, 1.0 - 2.59 * rm + 5.11 * rm * rm - 3.52 * rm * rm * rm);
    out.ir = rf.gamma * rf.wn_h_etas * eta_m;
    double q_ig = 250.0 + 1116.0 * mf;
    out.r0_ftmin = out.ir * rf.xi / (rf.rho_b * rf.eps * q_ig);
    return out;
}

}  // namespace rapid
