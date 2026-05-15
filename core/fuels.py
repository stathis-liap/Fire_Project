# fuels.py
# Fuel parameters calibrated to standard Rothermel fuel models and
# Mediterranean field measurements.  All units follow Rothermel (1972):
#   heat_content  — BTU/lb
#   surface_ratio — ft²/ft³  (σ)
#   fuel_load     — lb/ft²   (w₀, oven-dry)
#   fuel_depth    — ft        (δ)
#   moisture_ext  — fraction  (Mx)
#   bulk_density  — lb/ft³   (ρ_b = w₀ / δ)
#
# Reference values cross-checked against:
#   • Scott & Burgan (2005) Standard Fire Behavior Fuel Models (RMRS-GTR-153)
#   • Mitsopoulos & Dimitrakopoulos (2007) Canopy fuel characteristics of
#     Mediterranean pine forests (IJWF 16, 351-361)
#   • Fernandes et al. (2000) Shrubland fire behaviour in Portugal

GREEK_FUELS = {
    "Aleppo_Pine": {
        # Closest to FM TL5 (hardwood litter + understory)
        # Pine needle litter layer, ~2 kg/m² oven-dry ≈ 0.041 lb/ft²
        "heat_content":  8000,   # BTU/lb  (resinous needle litter)
        "surface_ratio": 1500,   # ft²/ft³ (pine needles, ~3 mm dia)
        "fuel_load":     0.041,  # lb/ft²  (≈ 2 kg/m²  needle+litter layer)
        "fuel_depth":    0.20,   # ft      (6 cm compacted litter bed)
        "moisture_ext":  0.25,   # fraction (resin raises extinction moisture)
        "bulk_density":  0.20,   # lb/ft³  (w₀/δ; loose packed needle bed)
    },

    "Phrygana_Low_Scrub": {
        # FM GS2 (moderate load dry climate grass-shrub) — very fast burning
        "heat_content":  8000,
        "surface_ratio": 2500,   # ft²/ft³ (small twigs + tiny leaves)
        "fuel_load":     0.023,  # lb/ft²  (≈ 1.1 kg/m²)
        "fuel_depth":    0.50,   # ft      (15 cm low shrub canopy)
        "moisture_ext":  0.15,   # fraction (dries out fastest of all types)
        "bulk_density":  0.046,  # lb/ft³
    },

    "Maquis_Dense_Shrub": {
        # FM SH7 (very high load dry shrub) — intense heat, slower spread
        "heat_content":  8500,   # BTU/lb  (high oil content in Cistus/Pistacia)
        "surface_ratio": 1800,   # ft²/ft³
        "fuel_load":     0.092,  # lb/ft²  (≈ 4.5 kg/m²  dense thicket)
        "fuel_depth":    1.30,   # ft      (≈ 40 cm tall shrub bed)
        "moisture_ext":  0.25,
        "bulk_density":  0.071,  # lb/ft³
    },

    "Dry_Grass": {
        # FM GR2 (Low Load, Dry Climate Grass) — Scott & Burgan (2005)
        # w0 and rho_b matched to standard model; sigma lowered to keep
        # rel_beta inside Rothermel's calibrated range.
        "heat_content":  8000,
        "surface_ratio": 2000,   # ft²/ft³ (standard GR2 SAV; sigma=3000 pushes rel_beta out of range)
        "fuel_load":     0.046,  # lb/ft²  (standard GR2 1-hr load ≈ 2.25 kg/m²)
        "fuel_depth":    1.00,   # ft      (30 cm standing dead grass)
        "moisture_ext":  0.15,   # fraction (GR2 standard extinction moisture)
        "bulk_density":  0.046,  # lb/ft³  (= w0/delta; keeps beta in valid range)
    },

    "Oak_Forest": {
        # FM TL3 (moderate load conifer litter) — leaf litter only burns
        "heat_content":  8000,
        "surface_ratio": 1200,   # ft²/ft³ (large broadleaf, lower ratio)
        "fuel_load":     0.030,  # lb/ft²  (≈ 1.5 kg/m²  leaf litter layer)
        "fuel_depth":    0.20,   # ft      (6 cm compacted leaf litter)
        "moisture_ext":  0.30,   # fraction (retains moisture well)
        "bulk_density":  0.15,   # lb/ft³
    },

    "Olive_Grove": {
        # Between TL1 and GS1 — managed orchard, sparse ground fuel
        "heat_content":  8000,
        "surface_ratio": 1400,   # ft²/ft³
        "fuel_load":     0.020,  # lb/ft²  (≈ 1.0 kg/m²  pruning debris + grass)
        "fuel_depth":    0.30,   # ft      (9 cm ground fuel layer)
        "moisture_ext":  0.20,
        "bulk_density":  0.067,  # lb/ft³
    },
}
