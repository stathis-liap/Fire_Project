# fuels.py
# Fuel parameters calibrated to standard Rothermel fuel models and
# Mediterranean / Greek field measurements.  All units follow Rothermel (1972):
#   heat_content  — BTU/lb
#   surface_ratio — ft²/ft³  (σ, SAV ratio)
#   fuel_load     — lb/ft²   (w₀, oven-dry 1-hr fine fuel)
#   fuel_depth    — ft        (δ, fuel-bed depth)
#   moisture_ext  — fraction  (Mx, moisture of extinction)
#   bulk_density  — lb/ft³   (ρ_b = w₀ / δ)
#
# Unit conversions used:
#   1 kg/m²  = 0.2048 lb/ft²
#   1 m      = 3.281 ft
#   1 kg/m³  = 0.0624 lb/ft³
#   bulk_density is always recalculated as fuel_load / fuel_depth
#
# ─── Primary references ────────────────────────────────────────────────────────
#  [SB05]  Scott, J.H. & Burgan, R.E. (2005). Standard fire behavior fuel models:
#          a comprehensive set for use with Rothermel's surface fire spread model.
#          USDA Forest Service RMRS-GTR-153.
#
#  [MD07]  Mitsopoulos, I.D. & Dimitrakopoulos, A.P. (2007). Canopy fuel
#          characteristics and potential crown fire behavior in Aleppo pine
#          (Pinus halepensis Mill.) forests. Annals of Forest Science 64(3):287–299.
#
#  [DIM02] Dimitrakopoulos, A.P. (2002). Mediterranean fuel models and potential
#          fire behaviour in Greece. International Journal of Wildland Fire 11(2):127–130.
#
#  [FER00] Fernandes, P.M., Catchpole, W.R. & Rego, F.C. (2000). Shrubland fire
#          behaviour in Portugal. International Journal of Wildland Fire 9(1):1–15.
#
#  [FER02] Fernandes, P.M. (2002). Fire spread and behaviour under maritime pine.
#          Forest Ecology and Management 158(1–3):145–157.
#
#  [GAN13] Ganteaume, A. et al. (2013). Effects of vegetation type, head fire and
#          slope on flammability of Cupressus sempervirens.
#          Annals of Forest Science 70(1):21–30.
#
#  [CHE98] Cheney, N.P., Gould, J.S. & Catchpole, W.R. (1998). Prediction of
#          fire spread in grasslands. International Journal of Wildland Fire 8(1):1–13.
#
#  [PAU07] Pausas, J.G. et al. (2007). Wildfires and the role of their drivers are
#          changing in the Iberian Peninsula. Global Ecology and Biogeography 17:490–503.
#
#  [XAN12] Xanthopoulos, G., Roussos, A. & Jimenez, E. (2012). Forest fire danger
#          in Greece: a preliminary study. In: Proc. 7th Int. Conference on Forest
#          Fire Research, Coimbra, Portugal.
#
#  [ARI10] Arianoutsou, M. et al. (2010). Post-fire regeneration of vegetation in
#          Mediterranean-type ecosystems of the eastern Mediterranean basin.
#          Plant Ecology 212(1):1–12.
#
#  [ROT72] Rothermel, R.C. (1972). A mathematical model for predicting fire spread
#          in wildland fuels. USDA Forest Service Research Paper INT-115.
#
#  [MEL10] Mell, W. et al. (2010). The wildland–urban interface fire problem –
#          current approaches and research needs.
#          International Journal of Wildland Fire 19(2):238–251.
#
#  [COH08] Cohen, J.D. & Stratton, R.D. (2008). Home Destruction Examination:
#          Grass Valley Fire. USDA Forest Service Technical Note RMRS-TN-28WWW.
#
#  [NF1144] NFPA 1144 (2018). Standard for Reducing Structure Ignition Hazards
#           from Wildfire. National Fire Protection Association, Quincy MA.
#
#  [BUT04] Butler, B.W. et al. (2004). Fire behavior associated with the 1994
#          South Canyon Fire on Storm King Mountain, Colorado.
#          Forest Science 50(3):408–428.
# ──────────────────────────────────────────────────────────────────────────────

GREEK_FUELS = {

    # ── PINE FORESTS ──────────────────────────────────────────────────────────

    "Aleppo_Pine": {
        # Pinus halepensis — dominant low-altitude pine of mainland Greece,
        # Attica, Peloponnese, Aegean islands.
        # Closest to FM TL5 (hardwood litter + sparse understory).
        # [MD07]: needle-bed load 1.5–3 kg/m², σ ~ 3500 m⁻¹ (≈ 1067 ft²/ft³)
        # [DIM02]: Mx 0.20–0.30; ROS up to 1.5 m/s in wind-driven events.
        "heat_content":  8000,   # BTU/lb  (resinous needle litter)
        "surface_ratio": 1500,   # ft²/ft³ (pine needles, ~3 mm diameter)
        "fuel_load":     0.041,  # lb/ft²  (≈ 2.0 kg/m² oven-dry needle litter)
        "fuel_depth":    0.20,   # ft      (6 cm compacted litter bed)
        "moisture_ext":  0.25,   # fraction
        "bulk_density":  0.205,  # lb/ft³  (= w₀/δ)
    },

    "Black_Pine": {
        # Pinus nigra — mountain pine, central/northern Greece (Pindos, Rhodope,
        # Olympus, Taygetos).  Grows at 600–1800 m.
        # Slightly moister environment than Aleppo; needles longer → lower σ.
        # [MD07], [SB05] FM TL5 adjusted for higher-elevation conditions.
        "heat_content":  8000,
        "surface_ratio": 1400,   # ft²/ft³ (longer needles, 5–8 cm)
        "fuel_load":     0.051,  # lb/ft²  (≈ 2.5 kg/m²; denser stand litter)
        "fuel_depth":    0.25,   # ft      (7.5 cm; slightly deeper bed)
        "moisture_ext":  0.25,
        "bulk_density":  0.204,  # lb/ft³
    },

    "Maritime_Pine": {
        # Pinus pinaster — western Greece (Kefalonia, Zakynthos, Lefkada),
        # some Ionian coast stands.  Very resinous, high fire danger.
        # [FER02]: fine-fuel load 2–5 kg/m²; flame lengths 3–8 m in extreme runs.
        "heat_content":  8500,   # BTU/lb  (high resin raises heat content)
        "surface_ratio": 1600,   # ft²/ft³
        "fuel_load":     0.061,  # lb/ft²  (≈ 3.0 kg/m²; dense resinous bed)
        "fuel_depth":    0.25,   # ft      (7.5 cm)
        "moisture_ext":  0.22,
        "bulk_density":  0.244,  # lb/ft³
    },

    "Stone_Pine": {
        # Pinus pinea — coastal dunes, parks, urban edges throughout Greece.
        # Umbrella form, large needles, less dense litter than Aleppo.
        # [SB05] FM TL3; lower bulk density due to coarser litter.
        "heat_content":  8000,
        "surface_ratio": 1200,   # ft²/ft³ (coarse needles, 8–15 cm)
        "fuel_load":     0.035,  # lb/ft²  (≈ 1.7 kg/m²)
        "fuel_depth":    0.18,   # ft      (5.5 cm)
        "moisture_ext":  0.25,
        "bulk_density":  0.194,  # lb/ft³
    },

    "Cypress": {
        # Cupressus sempervirens — extensively planted in Greek cemeteries,
        # rural estates, roadsides, wind-breaks; also natural stands in Crete.
        # EXTREMELY flammable: fine scale-like foliage + volatile essential oils.
        # [GAN13]: ignition probability > 90 % at RH < 25 %; among the most
        #          flammable Mediterranean species tested.
        "heat_content":  9000,   # BTU/lb  (essential-oil content ~12 000 BTU/kg)
        "surface_ratio": 2800,   # ft²/ft³ (very fine overlapping scales, ~0.5 mm)
        "fuel_load":     0.051,  # lb/ft²  (≈ 2.5 kg/m²; fine-twig + scale litter)
        "fuel_depth":    0.50,   # ft      (15 cm litter + retained crown fuel)
        "moisture_ext":  0.20,   # fraction (low Mx — ignites at high moisture)
        "bulk_density":  0.102,  # lb/ft³
    },

    "Greek_Fir": {
        # Abies cephalonica — endemic to Greek mountains (Parnassos, Giona,
        # Taygetos, Chelmos); forms dense high-altitude forests at 900–2000 m.
        # Moist habitat reduces fire risk; needle litter similar to silver fir.
        # [SB05] FM TL4 (small duff-laden conifer litter).
        "heat_content":  7800,   # BTU/lb  (slightly lower heat — moist environment)
        "surface_ratio": 1500,   # ft²/ft³
        "fuel_load":     0.041,  # lb/ft²  (≈ 2.0 kg/m²)
        "fuel_depth":    0.25,   # ft      (7.5 cm; deeper moist litter)
        "moisture_ext":  0.30,   # fraction (retains more moisture at altitude)
        "bulk_density":  0.164,  # lb/ft³
    },

    # ── BROADLEAF FORESTS ─────────────────────────────────────────────────────

    "Oak_Forest": {
        # Quercus pubescens (downy oak), Q. ilex (holm oak), Q. frainetto
        # (Hungarian oak) — dominant broadleaf of lowland–montane Greece.
        # [SB05] FM TL3; leaf litter has lower σ and retains more moisture.
        "heat_content":  8000,
        "surface_ratio": 1200,   # ft²/ft³ (large broadleaf, lower SAV)
        "fuel_load":     0.030,  # lb/ft²  (≈ 1.5 kg/m² leaf litter layer)
        "fuel_depth":    0.20,   # ft      (6 cm compacted leaf litter)
        "moisture_ext":  0.30,
        "bulk_density":  0.150,  # lb/ft³
    },

    "Chestnut_Forest": {
        # Castanea sativa — northern and central Greece (Pelion, Samothrace,
        # Kavala, Serres, parts of Macedonia).  Heavy leaf and spiny cupule
        # litter; moist humid slopes reduce fire frequency.
        # [SB05] FM TL5; broadleaf with heavy litter and some understory fuel.
        "heat_content":  7800,
        "surface_ratio": 1000,   # ft²/ft³ (large leaves + spiny cupules)
        "fuel_load":     0.035,  # lb/ft²  (≈ 1.7 kg/m²; thick litter carpet)
        "fuel_depth":    0.25,   # ft      (7.5 cm)
        "moisture_ext":  0.30,
        "bulk_density":  0.140,  # lb/ft³
    },

    "Beech_Forest": {
        # Fagus sylvatica subsp. moesiaca — Rhodope, Vikos gorge, Pindos,
        # Olympus; grows at 700–2000 m in humid continental climate.
        # Low fire risk in most years; vulnerable in extreme summer drought.
        # [SB05] FM TL6 (moderate load humid climate litter).
        "heat_content":  7800,
        "surface_ratio": 1000,   # ft²/ft³
        "fuel_load":     0.030,  # lb/ft²  (≈ 1.5 kg/m²)
        "fuel_depth":    0.25,   # ft
        "moisture_ext":  0.35,   # fraction (highest Mx — stays moist longest)
        "bulk_density":  0.120,  # lb/ft³
    },

    # ── SHRUBLANDS ────────────────────────────────────────────────────────────

    "Maquis_Dense_Shrub": {
        # Dense maquis / macchia: Pistacia lentiscus, Arbutus unedo,
        # Phillyrea latifolia, Ceratonia siliqua — 1–2 m height.
        # One of the most intense fire-producing vegetation types in Greece.
        # [DIM02]: ROS 0.3–1.0 m/s; flame lengths 3–7 m.
        # [SB05] FM SH7 (very high load dry shrub).
        "heat_content":  8500,   # BTU/lb  (high oil content in Pistacia/Arbutus)
        "surface_ratio": 1800,   # ft²/ft³
        "fuel_load":     0.092,  # lb/ft²  (≈ 4.5 kg/m² dense thicket)
        "fuel_depth":    1.30,   # ft      (≈ 40 cm)
        "moisture_ext":  0.25,
        "bulk_density":  0.071,  # lb/ft³
    },

    "Tall_Maquis": {
        # Tall macchia / mixed Arbutus–Pistacia–Phillyrea shrubland > 1.5 m;
        # transitions to open forest on abandoned land.  More fuel load and
        # taller than standard maquis; produces extreme fire behaviour.
        # [DIM02], [ARI10]; [SB05] FM SH5 (high load dry climate shrub).
        "heat_content":  8500,
        "surface_ratio": 1600,   # ft²/ft³
        "fuel_load":     0.115,  # lb/ft²  (≈ 5.6 kg/m²)
        "fuel_depth":    1.80,   # ft      (55 cm tall shrub bed)
        "moisture_ext":  0.25,
        "bulk_density":  0.064,  # lb/ft³
    },

    "Phrygana_Low_Scrub": {
        # Phrygana / batha: Sarcopoterium spinosum, Thymus capitatus,
        # Coridothymus capitatus, Cistus spp. — 10–40 cm height.
        # Very fast spreading, extremely aromatic, low-moisture extinction.
        # Covers large areas of Aegean islands, Crete, eastern mainland.
        # [DIM02]: fastest ROS of all Greek fuel types (~1.5–2.5 m/s).
        # [SB05] FM GS2 (moderate load dry climate grass-shrub).
        "heat_content":  8000,
        "surface_ratio": 2500,   # ft²/ft³ (tiny twigs + fine leaves)
        "fuel_load":     0.023,  # lb/ft²  (≈ 1.1 kg/m²)
        "fuel_depth":    0.50,   # ft      (15 cm)
        "moisture_ext":  0.15,   # fraction (lowest Mx — dries out fastest)
        "bulk_density":  0.046,  # lb/ft³
    },

    "Garrigue": {
        # Garrigue / degraded phrygana: Cistus monspeliensis, Lavandula stoechas,
        # Rosmarinus officinalis, Calicotome villosa — 30–80 cm.
        # More aromatic oils than standard phrygana; intermediate between
        # phrygana and maquis; covers calcareous slopes throughout Greece.
        # [FER00], [PAU07]; [SB05] FM GS2/SH2 transition.
        "heat_content":  8500,   # BTU/lb  (lavender/rosemary essential oils)
        "surface_ratio": 2000,   # ft²/ft³
        "fuel_load":     0.046,  # lb/ft²  (≈ 2.25 kg/m²)
        "fuel_depth":    0.80,   # ft      (25 cm)
        "moisture_ext":  0.18,
        "bulk_density":  0.058,  # lb/ft³
    },

    # ── GRASSES & AGRICULTURAL ────────────────────────────────────────────────

    "Dry_Grass": {
        # Natural dry grassland and pasture: Brachypodium retusum,
        # Poa bulbosa, Stipa spp. — widespread on dry slopes and plateaus.
        # [SB05] FM GR2 (low load dry climate grass).
        # [CHE98]: grass fires fastest-spreading of all wildfire types.
        "heat_content":  8000,
        "surface_ratio": 2000,   # ft²/ft³
        "fuel_load":     0.046,  # lb/ft²  (≈ 2.25 kg/m² standing dead grass)
        "fuel_depth":    1.00,   # ft      (30 cm standing dead grass)
        "moisture_ext":  0.15,
        "bulk_density":  0.046,  # lb/ft³
    },

    "Annual_Crops": {
        # Cereal stubble (wheat/barley/maize) remaining after July harvest;
        # most dangerous fuel in Greek agricultural plains (Thessaly, Macedonia,
        # Thrace).  Extremely fine, very low Mx; fires spread at 1–3 m/s in wind.
        # [CHE98]; [SB05] FM GR1 (short, sparse dry climate grass).
        "heat_content":  7500,   # BTU/lb  (dry straw, lower oil content)
        "surface_ratio": 3500,   # ft²/ft³ (very fine straw stems)
        "fuel_load":     0.034,  # lb/ft²  (≈ 1.7 kg/m² cut stubble)
        "fuel_depth":    1.00,   # ft      (30 cm standing stubble)
        "moisture_ext":  0.12,   # fraction (ignites at very low moisture)
        "bulk_density":  0.034,  # lb/ft³
    },

    "Abandoned_Agricultural": {
        # Post-agricultural fallow transitioning to pioneer scrub: Dittrichia
        # viscosa, Rubus spp., annual grasses, scattered Cistus.  Extremely
        # common in rural Greece due to land abandonment since 1980s.
        # [ARI10]: high fire frequency due to continuous fuel build-up.
        # [SB05] FM GS2 (moderate load dry climate grass-shrub).
        "heat_content":  7800,
        "surface_ratio": 2200,   # ft²/ft³
        "fuel_load":     0.028,  # lb/ft²  (≈ 1.4 kg/m²)
        "fuel_depth":    0.80,   # ft      (25 cm heterogeneous fuel bed)
        "moisture_ext":  0.15,
        "bulk_density":  0.035,  # lb/ft³
    },

    # ── PLANTATIONS & EXOTIC SPECIES ─────────────────────────────────────────

    "Eucalyptus": {
        # Eucalyptus camaldulensis (river red gum) and E. globulus (blue gum)
        # — widely planted across Attica, Peloponnese, Crete for timber,
        # wind-breaks and ornament.  Catastrophically flammable: bark strips
        # act as firebrands, volatile eucalyptus oil raises heat content,
        # fire crowns easily.  Responsible for rapid fire spread in several
        # major Greek fires (e.g. Kineta 2018, Mati 2018).
        # High fuel load includes bark strips + leaf litter + bark ribbons.
        # [XAN12], [FER00]; analogous to Australian forest fire models.
        "heat_content":  9500,   # BTU/lb  (eucalyptus oil ~10 000 BTU/kg)
        "surface_ratio": 1800,   # ft²/ft³
        "fuel_load":     0.092,  # lb/ft²  (≈ 4.5 kg/m²; litter + bark strips)
        "fuel_depth":    0.30,   # ft      (9 cm; compacted bark+leaf bed)
        "moisture_ext":  0.20,
        "bulk_density":  0.307,  # lb/ft³
    },

    # ── AGRICULTURAL / ORCHARD ───────────────────────────────────────────────

    "Olive_Grove": {
        # Olea europaea — dominant cultivated tree of Greek lowlands and
        # coastal areas; managed orchard with sparse ground fuel (mown grass
        # or bare soil between rows).
        # [DIM02]: low fire spread rate unless understory is overgrown.
        # [SB05] between FM TL1 and GS1.
        "heat_content":  8000,
        "surface_ratio": 1400,   # ft²/ft³
        "fuel_load":     0.020,  # lb/ft²  (≈ 1.0 kg/m²; pruning debris + grass)
        "fuel_depth":    0.30,   # ft      (9 cm ground fuel)
        "moisture_ext":  0.20,
        "bulk_density":  0.067,  # lb/ft³
    },

    "Vineyard": {
        # Vitis vinifera — extensive on Aegean islands, Peloponnese, Macedonia.
        # Sparse, heavily pruned woody vines; very little ground fuel unless
        # grass under-storey is present.  Very low fire spread potential.
        # [SB05] FM NB1 (no-burn, nearly non-combustible without ground fuel).
        "heat_content":  7800,
        "surface_ratio": 1200,   # ft²/ft³ (coarse woody prunings)
        "fuel_load":     0.010,  # lb/ft²  (≈ 0.5 kg/m²; sparse pruning debris)
        "fuel_depth":    0.20,   # ft      (6 cm ground layer)
        "moisture_ext":  0.20,
        "bulk_density":  0.050,  # lb/ft³
    },

    # ── RIPARIAN / WETLAND ────────────────────────────────────────────────────

    "Riparian_Vegetation": {
        # Platanus orientalis (oriental plane), Nerium oleander, Tamarix spp.,
        # Vitex agnus-castus, Populus spp. along rivers, streams and lake shores.
        # Higher year-round moisture content significantly reduces fire risk;
        # can burn in extreme drought.  [SB05] FM TL1 analogue.
        "heat_content":  7800,
        "surface_ratio": 1600,   # ft²/ft³
        "fuel_load":     0.025,  # lb/ft²  (≈ 1.2 kg/m²; moderate leaf litter)
        "fuel_depth":    0.30,   # ft
        "moisture_ext":  0.35,   # fraction (riparian moisture — same as beech)
        "bulk_density":  0.083,  # lb/ft³
    },

    # ── NON-COMBUSTIBLE ───────────────────────────────────────────────────────
    # Water is a zero-load fuel so valid_fuel = (w0 > 0) is False everywhere;
    # fire_model.py will produce p_spread = 0 for all Water cells.
    # get_fuel_at() returns {} for Water and Non_Combustible alike.

    "Water": {
        # Sea, lakes (Vegoritida, Kastoria, Ioannina, Trichonida, Yliki…),
        # rivers (Axios, Aliakmon, Acheloos, Evros…) and reservoirs.
        # All Rothermel parameters are zero — this fuel never ignites.
        "heat_content":  0,
        "surface_ratio": 0,
        "fuel_load":     0.0,
        "fuel_depth":    0.0,
        "moisture_ext":  1.0,    # arbitrarily high — impossible to dry out
        "bulk_density":  0.0,
    },

    # ── URBAN / WILDLAND-URBAN INTERFACE ─────────────────────────────────────
    # Both urban fuel types represent the ground-level combustible fraction only.
    # Structure-to-structure ignition (brand transport, radiation) is NOT modelled
    # by Rothermel; these parameters capture only how well the landscape fuel bed
    # around buildings and along roads can sustain propagating surface fire.
    #
    # Key references: [MEL10], [COH08], [NF1144], [BUT04].
    # [MEL10]: radiant heat flux from WUI fires; structure ignition dominated by
    #          brands and direct flame contact, not by ground surface spread.
    # [COH08]: combustible ground fuel in most Greek suburban areas ≈ 0.3–0.8 kg/m²
    #          (garden grass + woody debris), moisture of extinction > 35 %.
    # [NF1144]: standard requires defensible-space vegetation management within
    #           30 m of structures; beyond that, ground fuel is similar to
    #           Abandoned_Agricultural with sparse coverage.

    "Urban_Fabric": {
        # Discontinuous urban fabric: suburban residential, mixed-use zones,
        # tourist development on Aegean islands, peri-urban sprawl of Athens,
        # Thessaloniki, Patras.  Fire spread through this class in the Rothermel
        # model represents the GROUND-LEVEL fuel only — sparse garden grass,
        # ornamental shrubs, wooden fencing and debris between buildings.
        # Buildings themselves act as fuel sinks (stone/concrete wall heat
        # absorption raises effective Mx); this is captured by the high Mx value.
        #
        # [MEL10]: surface spread rate through suburban zones 0.03–0.12 m/s
        #          (20–30× slower than adjacent wildland fuels).
        # [COH08]: 0.3–0.8 kg/m² exposed combustible at standard Greek suburban
        #          density; defensible space vegetation management lowers w0.
        # Closest standard model: Scott & Burgan FM NB8 adjusted for WUI.
        "heat_content":  8000,   # BTU/lb  (mixed wood/garden vegetation)
        "surface_ratio":  800,   # ft²/ft³ (coarse wood + mixed debris, ~50 mm)
        "fuel_load":     0.010,  # lb/ft²  (≈ 0.5 kg/m² sparse garden combustibles)
        "fuel_depth":    0.50,   # ft      (15 cm heterogeneous ground fuel)
        "moisture_ext":  0.40,   # fraction (irrigated gardens, masonry heat sink)
        "bulk_density":  0.020,  # lb/ft³  (= w₀/δ)
    },

    "Urban_Road": {
        # Road network: motorways (E75, A1, A2, Egnatia Odos), national roads,
        # provincial and municipal streets throughout Greece.
        # The road SURFACE (asphalt / concrete) is non-combustible; this fuel
        # class represents only the ROADSIDE VERGE — sparse annual grass and
        # road debris between the kerb and the adjacent land.
        # In practice this produces near-zero ROS — roads act as firebreaks and
        # effectively block or channel fire spread in the landscape.
        #
        # [MEL10]: paved roads are commonly used as backfire lines in Greece;
        #          spread across a 6–8 m paved road is rare under moderate wind.
        # Closest standard model: Scott & Burgan FM NB1 (nearly non-burnable).
        "heat_content":  7500,   # BTU/lb  (dry roadside grass / organic debris)
        "surface_ratio": 2000,   # ft²/ft³ (fine grass stems)
        "fuel_load":     0.003,  # lb/ft²  (≈ 0.15 kg/m²  very sparse verge)
        "fuel_depth":    0.10,   # ft      (3 cm  thin verge)
        "moisture_ext":  0.12,   # fraction (exposed, dries fast — but w0 so low
                                 #           that ROS is always near zero anyway)
        "bulk_density":  0.030,  # lb/ft³
    },
}

# ── Convenience sets for quick classification checks ─────────────────────────

# Fuels that cannot sustain fire propagation (w0 = 0 or structurally inert).
# Landscape.get_fuel_at() returns {} for all of these; fire_model.py suppresses
# them via the valid_fuel = (w0 > 0) mask in _precompute_ros_grid().
NON_BURNING_FUELS = {"Water", "Urban_Fabric", "Urban_Road", "Non_Combustible"}

# Fuels treated as non-combustible by the landscape (appended outside GREEK_FUELS
# by Landscape.__init__ so the model never looks them up in this dict).
# "Non_Combustible" covers: roads, railways, urban fabric, bare rock, burnt scars.

# ── Canonical display colours (RGBA 0-255) for every fuel type ───────────────
# Used by fire_viewer_3d.py (3-D overlay), plot_results() (2-D fuel panel), and
# the GUI analysis tab so every visualisation shows consistent colours that
# visually match the ecological character of each fuel.
# Colour derivation follows EEA CORINE Land Cover colour convention where possible.
FUEL_DISPLAY_COLORS: dict = {
    # Conifer forests — dark greens
    "Aleppo_Pine":             (34,  120,  34, 160),
    "Black_Pine":              (20,  100,  20, 160),
    "Maritime_Pine":           (25,  110,  30, 160),
    "Stone_Pine":              (50,  140,  50, 155),
    "Cypress":                 (10,   80,  30, 155),
    "Greek_Fir":               (30,  130,  60, 160),
    # Broadleaf forests — mid greens
    "Oak_Forest":              (80,  160,  40, 155),
    "Chestnut_Forest":         (90,  150,  30, 155),
    "Beech_Forest":            (100, 170,  50, 155),
    # Shrublands — olive / yellow-greens
    "Maquis_Dense_Shrub":      (120, 155,  45, 155),
    "Tall_Maquis":             (110, 145,  40, 155),
    "Phrygana_Low_Scrub":      (180, 200, 100, 150),
    "Garrigue":                (170, 185,  75, 150),
    # Grass / agricultural — yellows / tans
    "Dry_Grass":               (210, 200,  75, 145),
    "Annual_Crops":            (225, 215,  95, 140),
    "Abandoned_Agricultural":  (185, 185, 105, 140),
    # Plantations / orchards
    "Eucalyptus":              (50,  160,  70, 155),
    "Olive_Grove":             (150, 168,  58, 150),
    "Vineyard":                (195, 158,  38, 145),
    # Riparian / water
    "Riparian_Vegetation":     (50,  170, 100, 155),
    "Water":                   (30,   80, 200, 200),
    # Urban / roads
    "Urban_Fabric":            (158, 148, 148, 180),
    "Urban_Road":              (100,  92,  90, 200),
    # Non-combustible (rock, bare soil, burnt scars)
    "Non_Combustible":         (120, 115, 108, 130),
}
# Fallback for any fuel name not explicitly listed above
FUEL_DISPLAY_COLOR_DEFAULT = (128, 128, 100, 120)


def fuel_color_rgba(name: str) -> tuple:
    """Return (R, G, B, A) 0-255 display colour for a fuel type name."""
    return FUEL_DISPLAY_COLORS.get(name, FUEL_DISPLAY_COLOR_DEFAULT)


def fuel_colors_for_names(fuel_names: list) -> list:
    """Return a list of (R,G,B,A) colours matching each entry in fuel_names.

    Safe for any length — unknown names receive the default grey.
    Intended for building matplotlib ListedColormap and legend patches.
    """
    return [fuel_color_rgba(n) for n in fuel_names]


