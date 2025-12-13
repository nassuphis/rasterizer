import math
import numpy as np
import colorsys
from scipy.ndimage import gaussian_filter, sobel, laplace, gaussian_laplace

DEFAULT_GAMMA = 1

_COLOR_NAME_MAP = {
    "black":      "000000",
    "white":      "FFFFFF",
    "red":        "FF0000",
    "green":      "00FF00",
    "blue":       "0000FF",
    "yellow":     "FFFF00",
    "cyan":       "00FFFF",
    "magenta":    "FF00FF",

    # Extra basics / darks
    "gray":       "808080",
    "lightgray":  "D3D3D3",
    "darkgray":   "A9A9A9",
    "navy":       "000080",
    "teal":       "008080",
    "olive":      "808000",
    "maroon":     "800000",

    # Warmer tones
    "orange":     "FFA500",
    "brown":      "A52A2A",
    "gold":       "FFD700",
    "coral":      "FF7F50",

    # Cooler / pastel-ish
    "skyblue":    "87CEEB",
    "darkgreen":  "006400",
    "lightgreen": "90EE90",

    # Purples & pinks
    "purple":     "800080",
    "violet":     "EE82EE",
    "indigo":     "4B0082",
    "pink":       "FFC0CB",
    "orchid":     "DA70D6",

    # Metals
    "silver":     "C0C0C0",

    # Reds / oranges / warm shades
    "crimson":        "DC143C",
    "salmon":         "FA8072",
    "lightsalmon":    "FFA07A",
    "tomato":         "FF6347",
    "orangered":      "FF4500",
    "darkorange":     "FF8C00",
    "goldenrod":      "DAA520",
    "darkgoldenrod":  "B8860B",
    "sienna":         "A0522D",
    "chocolate":      "D2691E",
    "sandybrown":     "F4A460",
    "peru":           "CD853F",
    "tan":            "D2B48C",
    "khaki":          "F0E68C",
    "darkkhaki":      "BDB76B",
    "firebrick":      "B22222",
    "darkred":        "8B0000",

    # Greens
    "lime":           "00FF00",  # alias to green
    "forestgreen":    "228B22",
    "seagreen":       "2E8B57",
    "mediumseagreen": "3CB371",
    "darkseagreen":   "8FBC8F",
    "springgreen":    "00FF7F",
    "lawngreen":      "7CFC00",
    "chartreuse":     "7FFF00",
    "olivedrab":      "6B8E23",

    # Blues / cyans
    "deepskyblue":      "00BFFF",
    "dodgerblue":       "1E90FF",
    "royalblue":        "4169E1",
    "steelblue":        "4682B4",
    "cornflowerblue":   "6495ED",
    "cadetblue":        "5F9EA0",
    "lightsteelblue":   "B0C4DE",
    "powderblue":       "B0E0E6",
    "turquoise":        "40E0D0",
    "mediumturquoise":  "48D1CC",
    "darkturquoise":    "00CED1",
    "lightseagreen":    "20B2AA",
    "midnightblue":     "191970",
    "darkslateblue":    "483D8B",

    # Purples / pinks
    "plum":             "DDA0DD",
    "thistle":          "D8BFD8",
    "mediumorchid":     "BA55D3",
    "darkorchid":       "9932CC",
    "darkmagenta":      "8B008B",
    "mediumvioletred":  "C71585",
    "deeppink":         "FF1493",
    "hotpink":          "FF69B4",
    "palevioletred":    "DB7093",

    # Very light / near-whites
    "mintcream":        "F5FFFA",
    "honeydew":         "F0FFF0",
    "aliceblue":        "F0F8FF",
    "ghostwhite":       "F8F8FF",
    "lavender":         "E6E6FA",
    "ivory":            "FFFFF0",
    "linen":            "FAF0E6",
    "oldlace":          "FDF5E6",
    "seashell":         "FFF5EE",
    "snow":             "FFFAFA",
    "floralwhite":      "FFFAF0",
    "beige":            "F5F5DC",

    # Grays / slates
    "slategray":        "708090",
    "darkslategray":    "2F4F4F",
    "lightslategray":   "778899",
    "gainsboro":        "DCDCDC",
    "dimgray":          "696969",

    # A few fun, slightly custom vibes
    "sunset":           "FFCC66",
    "ocean":            "006994",
    "forest":           "0B6623",
    "rose":             "FF66CC",
    "mint":             "98FF98",
    "peach":            "FFDAB9",
    "sand":             "C2B280",
    "charcoal":         "36454F",
    "coolgray":         "8C92AC",
    "warmwhite":        "FFF8E7",

    # Variants / other metals
    "lightsilver": "D8D8D8",
    "rosegold":    "B76E79",
    "copper":      "B87333",
    "bronze":      "CD7F32",
    "brass":       "B5A642",
    "platinum":    "E5E4E2",
    "steel":       "71797E",
    "iron":        "43464B",

    # What you asked for
    "nickel":      "727472",
    "nikel":       "727472",  # alias, in case you type it this way
    "cobalt":      "0047AB",  # cobalt blue pigment

    # Extra shiny-ish gray
    "chrome":      "B4B4B4",
    "titanium":    "8D8F91",

    # Racing / deep classic greens
    "racinggreen":        "004225",
    "britishracinggreen": "004225",  # alias

    # Gem / luxury greens
    "emerald":            "50C878",
    "jade":               "00A86B",
    "kellygreen":         "4CBB17",
    "shamrock":           "33CC99",

    # Natural / earthy greens
    "mossgreen":          "8A9A5B",
    "fern":               "4F7942",
    "pine":               "01796F",
    "jungle":             "29AB87",
    "avocado":            "568203",
    "sage":               "B2AC88",
    "wasabi":             "A8C545",

    # Soft / pastel greens
    "teagreen":           "D0F0C0",
    "seafoam":            "9FE2BF",

    # Loud / neon greens
    "neongreen":          "39FF14",
    "limepunch":          "C7EA46",

    # Vintage French ink swatches (from the color card)

    "rawsienna": "B87535",        # Rawsienna (Terre de Sienne)
    "terre_de_sienne": "B87535",

    "burntsienna": "97442A",      # Burntsienna (Sienne brûlée)
    "sienne_brulee": "97442A",

    "redbrown": "944544",         # Red-brown (Brun rouge)
    "brun_rouge": "944544",

    "sepia": "55381F",            # Sepia (Sépia)
    "sepia_fr": "55381F",

    "scarlet_ink": "D9534B",      # Scarlet (Écarlate)
    "ecarlate": "D9534B",

    "carmine_ink": "D2475C",      # Carmine (Carmin)
    "carmin": "D2475C",

    "vermillion_ink": "D85734",   # Vermilion (Vermillon)
    "vermillon": "D85734",

    "orange_ink": "DA6F34",       # Orange (Orange)
    "orange_fr": "DA6F34",

    "yellow_ink": "E3BA45",       # Yellow (Jaune)
    "jaune": "E3BA45",

    "lightgreen_ink": "83C0A0",   # Light green (Vert clair)
    "vert_clair": "83C0A0",

    "darkgreen_ink": "8CAD96",    # Dark green (Vert foncé)
    "vert_fonce": "8CAD96",

    "indigo_ink": "191B46",       # Indigo (Indigo)
    "indigo_fr": "191B46",

    "prussianblue_ink": "292791", # Prussian blue (Bleu de Prusse)
    "bleu_de_prusse": "292791",

    "ultramarine_ink": "2234A4",  # Ultramarine (Outremer)
    "outremer": "2234A4",

    "cobaltblue_ink": "4371BC",   # Cobalt blue (Bleu de cobalt)
    "bleu_de_cobalt": "4371BC",

    "purple_ink": "C54B93",       # Purple / magenta-ish (Pourpre)
    "pourpre": "C54B93",

    "lightviolet_ink": "452782",  # Light violet (Violet clair)
    "violet_clair": "452782",

    "darkviolet_ink": "292397",   # Dark violet (Violet foncé)
    "violet_fonce": "292397",

    "gray_ink": "363434",         # Gray (Gris)
    "gris_fr": "363434",

    "neutral_tint": "251E32",     # Neutral tint (Teinte neutre)
    "teinte_neutre": "251E32",

    "black_ink": "111010",        # Black (Noir)
    "noir_fr": "111010",


    # Spectral paper colors (from the wavelength table, approximate RGB)

    "spectral_red": "FF0000",         # Rouge spectral
    "rouge_spectral": "FF0000",

    "vermillion_spectral": "FF5300",  # Vermillon
    "vermillon_spectral": "FF5300",

    "minium_spectral": "FFA900",      # Minium (red lead)
    "minium": "FFA900",

    "spectral_orange": "FFBE00",      # Orangé
    "orange_spectral": "FFBE00",

    "pale_chrome_yellow": "FFF900",   # Jaune (chrome pâle)
    "jaune_chrome_pale": "FFF900",

    "yellow_greenish": "D2FF00",      # Jaune verdâtre
    "jaune_verdatre": "D2FF00",

    "yellow_green": "BFFF00",         # Vert jaune
    "vert_jaune": "BFFF00",

    "spectral_green": "85FF00",       # Vert
    "vert_spectral": "85FF00",

    "emerald_green_spectral": "44FF00",  # Vert émeraude
    "vert_emeraude_spectral": "44FF00",

    "cyan_blue_2": "00FF9D",          # Bleu cyané n° 2
    "bleu_cyane_2": "00FF9D",

    "ultramarine_natural": "00B9FF",  # Outremer naturel
    "outremer_naturel_spectral": "00B9FF",

    "ultramarine_artificial": "0036FF",  # Outremer artificiel
    "outremer_artificiel_spectral": "0036FF",

    "spectral_violet": "5100FF",      # Violet
    "violet_spectral": "5100FF",

}

def parse_color_spec(spec: str, default: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Parse a color spec into (R,G,B) in 0..255.

    Accepts:
        - "RRGGBB" hex
        - "#RRGGBB" hex
        - simple names: red, blue, yellow, ...
    """
    if not isinstance(spec, str):
        return default

    s = spec.strip()
    if not s:
        return default

    if s.startswith("#"):
        s = s[1:]

    # Name → hex mapping
    lower = s.lower()
    if lower in _COLOR_NAME_MAP:
        s = _COLOR_NAME_MAP[lower]

    if len(s) != 6:
        return default

    try:
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
    except ValueError:
        return default
    return float(r), float(g), float(b)

COLOR_TRI_STRINGS = {

    "redgold": "firebrick:black:sunset",
    "rg": "firebrick:black:sunset",
    "greencopper": "seagreen:black:copper",
    "gc": "seagreen:black:copper",
    "royalblueplatinum": "royalblue:black:platinum",
    "bp": "royalblue:black:platinum",
    "bauhaus_primaries":  "0038A8:F5F0E6:D00000",  # (-1 deep blue, 0 warm paper, +1 poster red)
    "itten_blue_orange":  "264653:F1FAEE:E76F51",  # (-1 blue-green, 0 light neutral, +1 red-orange)
    "scientific_red_blue":"313695:F7F7F7:A50026",  # (ColorBrewer-style diverging)
    "swiss_modern":       "444444:FAFAFA:E2001A",  # (-1 dark gray, 0 white, +1 Swiss red)
    "pastel_soft":        "A8DADC:F1FAEE:FFB4A2",  # (-1 aqua, 0 off-white, +1 pastel coral)
    "midcentury_earth":   "386641:F2E8CF:BC6C25",  # (-1 forest, 0 cream, +1 ochre/burnt orange)
    "cyberpunk_neon":     "00F5FF:351B3F:FF0A81",  # (-1 cyan, 0 dark purple, +1 hot magenta)
    "thermal_blackbody":  "00004B:3F3F3F:FFE45E",  # (-1 deep blue, 0 dark neutral, +1 hot yellow)
    "vintage_print":      "007F7F:FFF1D0:D7263D",  # (-1 cyan ink, 0 old paper, +1 magenta-red ink)
    "munsell_balanced":   "009080:808080:E05000",  # (-1 teal, 0 mid gray, +1 orange; similar lightness)
    #
    "bauhaus_blue_yellow":  "002C7F:F6F1E1:FFC400",  # -1 deep ultramarine, 0 warm paper, +1 Bauhaus yellow
    "fauvism_wild":         "005F73:FFF3B0:FF006E",  # -1 teal, 0 acid pale yellow, +1 hot magenta
    "impressionist_pastel": "8E9FE6:FFF7E6:F6B189",  # -1 periwinkle, 0 light cream, +1 peach
    "expressionist_moody":  "1B3B6F:4A4A4A:D1495B",  # -1 dark blue, 0 charcoal, +1 moody red‑pink
    "brutalist_concrete":   "1E1E1E:C4C4C4:FFD100",  # -1 near‑black, 0 concrete gray, +1 hazard yellow
    "minimal_blue":         "101820:F4F4F4:2D7FF9",  # -1 deep ink, 0 clean white, +1 accent blue
    "material_teal_orange": "00695C:FAFAFA:FF8F00",  # -1 teal, 0 off‑white, +1 material amber
    "terrain_elevation":    "005B96:A4DE02:C38E70",  # -1 sea blue, 0 bright grass, +1 highland brown
    "oceanographic_deep":   "03045E:CAF0F8:FF6F61",  # -1 abyss blue, 0 pale water, +1 living coral
    "psychedelic_70s":      "008B8B:FFF7D6:B5179E",  # -1 dark cyan, 0 warm cream, +1 electric purple
    "grayscale_magnitude":  "1A1A1A:7F7F7F:F5F5F5",  # -1 dark, 0 mid gray, +1 light; no hue, just magnitude
    "artdeco_jewel":        "004E64:F5F0E8:D4A017",  # -1 teal jewel, 0 ivory, +1 antique gold

     # --- Natural materials & metal (-1 shadow, 0 material, +1 metal/highlight) ---

    "stone_bronze":   "545E63:C2B8A3:CD7F32",  # slate shadow, warm stone, bronze
    "wood_copper":    "3B2F2F:8B5A2B:B87333",  # bark, mid wood, copper
    "concrete_steel": "2F3133:A8A8A8:D0D7DF",  # dark steel, concrete, polished steel
    "sand_iron":      "43464B:E2C290:FFF1C1",  # iron shadow, desert sand, sunlit sand
    "marble_gold":    "6E7B8B:F5F5F0:FFD700",  # cool marble veins, white marble, gold

    # --- Marvel comics (bold, punchy primaries) ---

    "marvel_spiderman": "003366:F0F0F0:C00018",  # navy, city paper, Spidey red
    "marvel_ironman":   "1A1A1A:851313:FFD700",  # dark armor, deep red, gold
    "marvel_captain":   "002868:FFFFFF:BF0A30",  # flag blue, star white, flag red
    "marvel_hulk":      "1A1433:3A5F0B:B2FF59",  # gamma purple, Hulk green, glowing green
    "marvel_cosmic":    "240046:0B1724:FF6FF2",  # deep space purple, night, cosmic magenta

    # --- Pop art (bright, poster-y, halftone vibes) ---

    "pop_primary_dots": "0057A8:FFF4C7:F0142F",  # comic blue, pale yellow paper, pop red
    "pop_cyan_magenta": "00B8FF:FFEEDB:FF1E8A",  # cyan, warm off-white, hot pink
    "pop_lipstick":     "001B44:FCE4EC:E0004D",  # navy outline, light skin/pink, lipstick red
    "pop_banana":       "004D40:FFF59D:FFB300",  # deep teal, banana yellow, golden peel
    "pop_tv":           "3F51B5:F5F5F5:FFEB3B",  # TV blue, static white, bright yellow

    # --- Andy Warhol (acid color combos, Marilyn / Campbell vibes) ---

    "warhol_marilyn_cyan":  "008FD3:FFE0B2:FF00A0",  # cyan, peach skin, hot pink
    "warhol_marilyn_lime":  "4141FF:FFE0B2:C6FF00",  # cobalt, peach, neon lime
    "warhol_campbell":      "2E7D32:FFF3E0:C62828",  # can green, label cream, soup red
    "warhol_fluoro_blocks": "00E5FF:FFEB3B:FF3D00",  # aqua, acid yellow, fluoro orange
    "warhol_duotone_purple":"311B92:EDE7F6:F50057",  # deep violet, lilac paper, pink

    # --- Rocket launch (night sky, flame, smoke, glow) ---

    "rocket_night_launch":   "020819:4B4B55:FF6A00",  # deep night, smoke, flame orange
    "rocket_sunrise_launch": "1B2A49:FFECB3:FF7043",  # pre-dawn blue, sunrise, plume orange
    "rocket_plume_blue":     "001B44:CFD8DC:64FFDA",  # night blue, exhaust haze, turquoise plume
    "rocket_control_room":   "0D1B2A:1B263B:FFB703",  # dark consoles, panels, warning amber
    "rocket_heatshield":     "40241A:9E5E34:FFD166",  # char, hot tile, glowing heat

    # --- Tropical fish (reef, neon stripes, coral) ---

    "tropical_parrotfish":   "004E7A:00B8A9:F8E16C",  # deep teal, bright teal, yellow fin
    "tropical_clownfish":    "003049:FFFFFF:FF7B00",  # deep sea, white stripe, clown orange
    "tropical_reef":         "14213D:25CED1:FCA311",  # deep reef, turquoise, coral gold
    "tropical_angel":        "011627:FDFCDC:FF3366",  # dark water, pale body, pink accents
    "tropical_neon_tetra":   "001B44:00B4D8:FF006E",  # night blue, neon blue stripe, magenta

    "ink_scarlet_gold":      "black_ink:scarlet_ink:gold",
    "ink_prussian_copper":   "neutral_tint:prussianblue_ink:copper",
    "ink_ultramarine_brass": "black_ink:ultramarine_ink:brass",
    "ink_sepia_gold":        "sepia:warmwhite:gold",
    "ink_purple_emerald":    "neutral_tint:purple_ink:emerald",
    "ink_indigo_rosegold":   "indigo_ink:warmwhite:rosegold",
    "ink_neutral_silver":    "gray_ink:mintcream:silver",
    "ink_carmine_ivory":     "black_ink:carmine_ink:ivory",
    "ink_verdant_bronze":    "darkgreen_ink:lightgreen_ink:bronze",
    "ink_spectral_violet":   "neutral_tint:spectral_violet:platinum",

    "spectral_fire":         "spectral_red:minium_spectral:pale_chrome_yellow",
    "spectral_grass_sky":    "spectral_green:yellow_green:ultramarine_natural",
    "spectral_teal_magenta": "cyan_blue_2:cyan:purple_ink",
    "spectral_autumn":       "vermillion_spectral:minium_spectral:yellow_greenish",
    "spectral_spring":       "yellow_greenish:teagreen:rose",
    "spectral_lagoon":       "cyan_blue_2:seafoam:ocean",
    "spectral_chartreuse_ink":"yellow_green:vert_clair:indigo_ink",
    "spectral_deepsea":      "neutral_tint:ultramarine_artificial:cyan_blue_2",

    "forest_moss_gold":      "forest:mossgreen:gold",
    "forest_mint_copper":    "forest:mint:copper",
    "forest_night_silver":   "forest:neutral_tint:silver",
    "forest_sand_sun":       "forest:sand:sunset",
    "forest_sage_bronze":    "forest:sage:bronze",
    "stone_moss_verdigris":  "slategray:mossgreen:mediumturquoise",

    "urban_concrete_teal":   "charcoal:lightsteelblue:teal",
    "urban_charcoal_neon":   "charcoal:coolgray:neongreen",
    "urban_mist_plum":       "dimgray:gainsboro:plum",
    "urban_slate_coral":     "darkslategray:lightslategray:coral",
    "urban_navy_turquoise":  "navy:lightgray:turquoise",

    "ocean_foam_gold":       "ocean:mintcream:goldenrod",
    "ocean_abyss_coral":     "midnightblue:ocean:coral",
    "ocean_ice_emerald":     "ocean:powderblue:emerald",
    "ocean_mint_cobalt":     "ocean:mint:cobalt",
    "ocean_sage_sun":        "ocean:sage:sunset",

    "desert_dawn_rose":      "sienna:peach:rose",
    "desert_noon_turquoise": "peru:khaki:turquoise",
    "desert_dusk_violet":    "sandybrown:darkkhaki:violet",
    "desert_rock_copper":    "rawsienna:tan:copper",
    "desert_sand_indigo":    "sand:khaki:indigo",

    "pastel_mint_peach":     "mintcream:mint:peach",
    "pastel_lilac_sun":      "lavender:thistle:sunset",
    "pastel_seafoam_coral":  "teagreen:seafoam:lightsalmon",
    "pastel_rose_sky":       "honeydew:rose:skyblue",
    "pastel_sage_gold":      "beige:sage:gold",

    "metal_copper_patina":   "charcoal:copper:mediumturquoise",
    "metal_bronze_teal":     "iron:bronze:teal",
    "metal_iron_amber":      "iron:coolgray:darkorange",
    "metal_nickel_magenta":  "nickel:lightsilver:darkmagenta",
    "metal_steel_lime":      "steel:lightsilver:chartreuse",

    "paper_ivory_ink":       "charcoal:ivory:black_ink",
    "paper_aliceblue_ink":   "charcoal:aliceblue:indigo_ink",
    "paper_beige_ink":       "charcoal:beige:gray_ink",
    "paper_lavender_purple": "neutral_tint:lavender:purple_ink",
    "paper_linen_sepia":     "charcoal:linen:sepia",

    "retro_moss_orange":     "mossgreen:khaki:darkorange",
    "retro_teal_coral":      "teal:lightgreen:coral",
    "retro_olive_magenta":   "olive:khaki:deeppink",
    "retro_navy_mustard":    "navy:coolgray:goldenrod",
    "retro_peach_plum":      "tan:peach:plum",

    "neon_magenta_lime":     "darkslategray:deeppink:neongreen",
    "neon_cyan_yellow":      "charcoal:cyan:pale_chrome_yellow",
    "neon_violet_orange":    "darkslateblue:spectral_violet:orangered",
    "neon_green_pink":       "charcoal:neongreen:hotpink",
    "neon_blue_rose":        "charcoal:dodgerblue:rose",

    # --- Ink + spectral hybrids ---
    "ink_ultra_cyan":        "black_ink:ultramarine_natural:cyan_blue_2",
    "ink_sienna_violet":     "sepia:rawsienna:spectral_violet",
    "ink_cobalt_rose":       "neutral_tint:cobaltblue_ink:rose",
    "ink_gray_emerald":      "gray_ink:mintcream:emerald",
    "ink_scarlet_teal":      "scarlet_ink:warmwhite:teal",
    "ink_verdigris_gold":    "darkgreen_ink:lightgreen_ink:gold",
    "ink_sunset_platinum":   "black_ink:sunset:platinum",
    "ink_pourpre_seagreen":  "purple_ink:lightviolet_ink:seagreen",

    # --- Spectral continuous palettes ---
    "spectral_autumn_heat":  "vermillion_spectral:yellow_green:spectral_green",
    "spectral_sunset_wave":  "spectral_orange:ultramarine_artificial:cyan_blue_2",
    "spectral_chartreuse_blue":"yellow_greenish:chartreuse:ultramarine_natural",
    "spectral_magenta_cold": "spectral_violet:indigo_ink:aliceblue",
    "spectral_tropical":     "yellow_green:vert_emeraude_spectral:turquoise",
    "spectral_rose_marine":  "rouge_spectral:mint:outremer",
    "spectral_violet_garden": "violet_spectral:vert_fonce:lightgreen_ink",
    "spectral_warm_ocean":   "minium_spectral:peach:ocean",

    # --- Earth / forest / natural ---
    "forest_lichen_gold":    "forest:teagreen:gold",
    "forest_bark_sun":       "pine:sienna:sunset",
    "forest_jade_sand":      "jungle:jade:sand",
    "forest_avocado_silver": "avocado:sage:silver",
    "forest_deep_peach":     "racinggreen:mint:peach",
    "forest_stone_brown":    "pine:coolgray:brown",
    "forest_ice_blue":       "forest:mintcream:skyblue",
    "forest_moss_copper":    "mossgreen:sage:copper",

    # --- Marine / oceanographic / deep sea ---
    "ocean_midnight_coral":  "midnightblue:ocean:coral",
    "ocean_teal_gold":       "lightseagreen:powderblue:goldenrod",
    "ocean_blue_saffron":    "darkturquoise:lightsteelblue:khaki",
    "ocean_silver_cyan":     "steel:lightgray:cyan",
    "ocean_reef_pink":       "ocean:turquoise:pink",
    "ocean_abyss_emerald":   "charcoal:ultramarine_artificial:emerald",
    "ocean_foam_copper":     "cadetblue:mintcream:copper",
    "ocean_pearl_magenta":   "ocean:ivory:mediumvioletred",

    # --- Pastel & soft neutral themes ---
    "pastel_seashell_plum":  "seashell:beige:plum",
    "pastel_blush_turquoise":"snow:rose:turquoise",
    "pastel_cloud_violets":  "ghostwhite:lavender:violet",
    "pastel_mint_honey":     "mint:honeydew:gold",
    "pastel_sage_rose":      "sage:mintcream:rose",
    "pastel_cream_punch":    "floralwhite:ivory:limepunch",
    "pastel_sky_sand":       "powderblue:linen:sand",
    "pastel_breeze_orchid":  "aliceblue:seashell:orchid",

    # --- Metals / luxury / industrial ---
    "metal_gold_ink":        "chrome:ivory:scarlet_ink",
    "metal_copper_emerald":  "iron:copper:emerald",
    "metal_bronze_ivory":    "bronze:oldlace:platinum",
    "metal_nickel_purple":   "nickel:coolgray:darkorchid",
    "metal_silver_sunset":   "silver:gainsboro:sunset",
    "metal_steel_teal":      "steel:lightsilver:teal",
    "metal_chrome_rose":     "chrome:aliceblue:rose",
    "metal_titanium_coral":  "titanium:coolgray:coral",

    # --- Retro, warm, vintage palettes ---
    "retro_brown_aqua":      "brown:tan:lightseagreen",
    "retro_sienna_sky":      "sienna:khaki:skyblue",
    "retro_gold_violet":     "gold:sandybrown:darkviolet_ink",
    "retro_olive_coral":     "olive:darkkhaki:coral",
    "retro_maroon_cream":    "maroon:linen:cream",
    "retro_steel_pink":      "steel:lightsteelblue:hotpink",
    "retro_firebrick_peach": "firebrick:lightgray:peach",
    "retro_sage_blue":       "sage:coolgray:cornflowerblue",

    # --- Neon / high-contrast experimental ---
    "neon_blue_punch":       "darkslategray:dodgerblue:limepunch",
    "neon_cobalt_pink":      "cobalt:lightgray:deeppink",
    "neon_green_orange":     "neongreen:coolgray:orangered",
    "neon_purple_sun":       "darkmagenta:ghostwhite:darkorange",
    "neon_coral_mint":       "charcoal:coral:mint",
    "neon_cyan_gold":        "cyan:lightgray:gold",
    "neon_rose_indigo":      "rose:aliceblue:indigo",
    "neon_chartreuse_ink":   "chartreuse:mintcream:indigo_ink",

    "desert_brass_shadow":      "brass:sand:charcoal",
    "desert_bronze_mist":       "bronze:tan:gainsboro",
    "desert_chocolate_azure":   "chocolate:sandybrown:deepskyblue",
    "desert_khaki_violet":      "khaki:darkkhaki:violet",
    "desert_mint_twilight":     "mint:sandybrown:midnightblue",
    "desert_olive_sun":         "olive:darkkhaki:pale_chrome_yellow",
    "desert_peach_stone":       "peach:sand:sepia",
    "desert_rose_turquoise":    "sandybrown:rose:turquoise",
    "desert_rust_sky":          "burntsienna:khaki:skyblue",
    "desert_sage_copper":       "sage:sand:copper",
    "desert_sand_ink":          "sand:linen:black_ink",
    "desert_sepia_dust":        "sepia:sand:floralwhite",
    "desert_sienna_ice":        "sienna:khaki:aliceblue",
    "desert_spectral_green":    "minium_spectral:sandybrown:spectral_green",
    "desert_sunset_ink":        "sunset:sand:black_ink",
    "desert_warmwhite_ember":   "warmwhite:sienna:darkred",

    "forest_bronze_dawn":       "forest:bronze:peach",
    "forest_charcoal_mint":     "charcoal:forest:mint",
    "forest_emerald_haze":      "forestgreen:emerald:mintcream",
    "forest_lichen_platinum":   "mossgreen:teagreen:platinum",
    "forest_mist_ink":          "forest:teagreen:neutral_tint",
    "forest_neutral_gold":      "neutral_tint:mossgreen:gold",
    "forest_olive_cream":       "olive:khaki:floralwhite",
    "forest_sage_copper":       "sage:sand:copper",
    "forest_seafoam_pine":      "seafoam:pine:ivory",
    "forest_sienna_mist":       "sienna:teagreen:mintcream",
    "forest_sky_moss":          "skyblue:mossgreen:beige",
    "forest_tan_ink":           "tan:forest:gray_ink",
    "forest_tea_rose":          "teagreen:rose:linen",
    "forest_warmwhite_fir":     "forest:warmwhite:darkgreen",
    "forest_wasabi_cloud":      "wasabi:mintcream:ghostwhite",
    "forest_forest_parchment":  "forest:floralwhite:sepia",  # (alias of that idea if you like)

    "ink_carmine_peach":        "black_ink:carmine_ink:peach",
    "ink_copper_shadow":        "black_ink:copper:charcoal",
    "ink_emerald_ivory":        "neutral_tint:emerald:ivory",
    "ink_forest_parchment":     "forest:floralwhite:sepia",
    "ink_gold_lavender":        "neutral_tint:gold:lavender",
    "ink_graphite_aqua":        "charcoal:cyan_blue_2:mintcream",
    "ink_indigo_warmwhite":     "indigo_ink:warmwhite:goldenrod",
    "ink_jade_beige":           "black_ink:jade:beige",
    "ink_neutral_rose":         "neutral_tint:rose:seashell",
    "ink_ocean_mist":           "ocean:powderblue:mintcream",
    "ink_pourpre_honeydew":     "gray_ink:purple_ink:honeydew",
    "ink_prussian_mint":        "neutral_tint:prussianblue_ink:mint",
    "ink_rose_paper":           "gray_ink:rose:mintcream",
    "ink_sepia_linen":          "sepia:linen:gold",
    "ink_ultramarine_sage":     "black_ink:ultramarine_ink:sage",
    "ink_violet_sand":          "black_ink:darkviolet_ink:sand",

    "metal_bronze_ink":         "bronze:charcoal:black_ink",
    "metal_bronze_lagoon":      "bronze:ocean:teagreen",
    "metal_brass_shadow":       "brass:dimgray:charcoal",
    "metal_chrome_rosewood":    "chrome:rose:chocolate",
    "metal_cobalt_sand":        "cobalt:sand:platinum",
    "metal_copper_foam":        "copper:teagreen:mintcream",
    "metal_copper_violet":      "copper:purple:lavender",
    "metal_iron_frost":         "iron:coolgray:mintcream",
    "metal_iron_mint":          "iron:mint:seafoam",
    "metal_nickel_ember":       "nickel:darkred:gold",
    "metal_nickel_saffron":     "nickel:peach:pale_chrome_yellow",
    "metal_platinum_ember":     "platinum:firebrick:darkgoldenrod",
    "metal_platinum_leaf":      "platinum:teagreen:forestgreen",
    "metal_rosegold_cloud":     "rosegold:mintcream:ghostwhite",
    "metal_steel_ice":          "steel:lightsteelblue:aliceblue",
    "metal_titanium_moss":      "titanium:mossgreen:forest",

    "ocean_avocado_shell":      "avocado:ocean:seashell",
    "ocean_charcoal_ice":       "charcoal:ocean:aliceblue",
    "ocean_chartreuse_surf":    "chartreuse:ocean:mintcream",
    "ocean_cerulean_sand":      "deepskyblue:ocean:sand",
    "ocean_copper_spray":       "darkturquoise:copper:mintcream",
    "ocean_emerald_linen":      "ocean:emerald:linen",
    "ocean_ink_foam":           "ocean:mintcream:black_ink",
    "ocean_jungle_sun":         "jungle:ocean:pale_chrome_yellow",
    "ocean_mint_lagoon":        "mint:ocean:teagreen",
    "ocean_neon_plankton":      "ocean:neongreen:cyan_blue_2",
    "ocean_nickel_foam":        "nickel:ocean:mintcream",
    "ocean_pine_moon":          "pine:midnightblue:ghostwhite",
    "ocean_platinum_wave":      "midnightblue:platinum:powderblue",
    "ocean_spectral_violet":    "ocean:ultramarine_artificial:spectral_violet",
    "ocean_teal_rose":          "teal:ocean:rose",
    "ocean_ultramarine_peach":  "ultramarine_natural:ocean:peach",

    "pastel_azure_peach":       "aliceblue:skyblue:peach",
    "pastel_cloud_scarlet":     "ghostwhite:rose:scarlet_ink",
    "pastel_copper_blossom":    "peach:copper:rose",
    "pastel_fern_honey":        "fern:honeydew:goldenrod",
    "pastel_ink_blue":          "mintcream:ultramarine_ink:skyblue",
    "pastel_linen_jade":        "linen:mint:jade",
    "pastel_mint_lilac":        "mint:ghostwhite:lavender",
    "pastel_mist_chartreuse":   "mintcream:powderblue:chartreuse",
    "pastel_moss_rose":         "teagreen:mossgreen:rose",
    "pastel_platinum_leaf":     "platinum:mintcream:teagreen",
    "pastel_rose_ivory":        "rose:ivory:mintcream",
    "pastel_shell_cyan":        "seashell:mintcream:cyan",
    "pastel_sky_melon":         "skyblue:honeydew:peach",
    "pastel_tea_ink":           "teagreen:linen:black_ink",
    "pastel_turquoise_paper":   "turquoise:mintcream:oldlace",
    "pastel_warmwhite_forest":  "warmwhite:teagreen:forest",

    "spectral_chartreuse_silver":"chartreuse:silver:dimgray",
    "spectral_copper_violet":   "vermillion_spectral:copper:spectral_violet",
    "spectral_emerald_sky":     "spectral_green:emerald:skyblue",
    "spectral_gold_pine":       "pale_chrome_yellow:forestgreen:pine",
    "spectral_ink_turquoise":   "black_ink:ultramarine_artificial:turquoise",
    "spectral_lime_ink":        "yellow_green:lightgreen_ink:neutral_tint",
    "spectral_midnight_aurora": "midnightblue:ultramarine_natural:limepunch",
    "spectral_mint_coral":      "emerald_green_spectral:mint:coral",
    "spectral_peach_forest":    "minium_spectral:peach:forest",
    "spectral_plum_foam":       "spectral_violet:plum:mintcream",
    "spectral_rose_cloud":      "spectral_red:rose:ghostwhite",
    "spectral_sage_sunrise":    "yellow_greenish:sage:peach",
    "spectral_scarlet_ice":     "scarlet_ink:powderblue:mintcream",
    "spectral_teal_saffron":    "spectral_orange:teal:pale_chrome_yellow",
    "spectral_violet_sand":     "spectral_violet:sand:floralwhite",
    "spectral_cyan_amber":      "cyan_blue_2:warmwhite:darkorange",

    "urban_charcoal_aqua":      "charcoal:teal:seafoam",
    "urban_concrete_rose":      "gainsboro:slategray:rose",
    "urban_coolgray_coral":     "coolgray:lightgray:coral",
    "urban_indigo_amber":       "indigo:slategray:darkorange",
    "urban_ink_cyan":           "black_ink:slategray:cyan",
    "urban_midnight_magenta":   "midnightblue:darkslategray:deeppink",
    "urban_midnight_signal":    "midnightblue:dimgray:goldenrod",
    "urban_navy_limepunch":     "navy:charcoal:limepunch",
    "urban_neon_orchid":        "darkslategray:orchid:neongreen",
    "urban_neutral_lime":       "coolgray:dimgray:limepunch",
    "urban_paper_graffiti":     "charcoal:mintcream:hotpink",
    "urban_plum_glass":         "dimgray:plum:aliceblue",
    "urban_royal_signal":       "royalblue:darkgray:yellow",
    "urban_silver_emerald":     "charcoal:silver:emerald",
    "urban_steel_chartreuse":   "steel:coolgray:chartreuse",
    "urban_teal_neon":          "darkslategray:teal:neongreen",
}

COLOR_LONG_STRINGS = {

    "test_long": (
        "red:green:blue:red:green:blue:red:green:blue"
    ),

    # Fantastic Four: deep space, suit white, FF blue, various blues & highlights
    "marvel_fantasticfour_long": (
        "001533:F5F8FF:0059D6:001A4D:1FA4FF:003A8C:"
        "F5F8FF:00214D:4DA3FF:000814:CCE0FF:003366:FFFFFF"
    ),
    #   0 deep navy (neg)
    #   1 suit white (zero)
    #   2 FF blue (pos)
    #   6 suit white again (middle, zero)

    # The Thing: rocky orange vs blue trunks
    "marvel_thing_long": (
        "3A1C0A:D77A21:004FAD:4D2A10:FF9B30:D77A21:"
        "0B1A3D:FFB866:05122C:7FB0FF:000814"
    ),
    #   0 deep rock shadow (neg)
    #   1 warm rock orange (zero)
    #   2 blue trunks (pos)
    #   5 rock orange again (middle, zero)

    # Doctor Doom: Latverian green cloak and cold armor
    "marvel_drdoom_long": (
        "10120F:1F5D3A:C8CACC:0B2618:3E8C54:1F5D3A:"
        "555B60:8FA29F:1D2227:C8CACC:050608"
    ),
    #   0 dark iron (neg)
    #   1 Doom green (zero)
    #   2 bright steel (pos)
    #   5 Doom green again (middle, zero)

    # Ultron: gunmetal body with red energy core
    "marvel_ultron_long": (
        "12141A:70747C:FF1133:2B2F36:9EA3AC:70747C:"
        "3A3F49:C3CAD2:1C1F26:FF5A6B:0A0B0F"
    ),
    #   0 deep gunmetal (neg)
    #   1 mid steel gray (zero)
    #   2 red core glow (pos)
    #   5 mid steel gray again (middle, zero)

    # Spidey: navy -> paper -> red, then wavy between dark/light blues & warms
    "marvel_spiderman_long": (
        "003366:F0F0F0:C00018:001D3D:FF4B2B:F0F0F0:"
        "8C001A:FF8C00:001B44:FFD700:000814"
    ),
    #   0  navy (neg)
    #   1  paper white (zero)
    #   2  Spidey red (pos)
    #   5  paper again (middle, zero)
    #   dark/light interleaved across blues, reds, oranges, yellow

    # Iron Man: armor shadow -> red core -> gold, with bouncing brightness
    "marvel_ironman_long": (
        "1A1A1A:851313:FFD700:3D0A0A:FF8F00:851313:"
        "FFE082:3B2F2F:FFF176:2B2B2B:FFFDE7"
    ),
    #   0  dark armor (neg)
    #   1  deep red (zero)
    #   2  gold (pos)
    #   5  deep red again (middle, zero)
    #   wavy between dark maroons, hot ambers, and pale golds

    # Hulk: purple shadows + gamma greens + neon pops
    "marvel_hulk_long": (
        "1A1433:3A5F0B:B2FF59:143D12:8BC34A:3A5F0B:"
        "4CAF50:10300A:81C784:1B5E20:E6FFB3"
    ),
    #   0  purple shadow (neg)
    #   1  dark Hulk green (zero)
    #   2  neon green (pos)
    #   5  dark Hulk green again (middle, zero)
    #   oscillates between deep greens and bright/lime/pale greens

    # Cosmic: deep space, magenta, cyan, yellow, back into dark & starlight
    "marvel_cosmic_long": (
        "240046:0B1724:FF6FF2:111827:4CC9F0:3C096C:"
        "0B1724:7B2CBF:FEE440:4361EE:FF9E00:03071E:F0F0F5"
    ),
    #   0  deep purple (neg)
    #   1  dark navy (zero)
    #   2  cosmic magenta (pos)
    #   6  dark navy again (middle, zero)
    #   lots of dark/light bouncing via cyan, yellow, blue, orange, starlight

    # Team palette: Cap blue / white / red + gold, teal, purple, etc.
    "marvel_team_long": (
        "002868:F5F5F5:BF0A30:1A1A1A:FFE082:0D1B2A:"
        "F5F5F5:00B4D8:FF6F00:7B2CBF:FFE082:003366:FFFDE7"
    ),
    #   0  flag blue (neg)
    #   1  paper white (zero)
    #   2  flag red (pos)
    #   6  paper white again (middle, zero)
    #   weaves through darks and brights: gold, deep blue, cyan, orange, purple

    # 1) Hard black/white striping + midgray center
    #    3-color version: black / white / black (pretty ugly)
    #    multipoint: strong banding / stripe-like transitions
    "debug_stripes_bw_11": (
        "000000:FFFFFF:000000:FFFFFF:000000:808080:"
        "FFFFFF:000000:FFFFFF:000000:FFFFFF"
    ),
    # idx 0..10, N=11 -> center at idx 5 (808080)

    # 2) Chaotic rainbow zigzag; center is neutral gray
    #    3-color: dark purple / neon yellow / electric blue (harsh)
    #    multipoint: wild oscillation through the spectrum
    "debug_rainbow_zigzag_13": (
        "2E004F:FFF700:0011FF:00FF5F:B00000:00F0FF:"
        "808080:FF00B4:004F4F:FF7F00:001A72:7FFF00:000000"
    ),
    # center (idx 6) = 808080

    # 3) "Thermal sawtooth": hot/cold repeating, zero is a dark gray
    #    3-color: deep navy / pale warm / bright orange (not balanced)
    #    multipoint: repeated dark->bright jumps like a saw
    "debug_thermal_saw_11": (
        "00003C:FFEEAA:FF8000:1C1C1C:FFD000:404040:"
        "FFA000:606060:FFE080:808080:FFF8C0"
    ),
    # center (idx 5) = 404040

    # 4) Bit-ish RGB ladder: marching through dark primaries to brights
    #    3-color: black / dark red / dark green (bad diverging)
    #    multipoint: interesting as a weird categorical-ish gradient
    "debug_rgb_bits_11": (
        "000000:200000:002000:000020:FF0000:00FF00:"
        "0000FF:00FFFF:FF00FF:FFFF00:FFFFFF"
    ),
    # center (idx 5) = 00FF00

    # 5) Sawtooth luminance: dark/light/dark/light with tiny hue shifts
    #    3-color: black / dark gray / light gray (very low interest)
    #    multipoint: a high-frequency luminance pattern along the range
    "debug_saw_luma_13": (
        "000000:333333:AAAAAA:222222:DDDDDD:111111:"
        "F0F0F0:2A2A2A:C8C8C8:1A1A1A:B0B0B0:080808:FFFFFF"
    ),
    # center (idx 6) = F0F0F0 (nearly white)
    "metal_checkerboard_13": (
        "101215:E2E3E5:2A2D32:F5F5F7:3B3F45:E0DFD8:7F848C:"
        "E0DFD8:3B3F45:F5F5F7:2A2D32:E2E3E5:101215"
    ),

     # 1) Cool chrome / mirror steel
    "metal_chrome_13": (
        "090B10:BFC3C9:F5F7FA:2C3139:D3D7DE:1B2027:"
        "BFC3C9:1B2027:D3D7DE:2C3139:F5F7FA:BFC3C9:090B10"
    ),

    # 2) Brushed aluminum (subtle, cool gray waves)
    "metal_brushed_aluminum_13": (
        "0E1013:AEB2B8:E4E7EB:3A3E44:C8CCD2:272B30:"
        "AEB2B8:272B30:C8CCD2:3A3E44:E4E7EB:AEB2B8:0E1013"
    ),

    # 3) Bronze (warm, statuary metal)
    "metal_bronze_13": (
        "23140C:9A6434:E0A35C:5B3419:C4833D:3A2111:"
        "9A6434:3A2111:C4833D:5B3419:E0A35C:9A6434:23140C"
    ),

    # 4) Copper (pipes, cookware, patina-ready)
    "metal_copper_13": (
        "1F120E:A15C34:E39A6A:5A2815:C87546:3A1B10:"
        "A15C34:3A1B10:C87546:5A2815:E39A6A:A15C34:1F120E"
    ),

    # 5) Rusted steel (oxidized orange + cool steel)
    "metal_rusted_steel_13": (
        "15171B:6E716F:C26A3A:243137:9A4F2C:3A474C:"
        "6E716F:3A474C:9A4F2C:243137:C26A3A:6E716F:15171B"
    ),

    # 6) Titanium (cool gray with blue/violet sheens)
    "metal_titanium_13": (
        "0C1016:7E8187:C4C7D0:303848:969AAF:222733:"
        "7E8187:222733:969AAF:303848:C4C7D0:7E8187:0C1016"
    ),

    # 7) Bi-metal: copper below zero, iron above
    "metal_bimetal_copper_iron_13": (
        "2A1308:7A746B:B7BCC3:4A2712:B66A36:D8945A:"
        "7A746B:9B9FA6:4A4F55:D7DADF:353A40:B7BCC3:101317"
    ),
    #   < 0 → mostly copper browns/oranges into the center
    #   > 0 → iron/steel grays and highlights

    # 8) Bi-metal: gold below zero, steel above
    "metal_bimetal_gold_steel_13": (
        "3A290A:8C7A3A:C0CBD5:A67C1F:F2C649:FFF0B0:"
        "8C7A3A:AEB7C1:4B5460:D9E1EA:333A44:C0CBD5:111318"
    ),

    # 9) Bi-metal: bronze below zero, silver above
    "metal_bimetal_bronze_silver_13": (
        "24130B:8F6033:D0D4DB:5A3118:C5843D:E8AF6A:"
        "8F6033:C0C5CC:4A4E54:E4E8EE:32353A:D0D4DB:0E1013"
    ),

    # 10) Bi-metal: cobalt-ish blue metal below, nickel above
    "metal_bimetal_cobalt_nickel_13": (
        "020819:3C5D9A:B5BABE:16356A:4F7AC4:A0B5E6:"
        "3C5D9A:9B9FA2:4E5356:D1D5D8:303335:B5BABE:0B0D0F"
    ),

    # Strong primaries, poster-y
    "bauhaus_primary_13": (
        "0038A8:F5F0E6:D00000:000000:F9E547:0038A8:F5F0E6:D00000:111111:F9E547:1E5AA8:F5F0E6:D00000"
    ),
    #   -1 deep cobalt blue, 0 warm paper, +1 poster red

    # Blue–yellow emphasis, very “schoolbook Bauhaus”
    "bauhaus_yellow_blue_13": (
        "002B7F:FAF3DD:FFC600:111111:005BBB:FFF1B2:FAF3DD:FFC600:003566:FFDD00:1A1A1A:FAF3DD:FFC600"
    ),

    # Red / black / white / yellow – classic graphic posters
    "bauhaus_red_black_13": (
        "111111:F5F5F5:D10000:2B2B2B:FFCC00:111111:F5F5F5:D10000:4A4A4A:FFE066:000000:F5F5F5:D10000"
    ),

    # Softer Bauhaus-influenced pastels
    "bauhaus_pastel_13": (
        "005F73:F8F3E6:FFB703:0B3C49:EAE2B7:0081A7:F8F3E6:FFB703:FF7B00:F4D58D:001427:F8F3E6:FFB703"
    ),

    # Minimal, UI-ish but still Bauhaus primaries
    "bauhaus_minimal_13": (
        "202124:FAFAFA:E53935:121212:FBC02D:1E88E5:FAFAFA:E53935:424242:FFF59D:1565C0:F5F5F5:E53935"
    ),

    # Grid / composition with black lines and primary blocks
    "bauhaus_grid_13": (
        "000000:F5F0E6:FF0000:0038A8:F9E547:000000:F5F0E6:FF0000:0038A8:F9E547:111111:F5F0E6:FF0000"
    ),

    # Shapes: blue (square), yellow (circle), red (triangle), black outlines
    "bauhaus_shapes_13": (
        "0038A8:FFF7E0:FFD000:000000:D10000:F4E4B0:FFF7E0:FFD000:0038A8:FFE066:111111:FFF7E0:FFD000"
    ),

    # Muted Bauhaus – deep blues, warm creams, red accent
    "bauhaus_muted_13": (
        "1C2A3A:F4F1E8:E53935:3F4A5A:F2D16B:26415A:F4F1E8:E53935:5A6B7A:F7E4A8:121212:F4F1E8:E53935"
    ),

    # Blue / yellow with modern aqua variants
    "bauhaus_blue_yellow_13": (
        "003F88:FAF3DD:FFB703:001219:8ECAE6:FFCB77:FAF3DD:FFB703:219EBC:FFDD00:000000:FAF3DD:FFB703"
    ),

    # Mostly monochrome with sharp color accents
    "bauhaus_monochrome_accent_13": (
        "202124:FAFAFA:FFC107:111111:FF5252:1E88E5:FAFAFA:FFC107:424242:FFCA28:1565C0:F5F5F5:FFC107"
    ),

     # 1) Deep blue night, spiritual yellow, red accents
    "blaue_reiter_night_11": (
        "08152A:F4F0E8:F6E45C:102A54:C0392B:F4F0E8:"
        "26499B:F2994A:050814:C6D4F5:0B1B32"
    ),
    # -1 deep blue, 0 warm paper, +1 luminous yellow
    # Wavy through dark/light blues, red, orange, pale sky

    # 2) Franz Marc-ish horses: blue, green, yellow, orange
    "blaue_reiter_marc_horses_9": (
        "0B2340:F5EBDD:3FA34D:F2C94C:F5EBDD:"
        "2F6DB5:F2994A:6C4AA5:050814"
    ),
    # -1 deep blue, 0 light parchment, +1 vivid green
    # Center is parchment again; fields of blue, yellow, orange, violet around it

    # 3) Kandinsky-like multipatch: red, yellow, blue, green, purple
    "blaue_reiter_kandinsky_11": (
        "151640:F6F3EB:D6453D:F6E45C:1E5AA8:F6F3EB:"
        "2F9E44:5C3C8C:F2994A:2CA5A5:08152A"
    ),
    # -1 deep indigo, 0 light paper, +1 warm red
    # Middle is paper again; ring of yellow, blue, green, purple, orange, turquoise

    # 4) Spiritual blue/white: quiet and luminous
    "blaue_reiter_spiritual_7": (
        "0A1638:F7F4ED:1E5AA8:F7F4ED:7EC3E6:5C3C8C:020616"
    ),
    # -1 deep ultramarine, 0 ivory, +1 strong blue
    # Center ivory again, then soft cyan, violet, midnight

    # 5) Fields & hills: greens, yellows, blues
    "blaue_reiter_fields_9": (
        "062218:F6F2E6:8CBF26:F2C94C:F6F2E6:"
        "2C7C78:2F6DB5:4C3C72:050814"
    ),
    # -1 deep forest, 0 light cream, +1 yellow‑green
    # Middle cream again; waves of teal, blue, violet, night

    # 6) Storm over the city: dramatic sky, lightning, red
    "blaue_reiter_storm_11": (
        "101319:F5F0E7:233B8B:050709:D9A441:F5F0E7:"
        "134A63:B52A2A:2B1640:C8D3EB:000000"
    ),
    # -1 dark slate, 0 warm paper, +1 intense blue
    # Center paper again; mustard lightning, teal, blood red, violet, pale sky, black

    # 1) Ember / cinders: deep char, bone paper, blood red, flare yellows
    "blaue_reiter_apocalypse_ember_11": (
        "140C0C:F5EEDF:D32F2F:2A1A14:FFB300:F5EEDF:"
        "5C2A1A:FFD54F:3B0F0F:FFE082:000000"
    ),
    # -1 deep char red-brown, 0 pale bone paper, +1 blood red

    # 2) Yellow–black judgement: hard yellow & soot
    "blaue_reiter_apocalypse_yellowblack_9": (
        "050507:F6F1DE:FFD000:1A1308:F6F1DE:"
        "3B2907:FFEA7A:000000:FFF8D0"
    ),
    # -1 nearly black, 0 parchment, +1 harsh yellow

    # 3) Blood sky: dark blue horizon, pale light, blood red
    "blaue_reiter_apocalypse_bloodsky_11": (
        "050B1A:F4F0E5:8B0000:0F1C38:F2C94C:F4F0E5:"
        "3C1C1C:FFA726:1B1033:FFE082:000000"
    ),
    # -1 deep navy, 0 warm off-white, +1 dark blood red

    # 4) Four horsemen: gold, plague green, blood, shadow
    "blaue_reiter_apocalypse_four_horsemen_11": (
        "090909:F5EBD8:FFB703:4B1A21:9BC53D:F5EBD8:"
        "5E3C99:F9844A:1A1A1A:FFE066:000000"
    ),
    # -1 black, 0 parchment, +1 gold

    # 5) Scorched earth: burnt browns & embers
    "blaue_reiter_apocalypse_scorched_9": (
        "120A06:F6EDDD:C4511F:3B1C0D:F6EDDD:"
        "6E391A:FFB561:1A0C06:FFE0B2"
    ),
    # -1 very dark brown, 0 light ash paper, +1 scorch orange

    # 6) Smoke and ash: almost monochrome with one molten accent
    "blaue_reiter_apocalypse_smoke_ash_7": (
        "070708:F3F3F3:BB7A00:F3F3F3:514C48:E0D4C4:000000"
    ),
    # -1 black, 0 very light gray, +1 molten amber
    # Gene Davis–ish vertical stripes: black / paper / magenta, then stripey
    "washington_gene_davis_stripes_11": (
        "111111:F5F5F5:FF006E:004E92:FFD500:F5F5F5:"
        "00A8E8:FF5A00:2E2E2E:FFC0CB:000000"
    ),

    # Noland targets / chevrons: blue / paper / red, then teal / gold / blue / salmon
    "washington_noland_target_9": (
        "001F3F:F4F2E8:D62828:008C7E:F4F2E8:"
        "FFB703:2F5DA8:FEC5BB:111111"
    ),

    # Morris Louis veils: deep violet, paper, cool blue, then aqua and warm gold
    "washington_louis_veil_7": (
        "120F2E:F5F1E8:4F7CAC:F5F1E8:9BD5C0:F2C57C:2F1A24"
    ),

    # Downing dots / grids: very poppy primaries on neutral ground
    "washington_downing_dots_9": (
        "000000:F5F5F5:0077B6:F72585:F5F5F5:"
        "3A0CA3:FFBA08:03071E:FFF3B0"
    ),

    # Mehring-ish soft fields: muted but clear color blocks
    "washington_mehring_soft_11": (
        "1C1C25:F7F2EA:FF6F61:88C0D0:F2C94C:F7F2EA:"
        "A3BE8C:5E81AC:F4A261:2E3440:F9E6E6"
    ),

    # Leon Berkowitz luminous atmosphere: blue–violet and warm haze
    "washington_berkowitz_luminous_9": (
        "050716:F6F3ED:527EA3:BFD7EA:F6F3ED:"
        "F5D0A9:C46F70:1E2740:000000"
    ),

    # Sam Gilliam–ish stained / draped color: wine, indigo, orange, green, gold
    "washington_gilliam_drape_11": (
        "050208:F5EFE8:9D174D:2F195F:E87D1E:F5EFE8:"
        "167E5D:FFB703:6A040F:240046:FDF0D5"
    ),

     # High-contrast BW + primaries, classic stripe poster
    "washington_stripe_bw_primary_11": (
        "111111:F5F5F5:D32F2F:000000:FFFFFF:F5F5F5:"
        "000000:0033CC:000000:F5F5F5:FFFF00"
    ),
    # -1 dark gray, 0 light paper, +1 strong red

    # CMYK-ish: black/white, cyan/magenta/yellow between
    "washington_stripe_cmyk_11": (
        "000000:F5F5F5:00B4FF:000000:FF00AA:F5F5F5:"
        "000000:FFFF00:000000:F5F5F5:000000"
    ),
    # -1 black, 0 white, +1 cyan

    # Neon nightclub stripes: black/white + magenta, green, cyan, yellow
    "washington_stripe_neon_13": (
        "050505:F7F7F7:FF00E6:000000:39FF14:000000:"
        "F7F7F7:00F0FF:050505:FFE600:000000:F7F7F7:FF00E6"
    ),
    # -1 near black, 0 cool white, +1 neon magenta

    # Subtle grayscale stripes (good for more restrained stripe textures)
    "washington_stripe_subtle_9": (
        "202124:F5F5F5:8E8E8E:111111:F5F5F5:3A3A3A:"
        "DADADA:111111:E0E0E0"
    ),
    # -1 dark gray, 0 soft white, +1 mid gray

    # Teal–orange stripe pack (cinematic but stripey)
    "washington_stripe_teal_orange_11": (
        "081217:F5F1E8:FF6F00:031017:00A8A8:F5F1E8:"
        "1A1A1A:FFB703:031017:F5F1E8:FF6F00"
    ),
    # -1 deep blue/black, 0 warm off-white, +1 orange

    # Violent rainbow stripes: B/W with RGBY slices
    "washington_stripe_rainbow_13": (
        "000000:F5F5F5:FF0000:000000:FFA500:F5F5F5:"
        "F5F5F5:FFFF00:000000:00FF00:000000:0000FF:000000"
    ),
    # -1 black, 0 white, +1 red

    # Pure “ink on paper” stripes, very print-like
    "washington_stripe_ink_paper_9": (
        "050505:F5F0E6:1A1A1A:000000:F5F0E6:111111:"
        "F5F0E6:000000:F5F0E6"
    ),
    # -1 almost black, 0 warm paper, +1 dark gray

    # Cool–warm alternation: blue/gray vs coral
    "washington_stripe_coolwarm_11": (
        "060B12:F5F5F5:FF6F61:000000:4C82C3:F5F5F5:"
        "111111:F4A261:000000:F5F5F5:264653"
    ),

    "washington_stripe_sorbet_11": (
        "101010:FDF7F2:FF8FA3:000000:FFCFA0:FDF7F2:"
        "000000:7FD1FF:000000:FDF7F2:9C88FF"
    ),
    # -1 deep charcoal, 0 warm off-white, +1 soft pink; pastel stripes with black breaks
    
    # -1 deep navy, 0 white, +1 warm coral
    # Deep sea → sand → pines / rocks
    "beach_coastline_11": (
        "355A7C:EDE7CF:9EAF8A:6BAAD7:D4CFB2:EDE7CF:"
        "B7A88E:999B76:675D45:3E3C2F:191B1F"
    ),

    # Water / reef vibes: deep → turquoise → pale sand
    "beach_reef_11": (
        "355A7C:EDE7CF:6BAAD7:678499:5897D9:EDE7CF:"
        "739AAB:95B7C1:D4CFB2:C4BEA0:B7A88E"
    ),

    # Cliffs & pines: shadows, rock, scrub, sky
    "beach_cliff_pines_11": (
        "191B1F:EDE7CF:89805F:3E3C2F:675D45:EDE7CF:"
        "B7A88E:999B76:9EAF8A:D4CFB2:3B7BF9"
    ),

    # Soft pastel foam & shallows
    "beach_pastel_foam_9": (
        "355A7C:EDE7CF:95B7C1:6BAAD7:EDE7CF:"
        "D4CFB2:C4BEA0:B7A88E:999B76"
    ),

    # Darker / twilight reinterpretation
    "beach_twilight_9": (
        "191B1F:EDE7CF:2574EA:355A7C:EDE7CF:"
        "3E3C2F:675D45:89805F:D4CFB2"
    ),

    # Olive scrub + warm sand
    "beach_olive_sand_7": (
        "3E3C2F:EDE7CF:999B76:EDE7CF:"
        "B7A88E:D4CFB2:9EAF8A"
    ),

    # Clear sky + bright water + warm sand
    "beach_clear_sky_11": (
        "355A7C:EDE7CF:3B7BF9:6BAAD7:D4CFB2:EDE7CF:"
        "5897D9:95B7C1:C4BEA0:2574EA:B7A88E"
    ),
    # -1 deep water blue, 0 warm sand, +1 vivid sky blue

    # Turquoise cove: dark shadow water into light sand + turquoise
    "beach_turquoise_cove_9": (
        "191B1F:EDE7CF:6BAAD7:3E3C2F:EDE7CF:"
        "D4CFB2:739AAB:C4BEA0:B7A88E"
    ),
    # -1 deep shadow, 0 sand, +1 turquoise shallows

    # Pebble shore: rocks, sand, scrub
    "beach_pebble_shore_7": (
        "3E3C2F:EDE7CF:999B76:EDE7CF:"
        "B7A88E:D4CFB2:675D45"
    ),
    # -1 dark rock, 0 pale sand, +1 olive-sand mix

    # Sky gradient over Bugliaca: deep horizon into high, pale sky
    "bugliaca_sky_11": (
        "3F5160:B2D2DF:DEE6E9:5D727F:88BFD7:B2D2DF:"
        "67ABCD:5AA3C8:519CC1:DFE7EA:85AABC"
    ),

    # Mountain faces & snow fields
    "bugliaca_mountains_11": (
        "1A1F23:DFE7EA:85AABC:1D292F:8BAEBE:DFE7EA:"
        "3D647E:324957:BDD9E3:224459:26343B"
    ),

    # Dark pine forest with glimpses of sky
    "bugliaca_forest_11": (
        "1D2B2C:416052:88BFD7:30463C:3F5160:416052:"
        "2A302E:67ABCD:30463C:1D2B2C:253B3D"
    ),

    # Deep valley at dusk, mostly shadow blues and greens
    "bugliaca_valley_dusk_9": (
        "1A1F23:30463C:3D647E:1B3343:30463C:"
        "253B3D:1D292F:202426:1D2B2C"
    ),

    # Fence and meadow in front of the view
    "bugliaca_fence_meadow_9": (
        "202426:506467:416052:253B3D:506467:"
        "30463C:203035:2A302E:B2D2DF"
    ),

    # High glacier light: snow and high-altitude blues
    "bugliaca_glacier_9": (
        "1A1F23:DFE7EA:88BFD7:5D727F:DFE7EA:"
        "BDD9E3:85AABC:519CC1:DEE6E9"
    ),

    # Panorama mix: sky, forest, and rock together
    "bugliaca_panorama_11": (
        "1A1F23:B2D2DF:416052:3D647E:88BFD7:B2D2DF:"
        "30463C:5AA3C8:253B3D:DFE7EA:1D2B2C"
    ),

    # Misty/overcast interpretation of the same view
    "bugliaca_mist_9": (
        "1D292F:8BAEBE:DEE6E9:5D727F:8BAEBE:"
        "30463C:DFE7EA:26343B:BDD9E3"
    ),

    # Evergreen focus: dense dark pines
    "bugliaca_evergreen_7": (
        "1D2B2C:30463C:416052:30463C:2A302E:253B3D:1D2B2C"
    ),

    # Soft gradient, trunks → mid‑snow → highlights
    "winter_forest_soft_11": (
        "141318:6C7788:EFEEEB:262931:545C6B:6C7788:"
        "818B9B:97A0AE:ABB4C0:C0C9D3:EFEEEB"
    ),

    # Darker emphasis, more about trunks and shadow
    "winter_forest_deep_9": (
        "141318:545C6B:ABB4C0:262931:545C6B:"
        "3B414E:6C7788:818B9B:C0C9D3"
    ),

    # High‑contrast, compact palette
    "winter_forest_contrast_7": (
        "141318:818B9B:EFEEEB:818B9B:"
        "3B414E:C0C9D3:6C7788"
    ),

    # Wavy, alternating dark/light like tree vs snow bands
    "winter_forest_wavy_11": (
        "141318:6C7788:EFEEEB:262931:97A0AE:6C7788:"
        "C0C9D3:3B414E:ABB4C0:262931:EFEEEB"
    ),

    # Misty / atmospheric version, skewed to lighter values
    "winter_forest_mist_11": (
        "3B414E:ABB4C0:EFEEEB:545C6B:C0C9D3:ABB4C0:"
        "818B9B:97A0AE:EFEEEB:6C7788:141318"
    ),
    # 6) Deep night snow – darker overall
    "winter_forest_night_11": (
        "141318:6C7788:EFEEEB:262931:818B9B:6C7788:"
        "97A0AE:3B414E:ABB4C0:262931:EFEEEB"
    ),

    # 7) Sky glow – lighter mids, darker ends
    "winter_forest_skyglow_9": (
        "141318:ABB4C0:EFEEEB:3B414E:ABB4C0:"
        "C0C9D3:6C7788:97A0AE:141318"
    ),

    # 8) Blue veils – cool blue‑gray center, soft whites
    "winter_forest_blue_veils_9": (
        "141318:97A0AE:EFEEEB:3B414E:97A0AE:"
        "C0C9D3:545C6B:ABB4C0:141318"
    ),

    # 9) Ink drawing – stark trunks with limited snow tones
    "winter_forest_ink_7": (
        "141318:6C7788:C0C9D3:6C7788:262931:ABB4C0:141318"
    ),

}

COLOR_STRINGS = {}
COLOR_STRINGS.update(COLOR_TRI_STRINGS)
COLOR_STRINGS.update(COLOR_LONG_STRINGS)

# ---------------------------------------------------------------------------
# Histogram equalization
# ---------------------------------------------------------------------------

def _hist_equalize(values: np.ndarray, nbins: int = 256) -> np.ndarray:
    """
    Simple 1D histogram equalization on 'values', returning t in [0,1].

    We use a fixed number of bins and map each value to the CDF bin
    it falls into. This is O(N) and works well for large images.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.zeros_like(values, dtype=np.float64)

    vmin = float(values.min())
    vmax = float(values.max())

    if (not math.isfinite(vmin)) or (not math.isfinite(vmax)) or vmax <= vmin:
        return np.zeros_like(values, dtype=np.float64)

    hist, bin_edges = np.histogram(values, bins=nbins, range=(vmin, vmax))
    cdf = hist.cumsum().astype(np.float64)
    if cdf[-1] <= 0.0:
        return np.zeros_like(values, dtype=np.float64)
    cdf /= cdf[-1]

    # For each value, find its bin and pick the CDF
    idx = np.searchsorted(bin_edges, values, side="right") - 1
    idx = np.clip(idx, 0, nbins - 1)
    t = cdf[idx]
    return t


# ---------------------------------------------------------------------------
# HSV helpers
# ---------------------------------------------------------------------------


def _rgb255_to_hsv01(rgb255: tuple[float, float, float]) -> tuple[float, float, float]:
    r, g, b = rgb255
    return colorsys.rgb_to_hsv(float(r) / 255.0, float(g) / 255.0, float(b) / 255.0)

def _interp_hue_short_arc(h0: float, h1: float, t: np.ndarray) -> np.ndarray:
    """
    Hue interpolation on a circle, shortest path. h0,h1 in [0,1], t array in [0,1].
    """
    dh = h1 - h0
    if dh > 0.5:
        dh -= 1.0
    elif dh < -0.5:
        dh += 1.0
    return (h0 + dh * t) % 1.0

def _hsv01_to_rgb255_batch(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Vectorized HSV(0..1) -> RGB(0..255). Returns float64 (...,3).
    """
    c = v * s
    hp = h * 6.0
    x = c * (1.0 - np.abs((hp % 2.0) - 1.0))
    m = v - c

    r_ = np.zeros_like(h)
    g_ = np.zeros_like(h)
    b_ = np.zeros_like(h)

    c0 = (0.0 <= hp) & (hp < 1.0)
    c1 = (1.0 <= hp) & (hp < 2.0)
    c2 = (2.0 <= hp) & (hp < 3.0)
    c3 = (3.0 <= hp) & (hp < 4.0)
    c4 = (4.0 <= hp) & (hp < 5.0)
    c5 = (5.0 <= hp) & (hp < 6.0)

    r_[c0], g_[c0], b_[c0] = c[c0], x[c0], 0.0
    r_[c1], g_[c1], b_[c1] = x[c1], c[c1], 0.0
    r_[c2], g_[c2], b_[c2] = 0.0, c[c2], x[c2]
    r_[c3], g_[c3], b_[c3] = 0.0, x[c3], c[c3]
    r_[c4], g_[c4], b_[c4] = x[c4], 0.0, c[c4]
    r_[c5], g_[c5], b_[c5] = c[c5], 0.0, x[c5]

    r = (r_ + m) * 255.0
    g = (g_ + m) * 255.0
    b = (b_ + m) * 255.0
    return np.stack([r, g, b], axis=-1)

# ---------------------------------------------------------------------------
# Field normalization
# ---------------------------------------------------------------------------

def _two_sided_t_and_masks(v: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        t     : float64 (H,W) in [0,1]
        finite: bool (H,W)
        neg   : bool (H,W) finite & v<0
        pos   : bool (H,W) finite & v>0

    params:
        norm  : "linear" | "eq"  (default "linear")
        gamma : float (default DEFAULT_GAMMA)
        nbins : int (for eq, default 256)
    """
    v = np.asarray(v, dtype=np.float64)
    finite = np.isfinite(v)
    neg = finite & (v < 0.0)
    pos = finite & (v > 0.0)

    t = np.zeros_like(v, dtype=np.float64)
    norm = params.get("norm", "linear")
    nbins = int(params.get("nbins", 256))

    if norm == "eq":
        if np.any(neg):
            t[neg] = _hist_equalize(np.abs(v[neg]), nbins=nbins)
        if np.any(pos):
            t[pos] = _hist_equalize(v[pos], nbins=nbins)
    elif norm == "linear":
        min_neg = float(v[neg].min()) if np.any(neg) else 0.0
        max_pos = float(v[pos].max()) if np.any(pos) else 0.0
        scale = max(abs(min_neg), abs(max_pos))
        scale = 1.0 if (not math.isfinite(scale)) or scale <= 0.0 else scale
        if np.any(neg):
            t[neg] = np.abs(v[neg]) / scale
        if np.any(pos):
            t[pos] = v[pos] / scale
    else:
        raise ValueError(f"Unknown norm {norm!r}. Use 'linear' or 'eq'.")

    t = np.clip(t, 0.0, 1.0)

    gamma = params.get("gamma", DEFAULT_GAMMA)
    gamma = 1.0 if float(gamma) <= 0.0 else float(gamma)
    if gamma != 1.0:
        t = t ** gamma

    return t, finite, neg, pos

def _parse_tri_colors(params: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resolve neg / zero / pos colors from params into RGB255 float arrays.

    params:
        neg_color  : str (default "FFFF00")
        zero_color : str (default "000000")
        pos_color  : str (default "FF0000")

    Returns:
        neg_rgb  : (3,) float64
        zero_rgb : (3,) float64
        pos_rgb  : (3,) float64
    """
    pos_spec  = params.get("pos_color",  "FF0000")
    zero_spec = params.get("zero_color", "000000")
    neg_spec  = params.get("neg_color",  "FFFF00")

    neg_rgb  = np.asarray(parse_color_spec(neg_spec,  (255.0, 255.0, 0.0)), dtype=np.float64)
    zero_rgb = np.asarray(parse_color_spec(zero_spec, (0.0,   0.0,   0.0)), dtype=np.float64)
    pos_rgb  = np.asarray(parse_color_spec(pos_spec,  (255.0, 0.0,   0.0)), dtype=np.float64)

    return neg_rgb, zero_rgb, pos_rgb

# ---------------------------------------------------------------------------
# RGB Colorizers
# ---------------------------------------------------------------------------

def rgb_scheme_mh(lyap: np.ndarray, params: dict) -> np.ndarray:
    """
    Markus & Hess style (RGB interpolation, linear normalization).
    """
    v = np.asarray(lyap, dtype=np.float64)
    if v.ndim != 2:
        raise ValueError("lyap array must be 2D (H, W)")

    H, W = v.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    t, finite, neg, pos = _two_sided_t_and_masks(v, dict(params, norm="linear"))
    if not np.any(finite): return rgb

    neg_rgb, zero_rgb, pos_rgb = _parse_tri_colors(params)

    if np.any(neg):
        tn = t[neg][:, None]
        rgb[neg] = np.rint((1.0 - tn) * zero_rgb + tn * neg_rgb).astype(np.uint8)

    if np.any(pos):
        tp = t[pos][:, None]
        rgb[pos] = np.rint((1.0 - tp) * zero_rgb + tp * pos_rgb).astype(np.uint8)

    return rgb

def rgb_scheme_mh_eq(lyap: np.ndarray, params: dict) -> np.ndarray:

    v = np.asarray(lyap, dtype=np.float64)
    if v.ndim != 2:
        raise ValueError("lyap array must be 2D (H, W)")

    H, W = v.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    # Shared normalization + masks
    t, finite, neg, pos = _two_sided_t_and_masks(v, dict(params, norm="eq"))
    if not np.any(finite): return rgb

    # Resolve endpoint colors once
    neg_rgb, zero_rgb, pos_rgb = _parse_tri_colors(params)

    # λ < 0 : zero -> neg
    if np.any(neg):
        tn = t[neg][:, None]
        rgb[neg] = np.rint((1.0 - tn) * zero_rgb + tn * neg_rgb).astype(np.uint8)

    # λ > 0 : zero -> pos
    if np.any(pos):
        tp = t[pos][:, None]
        rgb[pos] = np.rint((1.0 - tp) * zero_rgb + tp * pos_rgb).astype(np.uint8)

    return rgb

# ---------------------------------------------------------------------------
# composite rgb colorizers
# ---------------------------------------------------------------------------

def rgb_scheme_palette_eq(lyap: np.ndarray, params: dict) -> np.ndarray:
    """
    Colorize using a named palette from COLOR_TRI_STRINGS.
    """
    palette_name = params.get("palette")
    if palette_name is None:
        raise ValueError(
            "params['palette'] must be set to a key in COLOR_TRI_STRINGS"
        )

    if palette_name not in COLOR_TRI_STRINGS:
        raise KeyError(
            f"Unknown palette {palette_name!r}. "
            f"Available: {', '.join(sorted(COLOR_TRI_STRINGS.keys()))}"
        )

    palette_spec = COLOR_TRI_STRINGS[palette_name]
    try:      
        parts = palette_spec.split(":")
        neg_spec, zero_spec, pos_spec = parts[:3]
    except ValueError as exc:
        raise ValueError(
            f"Invalid COLOR_TRI_STRINGS entry for {palette_name!r}: {palette_spec!r} "
            "(expected 'NEG:ZERO:POS')"
        ) from exc

    # Clone params so we don't mutate the caller's dict
    sub_params = dict(params)
    sub_params["neg_color"] = neg_spec
    sub_params["zero_color"] = zero_spec
    sub_params["pos_color"] = pos_spec

    return rgb_scheme_mh_eq(lyap, sub_params)


def rgb_scheme_multipoint(lyap: np.ndarray, params: dict) -> np.ndarray:
    """
    Colorize using N color stops between -1 and +1.

    params:
        palette : str (preferred)
            Name of a palette in COLOR_STRINGS. Its value is a colon-
            separated list "HEX:HEX:HEX:...". All colors are used as
            equidistant stops in [-1, +1].

        color_string : str (optional override)
            If provided, overrides 'palette'. Same format as above.

        gamma : float (optional)
            Gamma applied to normalized coordinate in [0, 1].
            gamma <= 0 is treated as 1 (no gamma).

    Values outside [-1, +1] are clamped before mapping.
    Non-finite entries are left black.
    """
    arr = np.asarray(lyap, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("lyap array must be 2D (H, W)")

    H, W = arr.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    finite = np.isfinite(arr)
    if not np.any(finite):
        return rgb

    # Choose source of color string
    color_string = params.get("color_string")
    if not color_string:
        palette_name = params.get("palette")
        if not palette_name:
            raise ValueError(
                "scheme_multipoint requires either params['palette'] "
                "or params['color_string']"
            )
        try:
            color_string = COLOR_STRINGS[palette_name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown palette {palette_name!r} for scheme_multipoint"
            ) from exc

    # Parse into list of RGB triples
    specs = [s.strip() for s in color_string.split(":") if s.strip()]
    if len(specs) < 2:
        raise ValueError(
            "scheme_multipoint needs at least 2 colors "
            "in color_string / palette"
        )

    colors = []
    for spec in specs:
        r, g, b = parse_color_spec(spec, (0.0, 0.0, 0.0))
        colors.append((r, g, b))

    colors = np.asarray(colors, dtype=np.float64)
    N = colors.shape[0]

    # Map [-1, +1] -> [0, 1], clamp
    vals = arr[finite]
    t = (np.clip(vals, -1.0, 1.0) + 1.0) * 0.5  # in [0, 1]

    gamma = params.get("gamma", 1)
    gamma = 1.0 if gamma <= 0.0 else float(gamma)
    if gamma != 1.0:
        t = t ** gamma

    # N colors => N-1 segments in [0,1]
    segment_float = t * (N - 1)
    idx_low = np.floor(segment_float).astype(np.int64)
    idx_low = np.clip(idx_low, 0, N - 2)
    frac = segment_float - idx_low

    c0 = colors[idx_low]           # (M, 3)
    c1 = colors[idx_low + 1]       # (M, 3)
    frac = frac[:, np.newaxis]     # (M, 1) for broadcasting

    rgb_vals = np.rint((1.0 - frac) * c0 + frac * c1).astype(np.uint8)
    rgb[finite] = rgb_vals

    return rgb

# ---------------------------------------------------------------------------
# HSV Colorizers
# ---------------------------------------------------------------------------

def hsv_scheme_mh_eq(lyap: np.ndarray, params: dict) -> np.ndarray:
    """
    Markus & Hess style with histogram equalization, interpolating in HSV
    (shortest-arc hue), using shared two-sided normalization.

      λ < 0 : zero_color -> neg_color
      λ = 0 : zero_color
      λ > 0 : zero_color -> pos_color

    Normalization:
      - Uses _two_sided_t_and_masks(..., norm="eq")
      - t ∈ [0,1] is hist-eq(|λ|) on neg side and hist-eq(λ) on pos side
      - gamma applied inside the helper

    params:
        norm       : must be "eq" (forced internally)
        gamma      : float (optional, default DEFAULT_GAMMA)
        nbins      : int   (optional, default 256)

        pos_color  : str   (optional, default "FF0000")
        zero_color : str   (optional, default "000000")
        neg_color  : str   (optional, default "FFFF00")
    """
    v = np.asarray(lyap, dtype=np.float64)
    if v.ndim != 2:
        raise ValueError("lyap array must be 2D (H, W)")

    H, W = v.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)

    # Shared normalization + masks
    t, finite, neg, pos = _two_sided_t_and_masks(v, dict(params, norm="eq"))
    if not np.any(finite):
        return out

    # Resolve endpoint colors once
    neg_rgb, zero_rgb, pos_rgb = _parse_tri_colors(params)

    # Convert endpoints to HSV once
    neg_h, neg_s, neg_v = _rgb255_to_hsv01(tuple(neg_rgb))
    zero_h, zero_s, zero_v = _rgb255_to_hsv01(tuple(zero_rgb))
    pos_h, pos_s, pos_v = _rgb255_to_hsv01(tuple(pos_rgb))

    # λ < 0 : zero -> neg
    if np.any(neg):
        tn = t[neg]
        h = _interp_hue_short_arc(zero_h, neg_h, tn)
        s = zero_s + tn * (neg_s - zero_s)
        v_ = zero_v + tn * (neg_v - zero_v)

        rgb = _hsv01_to_rgb255_batch(h, s, v_)
        out[neg] = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)

    # λ > 0 : zero -> pos
    if np.any(pos):
        tp = t[pos]
        h = _interp_hue_short_arc(zero_h, pos_h, tp)
        s = zero_s + tp * (pos_s - zero_s)
        v_ = zero_v + tp * (pos_v - zero_v)

        rgb = _hsv01_to_rgb255_batch(h, s, v_)
        out[pos] = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)

    return out

def hsv_scheme_palette_eq(lyap: np.ndarray, params: dict) -> np.ndarray:
    """
    Equivalent to rgb_scheme_palette_eq, but routes to hsv_scheme_mh_eq
    so interpolation is done in HSV.
    """
    palette_name = params.get("palette")
    if palette_name is None:
        raise ValueError("params['palette'] must be set to a key in COLOR_TRI_STRINGS")

    if palette_name not in COLOR_TRI_STRINGS:
        raise KeyError(
            f"Unknown palette {palette_name!r}. "
            f"Available: {', '.join(sorted(COLOR_TRI_STRINGS.keys()))}"
        )

    palette_spec = COLOR_TRI_STRINGS[palette_name]
    try:
        parts = palette_spec.split(":")
        neg_spec, zero_spec, pos_spec = parts[:3]
    except ValueError as exc:
        raise ValueError(
            f"Invalid COLOR_TRI_STRINGS entry for {palette_name!r}: {palette_spec!r} "
            "(expected 'NEG:ZERO:POS')"
        ) from exc

    sub_params = dict(params)
    sub_params["neg_color"] = neg_spec
    sub_params["zero_color"] = zero_spec
    sub_params["pos_color"] = pos_spec

    return hsv_scheme_mh_eq(lyap, sub_params)


# ---------------------------------------------------------------------------
# Palette field colorizers (per-pixel palette via palette arithmetic)
# ---------------------------------------------------------------------------

def _norm01_percentile(x: np.ndarray, lo: float = 10.0, hi: float = 99.0) -> np.ndarray:
    """
    Robust normalize to [0,1] via percentiles (computed on finite entries).
    """
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros_like(x, dtype=np.float64)

    a = x[finite]
    q0, q1 = np.percentile(a, [float(lo), float(hi)])
    if (not math.isfinite(q0)) or (not math.isfinite(q1)) or (q1 <= q0):
        return np.zeros_like(x, dtype=np.float64)

    y = (x - q0) / (q1 - q0)
    return np.clip(y, 0.0, 1.0)

def _tri_palette_from_name(palette_name: str) -> np.ndarray:
    """
    Return tri-palette as float64 array shape (3,3) in RGB255:
        idx 0 = neg, 1 = zero, 2 = pos
    """
    if palette_name not in COLOR_TRI_STRINGS:
        raise KeyError(
            f"Unknown palette {palette_name!r}. "
            f"Available: {', '.join(sorted(COLOR_TRI_STRINGS.keys()))}"
        )
    palette_spec = COLOR_TRI_STRINGS[palette_name]
    parts = palette_spec.split(":")
    if len(parts) < 3:
        raise ValueError(
            f"Invalid COLOR_TRI_STRINGS entry for {palette_name!r}: {palette_spec!r} "
            "(expected 'NEG:ZERO:POS')"
        )
    neg_spec, zero_spec, pos_spec = parts[:3]
    neg_rgb = parse_color_spec(neg_spec, (0.0, 0.0, 0.0))
    zer_rgb = parse_color_spec(zero_spec, (0.0, 0.0, 0.0))
    pos_rgb = parse_color_spec(pos_spec, (0.0, 0.0, 0.0))
    P = np.asarray([neg_rgb, zer_rgb, pos_rgb], dtype=np.float64)
    return P  # (3,3)

def _blend_tri_palettes_rgb(P0: np.ndarray, P1: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Blend two tri-palettes into a per-pixel tri-palette field.

    P0,P1: (3,3) float64 in RGB255
    w: (H,W) in [0,1]
    returns: (H,W,3,3)
    """
    w = np.asarray(w, dtype=np.float64)
    return (1.0 - w[..., None, None]) * P0[None, None, :, :] + w[..., None, None] * P1[None, None, :, :]

def _tri_colorize_rgb_perpixel(v: np.ndarray, P: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Tri-palette interpolation with per-pixel palette P.

    v: (H,W) in [-1,1]
    P: (H,W,3,3) RGB255, stops [neg, zero, pos]
    t: (H,W) in [0,1] interpolation coordinate (usually abs(v) or equalized abs(v))

    returns uint8 (H,W,3)
    """
    v = np.asarray(v, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    H, W = v.shape
    out = np.zeros((H, W, 3), dtype=np.float64)

    finite = np.isfinite(v)
    if not np.any(finite):
        return out.astype(np.uint8)

    neg = finite & (v < 0.0)
    pos = finite & (v > 0.0)

    # stops
    Cn = P[:, :, 0, :]  # (H,W,3)
    Cz = P[:, :, 1, :]
    Cp = P[:, :, 2, :]

    if np.any(neg):
        tn = t[neg][:, None]
        out[neg] = (1.0 - tn) * Cz[neg] + tn * Cn[neg]

    if np.any(pos):
        tp = t[pos][:, None]
        out[pos] = (1.0 - tp) * Cz[pos] + tp * Cp[pos]

    return np.clip(np.rint(out), 0, 255).astype(np.uint8)

def _tri_colorize_hsv_perpixel(v: np.ndarray, P_rgb: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Same as _tri_colorize_rgb_perpixel, but interpolates in HSV with shortest-arc hue.
    Palette stops are given in RGB255; converted to HSV once per call.
    """
    v = np.asarray(v, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    H, W = v.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)

    finite = np.isfinite(v)
    if not np.any(finite):
        return out

    # Convert palette stops to HSV arrays (H,W,3stops)
    # P_rgb: (H,W,3,3) -> per stop convert rgb->hsv
    # We'll do it stop-by-stop with vectorization; rgb_to_hsv isn't vectorized, so reuse your custom batch HSV->RGB,
    # but RGB->HSV is not batched. We'll approximate by doing it per stop via numpy formulas.
    # Minimal deviation: do RGB blending + RGB->HSV conversion with explicit formulas here.

    Pr = np.clip(P_rgb[..., 0], 0, 255) / 255.0
    Pg = np.clip(P_rgb[..., 1], 0, 255) / 255.0
    Pb = np.clip(P_rgb[..., 2], 0, 255) / 255.0

    cmax = np.maximum(np.maximum(Pr, Pg), Pb)
    cmin = np.minimum(np.minimum(Pr, Pg), Pb)
    delta = cmax - cmin

    # Hue
    Hh = np.zeros_like(cmax)
    mask = delta > 1e-12
    # where max is R
    mr = mask & (cmax == Pr)
    mg = mask & (cmax == Pg)
    mb = mask & (cmax == Pb)

    Hh[mr] = ((Pg[mr] - Pb[mr]) / delta[mr]) % 6.0
    Hh[mg] = ((Pb[mg] - Pr[mg]) / delta[mg]) + 2.0
    Hh[mb] = ((Pr[mb] - Pg[mb]) / delta[mb]) + 4.0
    Hh = (Hh / 6.0) % 1.0

    # Sat
    Ss = np.zeros_like(cmax)
    nonzero = cmax > 1e-12
    Ss[nonzero] = delta[nonzero] / cmax[nonzero]

    # Val
    Vv = cmax

    # Now interpolate in HSV between stops per pixel, separately for neg/pos
    neg = finite & (v < 0.0)
    pos = finite & (v > 0.0)

    # stops in HSV
    # idx 0=neg, 1=zero, 2=pos
    Hn, Hz, Hp = Hh[:, :, 0], Hh[:, :, 1], Hh[:, :, 2]
    Sn, Sz, Sp = Ss[:, :, 0], Ss[:, :, 1], Ss[:, :, 2]
    Vn, Vz, Vp = Vv[:, :, 0], Vv[:, :, 1], Vv[:, :, 2]

    if np.any(neg):
        tn = t[neg]
        h = _interp_hue_short_arc(Hz[neg], Hn[neg], tn)
        s = Sz[neg] + tn * (Sn[neg] - Sz[neg])
        vv = Vz[neg] + tn * (Vn[neg] - Vz[neg])
        rgb = _hsv01_to_rgb255_batch(h, s, vv)
        out[neg] = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)

    if np.any(pos):
        tp = t[pos]
        h = _interp_hue_short_arc(Hz[pos], Hp[pos], tp)
        s = Sz[pos] + tp * (Sp[pos] - Sz[pos])
        vv = Vz[pos] + tp * (Vp[pos] - Vz[pos])
        rgb = _hsv01_to_rgb255_batch(h, s, vv)
        out[pos] = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)

    return out

# Field Features

def _laplacian_5pt(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    p = np.pad(v, 1, mode="edge")
    return (
        p[1:-1, 2:] + p[1:-1, :-2] + p[2:, 1:-1] + p[:-2, 1:-1]
        - 4.0 * p[1:-1, 1:-1]
    )

def _grad_components_scipy(v: np.ndarray, sigma: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(v, dtype=np.float64)
    if sigma and sigma > 0.0:
        v = gaussian_filter(v, sigma=float(sigma), mode="nearest")
    # sobel returns derivative-like responses
    gx = sobel(v, axis=1, mode="nearest")  # x = columns
    gy = sobel(v, axis=0, mode="nearest")  # y = rows
    return gx, gy

def _gradmag_scipy(v: np.ndarray, sigma: float = 0.0) -> np.ndarray:
    gx, gy = _grad_components_scipy(v, sigma=sigma)
    return np.sqrt(gx*gx + gy*gy)

def _feature_grad_dir(v, params:dict) -> np.array:
    theta = float(params.get("theta", 0.0))
    sigma = float(params.get("sigma", 2.0))
    gx, gy = _grad_components_scipy(v, sigma=sigma)
    proj = np.cos(theta) * gx + np.sin(theta) * gy
    proj_abs = np.abs(proj)
    gmag = np.sqrt(gx * gx + gy * gy) + 1e-12
    mode = str(params.get("mode", "align")).lower()
    if mode == "align": f = proj_abs / gmag
    elif mode == "strength": f = proj_abs
    else: raise ValueError(f"Unknown mode {mode!r}. Use 'align', 'strength'.")
    return f

def _feature_struct_tensor_coherence(v: np.ndarray, params: dict) -> np.ndarray:
    # Gaussian pre-blur before gradients
    sigma_pre = float(params.get("sigma_pre", 1.0))
    gx, gy =  _grad_components_scipy(v, sigma=sigma_pre)
    # Tensor smoothing scale
    sigma_tensor = float(params.get("sigma_tensor", 3.0))
    Jxx = gaussian_filter(gx * gx, sigma=sigma_tensor, mode="nearest")
    Jxy = gaussian_filter(gx * gy, sigma=sigma_tensor, mode="nearest")
    Jyy = gaussian_filter(gy * gy, sigma=sigma_tensor, mode="nearest")
    # Eigenvalue-based coherence
    tr = Jxx + Jyy
    det = (Jxx - Jyy)**2 + 4.0 * (Jxy**2)
    s = np.sqrt(np.maximum(det, 0.0))
    l1 = 0.5 * (tr + s)
    l2 = 0.5 * (tr - s)
    eps = 1e-12
    return (l1 - l2) / (l1 + l2 + eps)  # ∈ [0,1]

def _palette_weight_from_feature(v: np.ndarray, params: dict) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)

    w_feature = str(params.get("w_feature", "grad")).lower()

    # ---- compute raw feature f ----
    if w_feature == "grad":
        sigma = float(params.get("sigma", 0.0))
        f = _gradmag_scipy(v, sigma=sigma)

    elif w_feature == "gradx":
        sigma = float(params.get("sigma", 0.0))
        gx, gy = _grad_components_scipy(v,sigma=sigma)
        f = np.abs(gx)

    elif w_feature == "grady":
        sigma = float(params.get("sigma", 0.0))
        gx, gy = _grad_components_scipy(v,sigma=sigma)
        f = np.abs(gy)

    elif w_feature == "grad_dir":  # oriented energy along angle (radians)
        f = _feature_grad_dir(v, params)
 
    elif w_feature == "lap":
        sigma = float(params.get("sigma", 2.0))
        f = np.abs(gaussian_laplace(v, sigma=sigma, mode="nearest"))
        
    elif w_feature == "lvar":
        # local variance via box blurs; radius controlled by iterations
        sigma = float(params.get("sigma", 0.0))
        m = gaussian_filter(v, sigma=sigma, mode="nearest")
        m2 = gaussian_filter(v*v, sigma=sigma, mode="nearest")
        f = np.maximum(m2 - m * m, 0.0)

    elif w_feature == "dog":
        # difference of box blurs (poor man's DoG)
        sigma1 = float(params.get("sigma1", 1.0))
        sigma2 = float(params.get("sigma2", 4.0))
        if sigma2 <= sigma1: sigma2 = sigma1 + 1e-6
        g1 = gaussian_filter(v, sigma1, mode="nearest")
        g2 = gaussian_filter(v, sigma2, mode="nearest")
        b = g1 - g2
        mode = str(params.get("mode", "abs")).lower()
        if mode == "energy":
            energy_sigma = float(params.get("energy_sigma",1.0))
            f = gaussian_filter(b * b, sigma=energy_sigma, mode="nearest")
        else:
            f = np.abs(b)

    elif w_feature == "sign_coh":
        s = np.sign(v)
        sigma = float(params.get("sigma", 0.0))
        sbar = gaussian_filter(s, sigma=sigma, mode="nearest")
        f = np.abs(sbar)

    elif w_feature =="st_coh":
        f = _feature_struct_tensor_coherence(v,params)

    else:
        raise ValueError(
            f"Unknown w_feature {w_feature!r}. Try: "
            "'grad','gradx','grady','grad_dir','lap','lvar','dog','sign_coh'."
        )

    # ---- normalize f -> w in [0,1] ----
    w_lo = float(params.get("w_lo", 10.0))
    w_hi = float(params.get("w_hi", 99.0))
    w = _norm01_percentile(f, lo=w_lo, hi=w_hi)

    # smooth weight (separate from feature blurs)
    w_sigma = float(params.get("w_sigma", 0.0))
    if w_sigma > 0.0:
        w = gaussian_filter(w, sigma=w_sigma, mode="nearest")

    # gamma shaping on weight
    w_gamma = float(params.get("w_gamma", 1.0))
    if w_gamma > 0.0 and w_gamma != 1.0:
        w = np.clip(w, 0.0, 1.0) ** w_gamma

    # strength LAST -> guarantees w_strength=0 => paletteA everywhere
    strength = float(params.get("w_strength", 1.0))
    w = np.clip(w * strength, 0.0, 1.0)

    return w

def rgb_scheme_palette_field(lyap: np.ndarray, params: dict) -> np.ndarray:
    """
    Per-pixel tri-palette blending between two named tri-palettes.

    Key property (by design):
        - If w_strength == 0, Pf == paletteA everywhere, and the result matches
          the single-palette scheme (rgb_scheme_palette_eq) under the same
          normalization settings (norm/gamma/nbins).

    Required params:
        paletteA : str  (name in COLOR_TRI_STRINGS)
        paletteB : str  (name in COLOR_TRI_STRINGS)

    Uses shared normalization:
        t, finite, neg, pos = _two_sided_t_and_masks(v, params)
        where params["norm"] is typically "eq" or "linear".
    """
    v = np.asarray(lyap, dtype=np.float64)
    if v.ndim != 2:
        raise ValueError("lyap array must be 2D (H, W)")

    # palettes
    palA = params.get("paletteA")
    palB = params.get("paletteB")
    if not palA or not palB:
        raise ValueError("rgb_scheme_palette_field requires params['paletteA'] and params['paletteB']")

    P0 = _tri_palette_from_name(str(palA))
    P1 = _tri_palette_from_name(str(palB))

    # weight field (already applies w_strength internally)
    w = _palette_weight_from_feature(v, params)

    # per-pixel palette (H,W,3,3)
    Pf = _blend_tri_palettes_rgb(P0, P1, w)

    # interpolation coordinate t in [0,1] with shared semantics (norm + gamma)
    t, finite, neg, pos = _two_sided_t_and_masks(v, params)

    if not np.any(finite):
        return np.zeros((v.shape[0], v.shape[1], 3), dtype=np.uint8)

    return _tri_colorize_rgb_perpixel(v, Pf, t)




def hsv_scheme_palette_field(lyap: np.ndarray, params: dict) -> np.ndarray:
    """
    Per-pixel tri-palette blending between two named tri-palettes,
    but interpolate colors in HSV (shortest-arc hue).

    Key property:
        w_strength == 0  => Pf == paletteA everywhere, and this reduces to the
        single-palette HSV scheme under the same normalization (norm/gamma/nbins).

    Uses shared normalization:
        t, finite, neg, pos = _two_sided_t_and_masks(v, params)
    """
    v = np.asarray(lyap, dtype=np.float64)
    if v.ndim != 2:
        raise ValueError("lyap array must be 2D (H, W)")

    # palettes
    palA = params.get("paletteA")
    palB = params.get("paletteB")
    if not palA or not palB:
        raise ValueError("hsv_scheme_palette_field requires params['paletteA'] and params['paletteB']")

    P0 = _tri_palette_from_name(str(palA))
    P1 = _tri_palette_from_name(str(palB))

    # weight field (applies w_strength internally)
    w = _palette_weight_from_feature(v, params)

    # per-pixel palette in RGB255
    Pf_rgb = _blend_tri_palettes_rgb(P0, P1, w)

    # shared normalization (handles norm + gamma + nbins)
    t, finite, neg, pos = _two_sided_t_and_masks(v, params)
    if not np.any(finite):
        return np.zeros((v.shape[0], v.shape[1], 3), dtype=np.uint8)

    return _tri_colorize_hsv_perpixel(v, Pf_rgb, t)


