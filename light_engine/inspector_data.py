"""Inspector plain-words data (UI1): texture display names, field
explanations, and the Physics section's show-condition.

Qt-free on purpose (F1): the headless tests import this module
without PySide6. window.py imports these names back — this file
is the single source.
"""


# Inspector display names for procedural textures (UI1): plain words
# on screen, the internal name kept as the combo's item data.
TEXTURE_DISPLAY = (
    ("none", "None"),
    ("wood_grain", "Wood grain"),
    ("brushed_metal", "Brushed metal"),
    ("worn_leather", "Worn leather"),
)

# Plain-words explanations under each inspector field (U20). No
# em-dashes, no bare "&" — both are swept by the suite.
FIELD_HELP = {
    "name": "The name shown in the object list.",
    "position": "Where the object sits, in centimeters.",
    "rotation": "How far the object is turned, in degrees.",
    "scale": "How big the object is, compared to its original size. "
             "1 means unchanged.",
    "color": "The base color of the object.",
    "roughness": "Low values look shiny, high values look dull.",
    "metalness": "How much the surface acts like metal. Low is wood "
                 "or plastic, high is steel.",
    "texture": "A pattern on the surface. None leaves the plain color.",
    "smooth": "Smooths the edges between faces. Off shows each flat face.",
    "material_preset": "A starting point. Picking one fills the three "
                       "numbers below.",
    "material_look": "Also change the roughness and metalness to match "
                     "the preset.",
    "material_advanced": "Show every setting for this material. Off "
                         "shows only the ones that change the look most.",
    "density": "How heavy the material is, in grams per cubic "
               "centimeter. Heavy things hit harder.",
    "friction": "How much it grips when sliding. 0 slides like ice, "
                "1 grips like rubber.",
    "restitution": "How bouncy it is. 0 drops dead, 1 bounces back "
                   "all the way.",
    "cast_shadow": "Lets this object throw a shadow onto other things.",
    "receive_shadow": "Lets shadows from other objects fall on this one.",
    "emitter": "This object sprays particles.",
    # Per-material knobs (PO2): one line per knob in PARAM_UI
    # below. Same rules as the rest: plain words, ends with a
    # period, no em-dash, no bare "&".
    "param_ring_spacing": "How far apart the growth rings are, in "
                          "centimeters. Small numbers make fine grain.",
    "param_grain_strength": "How strong the grain shows. 0 is almost "
                            "flat, 1 is full strength.",
    "param_wobble": "How much the rings wander instead of running "
                    "straight. 0 is straight lines.",
    "param_speckle": "How strong the speckles are. 0 is almost flat.",
    "param_blotch": "How strong the soft blotches are. 0 is almost flat.",
    "param_pattern_strength": "How strong the pattern shows. 0 is almost "
                              "flat, 2 is twice as strong.",
    "param_veins": "How dark the veins are. 0 is almost no veins.",
    "param_vein_spacing": "How far apart the veins run, in centimeters.",
    "param_vein_wander": "How much the veins wander instead of running "
                         "straight. 0 is straight lines.",
    "param_layer_spacing": "How far apart the layers are, in centimeters.",
    "param_brick_length": "How long each brick is, in centimeters.",
    "param_course_height": "How tall each row of bricks is, in "
                           "centimeters.",
    "param_mortar": "How wide the gaps between bricks are. 0 is almost "
                    "no gap.",
    "param_variation": "How much the bricks differ in tone. 0 makes "
                       "them all alike.",
    "param_pits": "How dark the little pits are. 0 is almost no pits.",
    "param_stains": "How dark the stains are. 0 is almost no stains.",
    "param_fine_grain": "How strong the fine grain is. 0 is almost flat.",
    "param_streaks": "How strong the brushed streaks are. 0 is almost "
                     "flat.",
    "param_rust": "How much rust shows. 0 is clean metal.",
    "param_rust_patches": "How many rust patches gather. Fewer means "
                          "bigger patches.",
    "param_crystal_size": "How wide the crystals are, in centimeters.",
    "param_thread_spacing": "How far apart the threads are, in "
                            "centimeters.",
    "param_loop_size": "How wide the loops are, in centimeters.",
    "param_crease": "How strong the creases are. 0 is almost flat.",
    "param_blades": "How strong the grass blades read. 0 is almost flat.",
    "param_mow_stripes": "How many mow stripes run across. 2 is one "
                         "light band and one dark.",
    "param_clump_size": "How wide the clumps are, in centimeters.",
    "param_ripple_spacing": "How far apart the ripples are, in "
                            "centimeters.",
    # Layout rows (PO4): one line for the picker, one per size
    # label in LAYOUT_UI below. Same rules as the rest: plain
    # words, ends with a period, no em-dash, no bare "&".
    "layout": "How the pieces are arranged. Solid leaves the "
              "pattern as it is.",
    "layout_board_width": "How wide each board is, in centimeters.",
    "layout_board_length": "How long each board is, in centimeters.",
    "layout_strip_width": "How wide each strip is, in centimeters.",
    "layout_block_size": "How wide each block of strips is, in "
                         "centimeters.",
    "layout_square_size": "How wide each square is, in centimeters.",
    "layout_slab_width": "How wide each slab is, in centimeters.",
    "layout_slab_length": "How long each slab is, in centimeters.",
    # Wear rows (PO5): one line per wear type in WEAR_UI below.
    # Same rules as the rest: plain words, ends with a period,
    # no em-dash, no bare "&".
    "wear_dirt": "How dirty the surface is. 0 is clean, 100 is filthy.",
    "wear_grime": "How grimy the surface is. 0 is clean, 100 is caked.",
    "wear_edge_wear": "How worn the exposed edges look. 0 is sharp "
                      "and new.",
    "wear_rust": "How much rust shows. 0 is clean metal.",
    "wear_moss": "How much moss grows on it. 0 is none.",
    "wear_wetness": "How wet the surface is. 0 is dry, 100 is soaked.",
    "wear_cracks": "How far apart the cracks run, in centimeters. "
                   "0 means no cracks.",
}


# Per-material knobs (PO2): for each style, the rows the Material
# section shows, in order. Each knob: the registry param it edits,
# its plain-words label, its kind ("value" — the spin shows the
# param itself; "spacing_cm" — the spin shows a spacing in
# centimeters and the param is the count per tile, converted
# through textures.spacing_to_count / count_to_spacing), the spin
# range + step + decimals, its FIELD_HELP key, and "simple" (PO3):
# True when the knob is one of the 2-3 that change the look most
# and shows in Simple mode, False when it only shows in Advanced.
# The simple set is the defining knobs — the spacing (the
# pattern's scale, always simple where a style has one) and the
# strength — while the refinements stay Advanced-only: ring
# wobble (cathedral oak, burl), vein wander (Carrara marble),
# and tone variation (stretcher brick). Styles with one or two
# knobs show them all in Simple — there is nothing to hide.
# Every param in the registry has exactly one knob here — the
# suite pins that, so a style added without knobs fails loudly
# instead of silently hiding its parameters. Ranges keep the
# pattern in its sane band (see textures.py): a knob at either
# end still reads as the material, never as black or white noise.
# Spacing knobs bottom out where the 128-pixel bake can still
# draw the feature — the finest wood spacing (0.4 cm, 250 rings)
# is past Nyquist and reads as fine grain, which the spec's
# done-when names explicitly, so it stays.
PARAM_UI = {
    "wood_cathedral": (
        {"param": "rings", "label": "Ring spacing",
         "kind": "spacing_cm", "low": 0.4, "high": 25.0,
         "step": 0.1, "decimals": 1, "help": "param_ring_spacing", "simple": True},
        {"param": "strength", "label": "Grain strength",
         "kind": "value", "low": 0.0, "high": 1.0,
         "step": 0.05, "decimals": 2, "help": "param_grain_strength", "simple": True},
        {"param": "wobble", "label": "Ring wobble",
         "kind": "value", "low": 0.0, "high": 10.0,
         "step": 0.5, "decimals": 1, "help": "param_wobble", "simple": False},
    ),
    "wood_knotty_pine": (
        {"param": "rings", "label": "Ring spacing",
         "kind": "spacing_cm", "low": 0.4, "high": 25.0,
         "step": 0.1, "decimals": 1, "help": "param_ring_spacing", "simple": True},
        {"param": "strength", "label": "Grain strength",
         "kind": "value", "low": 0.0, "high": 1.0,
         "step": 0.05, "decimals": 2, "help": "param_grain_strength", "simple": True},
    ),
    "wood_burl": (
        {"param": "rings", "label": "Ring spacing",
         "kind": "spacing_cm", "low": 0.4, "high": 25.0,
         "step": 0.1, "decimals": 1, "help": "param_ring_spacing", "simple": True},
        {"param": "strength", "label": "Grain strength",
         "kind": "value", "low": 0.0, "high": 1.0,
         "step": 0.05, "decimals": 2, "help": "param_grain_strength", "simple": True},
        {"param": "wobble", "label": "Ring wobble",
         "kind": "value", "low": 0.0, "high": 16.0,
         "step": 0.5, "decimals": 1, "help": "param_wobble", "simple": False},
    ),
    "wood_weathered": (
        {"param": "strength", "label": "Grain strength",
         "kind": "value", "low": 0.0, "high": 1.0,
         "step": 0.05, "decimals": 2, "help": "param_grain_strength", "simple": True},
    ),
    "stone_granite": (
        {"param": "speckle", "label": "Speckle",
         "kind": "value", "low": 0.0, "high": 0.5,
         "step": 0.02, "decimals": 2, "help": "param_speckle", "simple": True},
        {"param": "blotch", "label": "Blotch",
         "kind": "value", "low": 0.0, "high": 0.3,
         "step": 0.02, "decimals": 2, "help": "param_blotch", "simple": True},
    ),
    "stone_limestone": (
        {"param": "strength", "label": "Pattern strength",
         "kind": "value", "low": 0.0, "high": 2.0,
         "step": 0.1, "decimals": 1, "help": "param_pattern_strength", "simple": True},
    ),
    "stone_marble_carrara": (
        {"param": "veins", "label": "Veins",
         "kind": "value", "low": 0.0, "high": 0.8,
         "step": 0.05, "decimals": 2, "help": "param_veins", "simple": True},
        {"param": "bands", "label": "Vein spacing",
         "kind": "spacing_cm", "low": 5.0, "high": 100.0,
         "step": 1.0, "decimals": 0, "help": "param_vein_spacing", "simple": True},
        {"param": "warp", "label": "Vein wander",
         "kind": "value", "low": 0.0, "high": 12.0,
         "step": 0.5, "decimals": 1, "help": "param_vein_wander", "simple": False},
    ),
    "stone_slate": (
        {"param": "layers", "label": "Layer spacing",
         "kind": "spacing_cm", "low": 2.0, "high": 25.0,
         "step": 0.5, "decimals": 1, "help": "param_layer_spacing", "simple": True},
        {"param": "strength", "label": "Pattern strength",
         "kind": "value", "low": 0.0, "high": 1.0,
         "step": 0.05, "decimals": 2, "help": "param_pattern_strength", "simple": True},
    ),
    "brick_stretcher": (
        {"param": "cols", "label": "Brick length",
         "kind": "spacing_cm", "low": 10.0, "high": 50.0,
         "step": 1.0, "decimals": 0, "help": "param_brick_length", "simple": True},
        {"param": "rows", "label": "Course height",
         "kind": "spacing_cm", "low": 5.0, "high": 25.0,
         "step": 0.5, "decimals": 1, "help": "param_course_height", "simple": True},
        {"param": "mortar", "label": "Mortar width",
         "kind": "value", "low": 0.0, "high": 0.2,
         "step": 0.01, "decimals": 2, "help": "param_mortar", "simple": True},
        {"param": "variation", "label": "Tone variation",
         "kind": "value", "low": 0.0, "high": 0.4,
         "step": 0.05, "decimals": 2, "help": "param_variation", "simple": False},
    ),
    "concrete_raw": (
        {"param": "strength", "label": "Pattern strength",
         "kind": "value", "low": 0.0, "high": 2.0,
         "step": 0.1, "decimals": 1, "help": "param_pattern_strength", "simple": True},
        {"param": "pits", "label": "Pits",
         "kind": "value", "low": 0.0, "high": 0.2,
         "step": 0.01, "decimals": 2, "help": "param_pits", "simple": True},
    ),
    "concrete_aged": (
        {"param": "stains", "label": "Stains",
         "kind": "value", "low": 0.0, "high": 0.6,
         "step": 0.05, "decimals": 2, "help": "param_stains", "simple": True},
        {"param": "strength", "label": "Fine grain",
         "kind": "value", "low": 0.0, "high": 0.2,
         "step": 0.01, "decimals": 2, "help": "param_fine_grain", "simple": True},
    ),
    "metal_brushed": (
        {"param": "streak", "label": "Streaks",
         "kind": "value", "low": 0.0, "high": 0.3,
         "step": 0.02, "decimals": 2, "help": "param_streaks", "simple": True},
        {"param": "fine", "label": "Fine grain",
         "kind": "value", "low": 0.0, "high": 0.2,
         "step": 0.01, "decimals": 2, "help": "param_fine_grain", "simple": True},
    ),
    "metal_rusted": (
        {"param": "rust", "label": "Rust",
         "kind": "value", "low": 0.0, "high": 0.8,
         "step": 0.05, "decimals": 2, "help": "param_rust", "simple": True},
        {"param": "patches", "label": "Rust patches",
         "kind": "value", "low": 1.0, "high": 10.0,
         "step": 1.0, "decimals": 0, "help": "param_rust_patches", "simple": True},
    ),
    "metal_galvanised": (
        {"param": "spangle", "label": "Crystal size",
         "kind": "spacing_cm", "low": 5.0, "high": 50.0,
         "step": 1.0, "decimals": 0, "help": "param_crystal_size", "simple": True},
        {"param": "strength", "label": "Pattern strength",
         "kind": "value", "low": 0.0, "high": 2.0,
         "step": 0.1, "decimals": 1, "help": "param_pattern_strength", "simple": True},
    ),
    "fabric_canvas": (
        {"param": "threads", "label": "Thread spacing",
         "kind": "spacing_cm", "low": 2.0, "high": 20.0,
         "step": 0.5, "decimals": 1, "help": "param_thread_spacing", "simple": True},
        {"param": "strength", "label": "Pattern strength",
         "kind": "value", "low": 0.0, "high": 2.0,
         "step": 0.1, "decimals": 1, "help": "param_pattern_strength", "simple": True},
    ),
    "fabric_wool_loop": (
        {"param": "loops", "label": "Loop size",
         "kind": "spacing_cm", "low": 2.0, "high": 25.0,
         "step": 0.5, "decimals": 1, "help": "param_loop_size", "simple": True},
        {"param": "strength", "label": "Pattern strength",
         "kind": "value", "low": 0.0, "high": 2.0,
         "step": 0.1, "decimals": 1, "help": "param_pattern_strength", "simple": True},
    ),
    "fabric_leather": (
        {"param": "blotch", "label": "Blotch",
         "kind": "value", "low": 0.0, "high": 0.4,
         "step": 0.02, "decimals": 2, "help": "param_blotch", "simple": True},
        {"param": "crease", "label": "Crease",
         "kind": "value", "low": 0.0, "high": 0.3,
         "step": 0.02, "decimals": 2, "help": "param_crease", "simple": True},
    ),
    "ground_lawn": (
        {"param": "blades", "label": "Blades",
         "kind": "value", "low": 0.0, "high": 0.4,
         "step": 0.02, "decimals": 2, "help": "param_blades", "simple": True},
        {"param": "stripes", "label": "Mow stripes",
         "kind": "value", "low": 2.0, "high": 10.0,
         "step": 1.0, "decimals": 0, "help": "param_mow_stripes", "simple": True},
    ),
    "ground_soil": (
        {"param": "strength", "label": "Pattern strength",
         "kind": "value", "low": 0.0, "high": 2.0,
         "step": 0.1, "decimals": 1, "help": "param_pattern_strength", "simple": True},
        {"param": "clumps", "label": "Clump size",
         "kind": "spacing_cm", "low": 5.0, "high": 50.0,
         "step": 1.0, "decimals": 0, "help": "param_clump_size", "simple": True},
    ),
    "ground_sand": (
        {"param": "ripples", "label": "Ripple spacing",
         "kind": "spacing_cm", "low": 2.0, "high": 20.0,
         "step": 0.5, "decimals": 1, "help": "param_ripple_spacing", "simple": True},
        {"param": "strength", "label": "Pattern strength",
         "kind": "value", "low": 0.0, "high": 2.0,
         "step": 0.1, "decimals": 1, "help": "param_pattern_strength", "simple": True},
    ),
}


def param_knobs_for(style, advanced):
    """The knobs the Material section shows for a style (PO3):
    every knob in Advanced, only the simple ones in Simple. A
    knob without a "simple" flag counts as simple, so a new knob
    shows instead of vanishing — but the suite pins every knob
    carrying the flag explicitly, so that default is a backstop,
    not a way to skip the choice. PO4's layout rows are
    Advanced-only too, but they are not param knobs — they slot
    into the same container in window.py, under the same toggle
    (see layout_rows_for); PO5's wear rows follow the same
    pattern (see wear_rows_for)."""
    knobs = PARAM_UI.get(style, ())
    if advanced:
        return knobs
    return tuple(k for k in knobs if k.get("simple", True))


# Layout size rows (PO4): for each layout that has sizes, the rows
# the Material section shows under the layout picker, in order.
# Each row: the layout_size key it edits ("board_width" or
# "board_length" — the keys are shared across layouts, the labels
# are per-layout), its plain-words label, the spin range + step +
# decimals, and its FIELD_HELP key. The picker relabels the same
# two keys per layout (parquet's length is its block size, a
# slab's width and length are the slab's sides), so sizes persist
# across layout switches. Solid and the brick bonds have no entry:
# solid has no pieces, and brick sizes itself from the brick
# length / course height knobs. Ranges keep the pieces drawable:
# the narrowest strip (3 cm) is still several pixels wide on a
# 1 m face, and the longest board (120 cm) still shows joints on
# a big floor. Defaults (textures.LAYOUT_DEFAULTS) sit inside
# every range — the suite pins that.
LAYOUT_UI = {
    "boards": (
        {"key": "board_width", "label": "Board width",
         "low": 5.0, "high": 25.0, "step": 0.5, "decimals": 1,
         "help": "layout_board_width"},
        {"key": "board_length", "label": "Board length",
         "low": 30.0, "high": 120.0, "step": 1.0, "decimals": 0,
         "help": "layout_board_length"},
    ),
    "strip": (
        {"key": "board_width", "label": "Strip width",
         "low": 3.0, "high": 12.0, "step": 0.5, "decimals": 1,
         "help": "layout_strip_width"},
        {"key": "board_length", "label": "Board length",
         "low": 30.0, "high": 120.0, "step": 1.0, "decimals": 0,
         "help": "layout_board_length"},
    ),
    "parquet": (
        {"key": "board_width", "label": "Strip width",
         "low": 4.0, "high": 15.0, "step": 0.5, "decimals": 1,
         "help": "layout_strip_width"},
        {"key": "board_length", "label": "Block size",
         "low": 20.0, "high": 60.0, "step": 1.0, "decimals": 0,
         "help": "layout_block_size"},
    ),
    "chevron": (
        {"key": "board_width", "label": "Board width",
         "low": 5.0, "high": 25.0, "step": 0.5, "decimals": 1,
         "help": "layout_board_width"},
        {"key": "board_length", "label": "Board length",
         "low": 30.0, "high": 120.0, "step": 1.0, "decimals": 0,
         "help": "layout_board_length"},
    ),
    "herringbone": (
        {"key": "board_width", "label": "Board width",
         "low": 5.0, "high": 25.0, "step": 0.5, "decimals": 1,
         "help": "layout_board_width"},
        {"key": "board_length", "label": "Board length",
         "low": 30.0, "high": 120.0, "step": 1.0, "decimals": 0,
         "help": "layout_board_length"},
    ),
    "basket_weave": (
        {"key": "board_width", "label": "Board width",
         "low": 5.0, "high": 20.0, "step": 0.5, "decimals": 1,
         "help": "layout_board_width"},
        {"key": "board_length", "label": "Square size",
         "low": 20.0, "high": 60.0, "step": 1.0, "decimals": 0,
         "help": "layout_square_size"},
    ),
    "slab": (
        {"key": "board_width", "label": "Slab width",
         "low": 20.0, "high": 120.0, "step": 1.0, "decimals": 0,
         "help": "layout_slab_width"},
        {"key": "board_length", "label": "Slab length",
         "low": 20.0, "high": 120.0, "step": 1.0, "decimals": 0,
         "help": "layout_slab_length"},
    ),
}


def layout_rows_for(layout):
    """The size rows the Material section shows for a layout (PO4):
    the LAYOUT_UI entry, or no rows for solid and the brick bonds."""
    return LAYOUT_UI.get(layout, ())


# Wear rows (PO5): for each wear type, the row the Material
# section shows under the "Life" label, Advanced-only, after the
# layout rows. Each row: the wear key it edits, its plain-words
# label, its kind ("percent" — a 0-100 amount; "cracks_cm" — a
# fissure spacing in cm, 0 meaning no cracks), the spin range +
# step + decimals, and its FIELD_HELP key. Which rows a style
# shows comes from the registry (textures.wear_for_style), so
# this table carries one row per wear type, not one per style.
# The values are stored in the material's "wear" dict as
# overrides only — 0 stores nothing, so old scenes (no "wear"
# key) load clean and dry.
WEAR_UI = {
    "dirt": {"key": "dirt", "label": "Dirt", "kind": "percent",
             "low": 0.0, "high": 100.0, "step": 5.0, "decimals": 0,
             "help": "wear_dirt"},
    "grime": {"key": "grime", "label": "Grime", "kind": "percent",
              "low": 0.0, "high": 100.0, "step": 5.0, "decimals": 0,
              "help": "wear_grime"},
    "edge_wear": {"key": "edge_wear", "label": "Edge wear",
                  "kind": "percent",
                  "low": 0.0, "high": 100.0, "step": 5.0,
                  "decimals": 0, "help": "wear_edge_wear"},
    "rust": {"key": "rust", "label": "Rust", "kind": "percent",
             "low": 0.0, "high": 100.0, "step": 5.0, "decimals": 0,
             "help": "wear_rust"},
    "moss": {"key": "moss", "label": "Moss", "kind": "percent",
             "low": 0.0, "high": 100.0, "step": 5.0, "decimals": 0,
             "help": "wear_moss"},
    "wetness": {"key": "wetness", "label": "Wetness",
                "kind": "percent",
                "low": 0.0, "high": 100.0, "step": 5.0,
                "decimals": 0, "help": "wear_wetness"},
    "cracks": {"key": "cracks", "label": "Crack spacing",
               "kind": "cracks_cm",
               "low": 0.0, "high": 40.0, "step": 1.0, "decimals": 0,
               "help": "wear_cracks"},
}


def wear_rows_for(style):
    """The wear rows the Material section shows for a style (PO5):
    one row per wear type the style takes, in registry order.
    A style-less material takes the generic set, so the rows show
    whenever Advanced is on."""
    from .textures import wear_for_style
    return tuple(WEAR_UI[key] for key in wear_for_style(style))


def _physics_enabled(obj):
    """Whether the Physics section applies to an object (UI1): the
    object carries a physics dict with enabled true. Nothing sets
    that yet — T20 owns the physics controls that land in the
    section. None (no selection) is never physics-enabled."""
    if not isinstance(obj, dict):
        return False
    physics = obj.get("physics")
    return isinstance(physics, dict) and bool(physics.get("enabled", False))
