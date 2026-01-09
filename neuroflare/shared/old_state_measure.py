# Begin Experiment
# ------------------------------------------------------------
# code_state_measure_setup
# Global setup logic for the stateMeasure routine.
#
# Handles scaling rules, slider configuration helpers, and
# shared variables needed to dynamically construct rating
# scales based on loop parameters.
#
# The stateMeasure routine depends on this setup code to
# size components correctly and interpret SAM/text settings.
# ------------------------------------------------------------

# --- Helper functions for dynamic slider configuration ---
def parse_tick_values(raw):
    # If PsychoPy already parsed it into a list, just return it
    if isinstance(raw, list):
        return raw
    # Otherwise assume it's a comma-separated string
    return [int(x.strip()) for x in raw.split(',')]

# ===============================================================
# ============== STATE-MEASURE SPECIFIC GEOMETRY ================
# ===============================================================

def sm_compute_tick_positions(tick_values, rating_type):
    """
    Compute slider size, slider position, and min/max label positions
    for both SAM and Generic rating types.
    """

    if rating_type == "SAM":
        # SAM uses the precomputed geometry
        slider_size = (sm_sam_comp_slider_width, sm_sam_comp_slider_size_height)
        slider_pos  = (0, sm_sam_comp_slider_pos_y)

        # Tick positions already computed globally
        tick_positions = sm_sam_tick_positions

        label_min_pos = (tick_positions[0],  sm_sam_comp_label_pos_y)
        label_max_pos = (tick_positions[-1], sm_sam_comp_label_pos_y)

    else:
        # Generic slider: evenly spaced ticks across the slider width
        slider_size = (sm_gen_comp_slider_size_width, sm_gen_comp_slider_size_height)
        slider_pos  = (0, sm_gen_comp_slider_pos_y)

        # Compute generic tick positions dynamically
        num_ticks = len(tick_values)
        half_width = sm_gen_comp_slider_size_width / 2

        # Even spacing across the width
        tick_positions = [
            -half_width + (i / (num_ticks - 1)) * sm_gen_comp_slider_size_width
            for i in range(num_ticks)
        ]

        # Labels sit below the slider
        label_min_pos = (tick_positions[0],  sm_gen_comp_label_pos_y)
        label_max_pos = (tick_positions[-1], sm_gen_comp_label_pos_y)

    return {
        "slider_size": slider_size,
        "slider_pos": slider_pos,
        "tick_positions": tick_positions,
        "label_min_pos": label_min_pos,
        "label_max_pos": label_max_pos
    }

# ===============================================================
# =================== GEN SPECIFIC GEOMETRY =====================
# ===============================================================

# --- Generic Slider Scaling ---
# Full-width slider across screen
# Generic slider (used when rating_type != "SAM")

# Slider layout (height units)
sm_gen_comp_slider_size_width  = aspect * 0.6  # 60% of screen width
sm_gen_comp_slider_size_height = 0.1           # slider thickness/height
sm_gen_comp_slider_pos_y       = 0             # vertical position of the slider center

# Label Min/Max layout
sm_gen_comp_label_pos_y = sm_gen_comp_slider_pos_y - 0.2

# ===============================================================
# =================== SAM SPECIFIC GEOMETRY =====================
# ===============================================================

# --- SAM image geometry (pixels) ---
sm_sam_img_px_width  = 1168
sm_sam_img_px_height = 231
sm_sam_img_aspect    = sm_sam_img_px_width / sm_sam_img_px_height  # ≈ 5.056

# Pictogram/gap geometry (pixels)
# Note: there are 5 pictograms and 4 gaps spanning the full image width
sm_sam_pictogram_px_width     = 215
sm_sam_pictogram_gap_px_width = 23

# --- Layout decisions (height units) ---
sm_sam_comp_img_size_height = 0.25                                             # 25% of screen height
sm_sam_comp_img_size_width  = sm_sam_comp_img_size_height * sm_sam_img_aspect  # scaled width based on aspect
sm_sam_comp_img_pos_y       = 0.1                                              # vertical position of the image

# Message Layout
sm_sam_comp_message_pos_y = sm_sam_comp_img_pos_y + 0.2

# Slider layout (height units)
sm_sam_comp_slider_pos_y       = -0.09      # vertical position of the slider center
sm_sam_comp_slider_size_height = 0.06       # slider thickness/height
sm_sam_slider_px_width    = sm_sam_img_px_width - 215
sm_sam_comp_slider_width  = sm_sam_comp_img_size_width * (sm_sam_slider_px_width / sm_sam_img_px_width)

# Scale factor: px -> 'height' units within the image width
sm_sam_scale_factor = sm_sam_comp_img_size_width / sm_sam_img_px_width

sm_sam_pictogram_w  = sm_sam_pictogram_px_width * sm_sam_scale_factor
sm_sam_gap_w        = sm_sam_pictogram_gap_px_width * sm_sam_scale_factor

# --- Compute 9 tick x-positions aligned to pictograms ---
sm_sam_tick_positions = []
sm_sam_tick_x = -sm_sam_comp_img_size_width / 2 + sm_sam_pictogram_w / 2

for i in range(9):
    sm_sam_tick_positions.append(sm_sam_tick_x)
    if i % 2 == 0:
        sm_sam_tick_x += sm_sam_pictogram_w / 2 + sm_sam_gap_w / 2
    else:
        sm_sam_tick_x += sm_sam_gap_w / 2 + sm_sam_pictogram_w / 2

# Label Min/Max layout
sm_sam_comp_label_pos_y = sm_sam_comp_slider_pos_y - 0.15


# code_sm_helper
# Begin Routine
allowContinue = False
hello_there = 23

# Parse tick values from the condition file
tick_values_parsed = parse_tick_values(tick_values)

try:
    logging.debug(f"Starting state-measure in loop: {currentLoop.name}")
    logging.debug(f"State-measure current row: {rating_category}, {rating_type}")
    logging.debug(f"State-measure tick values: {tick_values_parsed}")
except:
    logging.error("Error printing current state measure row.")

# Compute geometry based on rating type
sm_geometry = sm_compute_tick_positions(tick_values_parsed, rating_type)

sm_comp_slider_size = sm_geometry["slider_size"]
sm_comp_slider_pos  = sm_geometry["slider_pos"]
sm_comp_label_min_pos = sm_geometry["label_min_pos"]
sm_comp_label_max_pos = sm_geometry["label_max_pos"]

t_sm_label_min = visual.TextStim(
    win=win,
    text=rating_min_label,
    pos=(sm_comp_label_min_pos[0], sm_comp_label_min_pos[1]),
    wrapWidth=aspect20,
    anchorVert='top',
    height=0.05,
    color='white',
    font='Noto Sans'
)

t_sm_label_max = visual.TextStim(
    win=win,
    text=rating_max_label,
    pos=(sm_comp_label_max_pos[0], sm_comp_label_max_pos[1]),
    wrapWidth=aspect20,
    anchorVert='top',
    height=0.05,
    color='white',
    font='Noto Sans'
)

#t_label_min.setPos(sm_comp_label_min_pos)
#t_label_max.setPos(sm_comp_label_max_pos)

# --- Create a fresh slider for this trial ---
sm_slider = visual.Slider(
    win=win,
    ticks=tick_values_parsed,
    labels=tick_values_parsed,
    granularity=granularity,
    style=style.lower(),
    pos=sm_comp_slider_pos,
    size=sm_comp_slider_size,
    labelHeight=0.05,
    colorSpace='rgb',
    markerColor='Red',
    lineColor='White',
    labelColor='LightGray',
    font='Noto Sans'
)

# --- SAM image logic ---
if rating_type == 'SAM':
    image_SAM.setImage(picture_path)
    image_SAM.setAutoDraw(True)
else:
    image_SAM.setAutoDraw(False)

# Each Frame
# Draw state measure components
sm_slider.draw()
t_sm_label_min.draw()
t_sm_label_max.draw()

# If rated, allow spacebar to end the routine
if sm_slider.getRating() is not None:
    allowContinue = True

# End Routine
currentLoop.addData('rating', sm_slider.getRating())
currentLoop.addData('rating_rt', sm_slider.getRT())
currentLoop.addData('hello-there', hello_there)
try:
    logging.data(f"State-measure rating: {rating_category}, {sm_slider.getRating()}")
except:
    logging.error("Error printing state-measure rating")
