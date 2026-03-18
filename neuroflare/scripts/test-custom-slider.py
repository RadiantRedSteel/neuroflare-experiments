from psychopy import visual, core, event
from psychopy.visual.slider import Slider

# ============================================================
# =================== CONFIGURATION ==========================
# ============================================================

# Window settings
WINDOW_SIZE = [1900, 1440]
WINDOW_COLOR = (-0.5, -0.5, -0.5)
WINDOW_UNITS = 'height'

# PHODA image settings
PS_PHODA_IMG_PATH = 'neuroflare/experiments/phoda/phoda-stimuli/002.jpg'
PS_PHODA_IMG_PX_WIDTH = 283
PS_PHODA_IMG_PX_HEIGHT = 425
PS_PHODA_IMG_HEIGHT_RATIO = 0.62  # Percentage of screen height
PS_PHODA_IMG_POS_Y = 0.13  # Vertical position (0 = center, positive = above center)

# Slider settings
PS_SLIDER_WIDTH_RATIO = 0.8  # Percentage of screen width
PS_SLIDER_HEIGHT = 0.08
PS_SLIDER_MARKER_WIDTH = 0.02
PS_SLIDER_GRANULARITY = 0  # 0 for continuous (VAS), 1 for integers
PS_SLIDER_TICKS = range(0, 101)

# Spacing
PS_IMAGE_SLIDER_GAP = 0.08  # Vertical space between image and slider
PS_SLIDER_LABELS_GAP = 0.03  # Vertical space between slider and numeric labels

# Label settings
PS_LABEL_VALUES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
PS_LABEL_HEIGHT = 0.03
PS_LABEL_FONT = 'Noto Sans'
PS_LABEL_COLOR = 'white'

# Endpoint labels
PS_LABEL_LEFT_TEXT = 'Not at all\nharmful'
PS_LABEL_RIGHT_TEXT = 'Extremely\nharmful'
PS_LABEL_ENDPOINT_HEIGHT = 0.025
PS_LABEL_ENDPOINT_OFFSET = 0.055 # subtracted from label y-pos

# Rating display
PS_RATING_TEXT_HEIGHT = 0.04
PS_RATING_TEXT_DEFAULT = 'Click or drag the slider to rate'

# ============================================================
# =================== HELPER FUNCTIONS =======================
# ============================================================

def calculate_phoda_geometry(win):
    """
    Calculate all geometry for PHODA image and slider layout.
    Returns a dictionary with all positioning and sizing values.
    """
    screen_width, screen_height = win.size
    aspect = screen_width / screen_height
    
    # Image geometry
    img_aspect = PS_PHODA_IMG_PX_WIDTH / PS_PHODA_IMG_PX_HEIGHT
    img_height = PS_PHODA_IMG_HEIGHT_RATIO
    img_width = img_height * img_aspect
    
    # Slider geometry
    slider_width = aspect * PS_SLIDER_WIDTH_RATIO
    slider_height = PS_SLIDER_HEIGHT
    
    # Vertical positioning
    img_pos_y = PS_PHODA_IMG_POS_Y  # Use the defined position for the image
    img_bottom = img_pos_y - img_height / 2
    slider_pos_y = img_bottom - PS_IMAGE_SLIDER_GAP
    label_pos_y = slider_pos_y - slider_height / 2 - PS_SLIDER_LABELS_GAP
    
    # Slider boundaries
    slider_left = -slider_width / 2
    slider_right = slider_width / 2
    
    return {
        'aspect': aspect,
        'img_width': img_width,
        'img_height': img_height,
        'img_pos_y': img_pos_y,
        'slider_width': slider_width,
        'slider_height': slider_height,
        'slider_pos_y': slider_pos_y,
        'slider_left': slider_left,
        'slider_right': slider_right,
        'label_pos_y': label_pos_y
    }

def create_custom_slider(win, geom):
    """
    Create and customize a PHODA slider with narrow marker.
    """
    slider = visual.Slider(
        win=win,
        size=(geom['slider_width'], geom['slider_height']),
        pos=(0, geom['slider_pos_y']),
        ticks=PS_SLIDER_TICKS,
        labels=None,
        granularity=PS_SLIDER_GRANULARITY,
        style='scrollbar',
        flip=False
    )
    
    # Customize marker width
    slider.marker.size = (PS_SLIDER_MARKER_WIDTH, slider.marker.size[1])
    
    # Adjust line size (compensate for scrollbar's 20% extra width)
    slider.line.size = (geom['slider_width'] + PS_SLIDER_MARKER_WIDTH, slider.line.size[1])
    
    return slider

def create_slider_labels(win, geom):
    """
    Create numeric labels positioned along the slider.
    """
    labels = []
    slider_left = geom['slider_left']
    slider_right = geom['slider_right']
    label_pos_y = geom['label_pos_y']
    
    for label_val in PS_LABEL_VALUES:
        label_ratio = label_val / 100
        x_pos = slider_left + (slider_right - slider_left) * label_ratio
        
        label_obj = visual.TextStim(
            win=win,
            text=str(label_val),
            pos=(x_pos, label_pos_y),
            height=PS_LABEL_HEIGHT,
            anchorVert='top',
            color=PS_LABEL_COLOR,
            font=PS_LABEL_FONT
        )
        labels.append(label_obj)
    
    return labels

def create_endpoint_labels(win, geom):
    """
    Create left and right endpoint description labels.
    """
    label_left = visual.TextStim(
        win=win,
        text=PS_LABEL_LEFT_TEXT,
        pos=(geom['slider_left'], geom['label_pos_y'] - PS_LABEL_ENDPOINT_OFFSET),
        height=PS_LABEL_ENDPOINT_HEIGHT,
        anchorVert='top',
        color=PS_LABEL_COLOR,
        font=PS_LABEL_FONT
    )
    
    label_right = visual.TextStim(
        win=win,
        text=PS_LABEL_RIGHT_TEXT,
        pos=(geom['slider_right'], geom['label_pos_y'] - PS_LABEL_ENDPOINT_OFFSET),
        height=PS_LABEL_ENDPOINT_HEIGHT,
        anchorVert='top',
        color=PS_LABEL_COLOR,
        font=PS_LABEL_FONT
    )
    
    return label_left, label_right

def run_phoda_trial():
    """
    Draw all PHODA components on the window.
    """
    phoda_image.draw()
    phoda_slider.draw()
    for label in manual_labels:
        label.draw()
    phoda_label_left.draw()
    phoda_label_right.draw()
    
    # Update rating display in real-time
    if phoda_slider.markerPos is not None:
        if PS_SLIDER_GRANULARITY == 0:
            # Show float for VAS (continuous)
            phoda_rating_text.text = f'Current rating: {phoda_slider.markerPos:.1f}'
        else:
            # Show integer for discrete scale
            phoda_rating_text.text = f'Current rating: {int(phoda_slider.markerPos)}'
    else:
        phoda_rating_text.text = PS_RATING_TEXT_DEFAULT
    
    phoda_rating_text.draw()

# ============================================================
# =================== INITIALIZE EXPERIMENT ==================
# ============================================================

# Create window
win = visual.Window(WINDOW_SIZE, color=WINDOW_COLOR, monitor="testMonitor", units=WINDOW_UNITS)

# Calculate all geometry
ps_geom = calculate_phoda_geometry(win)

# ============================================================
# =================== CREATE COMPONENTS ======================
# ============================================================

# PHODA image
phoda_image = visual.ImageStim(
    win=win,
    image=PS_PHODA_IMG_PATH,
    units='height',
    pos=(0, ps_geom['img_pos_y']),
    size=(ps_geom['img_width'], ps_geom['img_height'])
)

# Custom slider
phoda_slider = create_custom_slider(win, ps_geom)

# Slider labels
manual_labels = create_slider_labels(win, ps_geom)

# Endpoint labels
phoda_label_left, phoda_label_right = create_endpoint_labels(win, ps_geom)

# Rating display text
phoda_rating_text = visual.TextStim(
    win=win,
    text='',
    pos=(0, ps_geom['slider_pos_y']),
    height=PS_RATING_TEXT_HEIGHT,
    bold=True,
    color=PS_LABEL_COLOR,
    font=PS_LABEL_FONT
)

# ============================================================
# =================== MAIN EXPERIMENT LOOP ===================
# ============================================================

# Main loop
while True:
    # Draw all components
    phoda_image.draw()
    phoda_slider.draw()
    for label in manual_labels:
        label.draw()
    phoda_label_left.draw()
    phoda_label_right.draw()
    
    # Update rating display in real-time
    if phoda_slider.markerPos is not None:
        if PS_SLIDER_GRANULARITY == 0:
            # Show float for VAS (continuous)
            phoda_rating_text.text = f'Current rating: {phoda_slider.markerPos:.1f}'
        else:
            # Show integer for discrete scale
            phoda_rating_text.text = f'Current rating: {int(phoda_slider.markerPos)}'
    else:
        phoda_rating_text.text = PS_RATING_TEXT_DEFAULT
    
    phoda_rating_text.draw()
    win.flip()
    
    # Check for responses
    keys = event.getKeys(['space', 'escape'])
    if 'space' in keys and phoda_slider.rating is not None:
        final_rating = phoda_slider.getRating()
        print(f'Final rating: {final_rating}')
        break
    elif 'escape' in keys:
        print('Experiment cancelled')
        break

# ============================================================
# =================== CLEANUP ================================
# ============================================================

win.close()
core.quit()
