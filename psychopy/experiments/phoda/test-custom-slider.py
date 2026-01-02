from psychopy import visual, core, event
from psychopy.visual.slider import Slider

# Create window
win = visual.Window([1600, 1050], color='gray', monitor="testMonitor", units='height')

# Create slider WITHOUT labels (we'll add them manually)
phoda_slider = visual.Slider(
    win=win,
    size=(1.1, 0.08),
    pos=(0, 0.08),
    ticks=range(0, 101),  # All integer values 0-100 are selectable
    labels=None,  # No labels from slider itself
    granularity=1,  # Ensures integer values only
    style='scrollbar',
    flip=False
)

# Adjust marker width - set directly on the marker itself
narrow_width = 0.02  # Width in height units
phoda_slider.marker.size = (narrow_width, phoda_slider.marker.size[1])

# Shrink the line back to normal size (scrollbar adds 20% extra width)
phoda_slider.line.size = (phoda_slider.size[0], phoda_slider.line.size[1])

# Create manual labels positioned below the slider
label_values = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
manual_labels = []
slider_left = phoda_slider.pos[0] - phoda_slider.size[0] / 2
slider_right = phoda_slider.pos[0] + phoda_slider.size[0] / 2
slider_bottom = phoda_slider.pos[1] - phoda_slider.size[1] / 2

for label_val in label_values:
    # Convert label value to x position on slider
    label_ratio = label_val / 100
    x_pos = slider_left + (slider_right - slider_left) * label_ratio
    y_pos = slider_bottom - 0.04  # Position below the slider
    
    label_obj = visual.TextStim(
        win=win,
        text=str(label_val),
        pos=(x_pos, y_pos),
        height=0.02,
        anchorVert='top'
    )
    manual_labels.append(label_obj)

# Instruction text
instruction_text = visual.TextStim(
    win=win,
    text='Rate the impact from 0 (Not impactful) to 100 (Extremely impactful)\n\nClick or drag the slider, then press SPACE to confirm',
    pos=(0, 0.33),
    height=0.03,
    wrapWidth=1.2
)

# Display current rating
phoda_rating_text = visual.TextStim(
    win=win,
    text='',
    pos=(0, -0.25),
    height=0.04,
    bold=True
)

# Label descriptions
phoda_label_left = visual.TextStim(
    win=win,
    text='Not impactful',
    pos=(-0.5, -0.13),
    height=0.025
)

phoda_label_right = visual.TextStim(
    win=win,
    text='Extremely impactful',
    pos=(0.5, -0.13),
    height=0.025
)

# Main loop
while True:
    # Draw everything
    instruction_text.draw()
    phoda_slider.draw()
    for label in manual_labels:
        label.draw()
    phoda_label_left.draw()
    phoda_label_right.draw()
    
    # Update rating display in real-time (shows while dragging)
    if phoda_slider.markerPos is not None:
        phoda_rating_text.text = f'Current rating: {int(phoda_slider.markerPos)}'
    else:
        phoda_rating_text.text = 'No rating selected yet'
    
    
    
    phoda_rating_text.draw()
    win.flip()
    
    # Check for key presses
    keys = event.getKeys(['space', 'escape'])
    if 'space' in keys and phoda_slider.rating is not None:
        print(f'Final rating: {int(phoda_slider.rating)}')
        break
    elif 'escape' in keys:
        print('Experiment cancelled')
        break


# Cleanup
win.close()
core.quit()
