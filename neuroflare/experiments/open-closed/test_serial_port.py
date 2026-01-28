from psychopy import visual, core, event

# Make 'neuroflare' package importable
import os, sys
_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
    
from neuroflare.shared.trigger_box import TriggerBoxManager

# Window settings
WINDOW_SIZE = [1900, 1440]
WINDOW_COLOR = (-0.5, -0.5, -0.5)
WINDOW_UNITS = 'height'

# Create window
win = visual.Window(WINDOW_SIZE, color=WINDOW_COLOR, monitor="testMonitor", units=WINDOW_UNITS)

# Create helper to manage the TriggerBox connection.
tb = TriggerBoxManager(preferred_ports=["COM3"], baudrate=9600)
tb.begin_experiment()


# Main loop
while True:

    win.flip()

    keys = event.getKeys(['space', 'escape'])
    if 'space' in keys:
        tb.start(name='test_pulse', value=5)
        core.wait(0.01)
        tb.stop(name='test_pulse')
        #print(tb.get_status())
        if tb.get_status()["warning_no_connection"]:
            print("Warning: No TriggerBox connection!")
    elif 'escape' in keys:
        print('Experiment cancelled')
        break


win.close()
core.quit()
