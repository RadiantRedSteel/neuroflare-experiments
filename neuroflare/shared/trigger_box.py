from __future__ import annotations
import time
from typing import Optional
from psychopy import logging

try:
    import serial
    import serial.tools.list_ports
except Exception:  # pragma: no cover - environment dependent
    serial = None  # type: ignore

class TriggerBoxManager:
    """
    Manage a Brain Products TriggerBox over a virtual serial port.
    Specfically, the TriggerBox appears as a USB virtual COM port.

    Provides connection discovery, reconnection, and high/low trigger management
    (0-255) without crashing when the COM port is missing. Designed to be instantiated 
    once in a "Begin Experiment" code component and called from per-routine code.

    ===========================================================================
    USAGE IN PSYCHOPY BUILDER
    ===========================================================================
    
    1. SETUP CODE COMPONENT (place at the very start of your experiment):
       
        Create a Code Component in your setup routine (e.g., "experiment_setup") and add this to "Begin Experiment":
       
            from neuroflare.shared.trigger_box import TriggerBoxManager
            tb = TriggerBoxManager()
            tb.begin_experiment() 
        
        Depending on your environment, you may need a wrapper to make the 'neuroflare'
        package importable, similar to this:

            import os, sys
            _THIS_DIR = os.path.dirname(__file__)
            _REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..', '..'))
            if _REPO_ROOT not in sys.path:
                sys.path.insert(0, _REPO_ROOT)
        
        Then, in "End Routine" of the same component, add:

            tb.send_event(value=1, name="experiment_start")
            tb.send_idle() # immediately return to baseline

        This helps to clear out any residual triggers, and also acts as both a marker and connection insurance.
       
        Lastly, add this to "End Experiment":
       
            tb.end_experiment()
    
    2. TRIGGER CODE COMPONENTS (place BELOW the component you're triggering):
       
        Important: Component order matters! Each frame, Builder executes components
        from top to bottom. Your trigger code must come AFTER the component it
        synchronizes with, so both have updated status values.
       
        You can create multiple trigger components with different names and values.
        Values range from 0-255 (8-bit). I recommended staying consistent.
       
        In "Each Frame":
       
            if cross_fixation.status == STARTED:
                tb.send_event(value=10, name="fixation")
            if cross_fixation.status == STOPPED:
                tb.send_idle()

        The trigger methods take note of previous successfully sent values.
        Therefore, if you're worried about if that send_idle() never went through,
        you can call it again without worry - put it in the end routine of other routines.

        Because the TriggerBox can only output one value at a time, overlapping events
        (e.g., image onset and rating clicks) require careful condition logic to avoid overwriting 
        triggers or producing one-frame idle gaps.

        For example, you might want to mark the duration of an image, but also flag when a rating happens.
        With correct condition checks, you can meld them together. However, this could lead
        to the rating trigger lasting one frame only, which may be too brief for your recording system.

        If you need a trigger to last longer than one frame (e.g., PHODA rating clicks),
        implement a short-duration pulse in your experiment logic using a timer. For example:

        In "Begin Routine"

            from psychopy import core # if not already imported
            rating_pulse_active = False
            rating_pulse_sent = False
            rating_timer = core.Clock()

        In "Each Frame"

            # 1. Rating pulse takes priority
            if rating_pulse_active:
                # End pulse?
                if rating_timer.getTime() > 1.0:
                    rating_pulse_active = False
                    # Immediately restore correct image state
                    if image_phoda.status == STARTED:
                        tb.send_event(40, "image")
                    else:
                        tb.send_idle()
                # If pulse still active, do nothing else this frame
                else:
                    return  # Skip image logic entirely

            # 2. Image logic (only runs when no rating pulse is active)
            if image_phoda.status == STARTED:
                tb.send_event(40, "image")
            elif image_phoda.status == STOPPED:
                tb.send_idle()

            # 3. Rating click detection (starts a pulse)
            if slider_phoda.getRating() is not None and not rating_pulse_sent:
                tb.send_event(49, "rating_pulse")
                rating_pulse_active = True
                rating_pulse_sent = True
                rating_timer.reset()
    
    3. MONITORING CONNECTION (optional):
       
        To display connection status in your UI:
       
        In "Each Frame" of a code component:
            if tb.get_status()["connected"] is False:
                warning_text = "No TriggerBox connection!"
            else:
                warning_text = ""

        Then display `$warning_text` in a Text component.

    ===========================================================================
    FEATURES
    ===========================================================================

    - Trigger values: 0-255 (8-bit); stick to a pattern for your experiment!
    - Custom names: Name each trigger however you like (e.g., "fixation", "stimulus") (for logging)
    - Single output: TriggerBox outputs one 8-bit value at a time; each send overwrites previous
    - Auto-discovery: Scans for TriggerBox by description if preferred port fails
    - Graceful degradation: Continues experiment if port is missing (logs warnings)
    - Reconnection: Attempts to reconnect every x seconds if disconnected
    - Deduplication: Tracks current trigger value; only sends first send_event() per frame

    ===========================================================================
    RECOMMENDED TRIGGER MAP (for consistency across tasks)
    ===========================================================================
    Global:
        0     idle baseline (when nothing else is happening)
        1     experiment start
        255   experiment end (reset byte, sent at experiment end)

    Open/Closed Eyes (10-19):
        10    fixation duration for eyes open
        11    fixation duration for eyes closed

    Open/Closed Eyes 2x (20-29):
        20    fixation duration for eyes open
        21    fixation duration for eyes closed

    Emotion Regulation (30-39):
        30    neutral image duration
        31    practice image duration
        32    negative image duration
        ...
        38    regulation cue

    PHODA (40-49):
        40    image duration
        ...
        49    rating pulse

    Pain Regulation (50-59):
        50    stimulus duration
        ...
        59    rating pulse    
    
    State Measure (200-239):
        200   placeholder 
    
    Audio (global):
        240   sound played
    """

    def __init__(
        self,
        *,
        preferred_ports = ("COM3", "COM4", "COM5"), # Add common defaults if needed
        description_hint: str = "TriggerBox", # Description hint to identify the device in Device Manager
        baudrate: int = 9600, # Standard baudrate, though our TriggerBox doesn't actually care
        idle_byte: int = 0x00, # Idle baseline byte
        reset_byte: int = 0xFF, # Experiment end byte
        reconnect_interval_sec: float = 1.0, # seconds between reconnection attempts
    ):
        self.preferred_ports = list(preferred_ports)
        self.description_hint = description_hint
        self.baudrate = baudrate
        self.idle_byte = idle_byte
        self.reset_byte = reset_byte
        self.reconnect_interval_sec = reconnect_interval_sec

        self.serial_device = None
        self.port_in_use: Optional[str] = None
        self.connected: bool = False
        self.last_connection_state: bool = None # ensures first log fires
        self.last_trigger_value: Optional[int] = None
        self._last_reconnect_attempt: float = 0.0

    # ------------------------------------------------------------------
    # Logging helpers (because PsychoPy logging doesn't like to update)
    # ------------------------------------------------------------------
    def _log_error(self, msg: str) -> None:
        """Log an error message, and flush immediately."""
        logging.error(msg)
        logging.flush()

    def _log_warning(self, msg: str) -> None:
        """Log a warning message, and flush immediately."""
        logging.warning(msg)
        logging.flush()

    def _log_info(self, msg: str) -> None:
        """Log an info message, and flush immediately."""
        logging.info(msg)
        logging.flush()

    def _log_data(self, msg: str) -> None:
        """Log a data message, and flush immediately."""
        logging.data(msg)
        logging.flush()
    
    def _log_state_change(self) -> None:
        """Log only when connection state changes."""
        if self.connected != self.last_connection_state:
            if self.connected:
                self._log_info(f"TriggerBoxManager: connected on {self.port_in_use}")
            else:
                self._log_warning("TriggerBoxManager: disconnected")
            self.last_connection_state = self.connected
    
    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------
    def begin_experiment(self) -> None:
        """Placed at Begin Experiment to initialize connection."""
        self._connect_if_needed(initial=True)

    def end_experiment(self) -> None:
        """Placed at End Experiment to close connection and send reset byte."""
        self._send_reset(name="experiment_end")
        if self.serial_device:
            try:
                self.serial_device.close()
            except Exception:
                pass
        self.connected = False
        self.port_in_use = None
        self._log_state_change()

    def _connect_if_needed(self, *, initial=False) -> None:
        """Attempt to connect if needed, with optional initial flag to bypass throttling."""
        now = time.time()
        # Skip the interval if initial is set to True, otherwise throttle attempts
        if not initial and (now - self._last_reconnect_attempt) < self.reconnect_interval_sec:
            return
        self._last_reconnect_attempt = now

        if serial is None:
            self.connected = False
            self._log_state_change()
            return

        # Build candidate port list
        candidates = list(self.preferred_ports)
        # Then scan for matching description, in case it's on a different port
        try:
            for port in serial.tools.list_ports.comports():
                if self.description_hint.lower() in port.description.lower():
                    candidates.append(port.device)
        except Exception:
            pass

        # Try ports
        for port_name in dict.fromkeys(candidates):  # remove duplicates
            try:
                self._open_serial(port_name)
                self.connected = True
                self.port_in_use = port_name
                self._log_state_change()
                return
            except Exception:
                continue

        # If all failed
        self.connected = False
        self.port_in_use = None
        self._log_state_change()

    def _open_serial(self, port_name: str) -> None:
        """Open a serial connection to the specified port."""
        # Only called when we intend to open a new connection
        # This happens when connecting for the first time or reconnecting
        if self.serial_device is not None:
            try:
                # Close existing connection, if any to avoid port conflicts and resource leaks
                self.serial_device.close()
            except Exception:
                pass
        # Open new connection
        self.serial_device = serial.Serial(port=port_name, baudrate=self.baudrate, timeout=0)

    # ------------------------------------------------------------
    # Sending values
    # ------------------------------------------------------------
    def _send_value(self, value: int, name: Optional[str]) -> bool:
        """Attempts to send a byte value (0-255) to the TriggerBox."""
        if not (0 <= value <= 255):
            raise ValueError(f"Trigger value {value} is out of range (0-255).")

        # Let's not overwork our computer if we don't have to!
        if self.last_trigger_value == value:
            return True
        
        # Ensure connection (with throttled reconnect attempts)
        if not self.connected or not self.serial_device:
            self._connect_if_needed()

        # If still not connected, give up for now (will retry on next call)
        if not self.connected or not self.serial_device:
            return False
        
        # Send the byte
        try:
            self.serial_device.write(bytes([value]))
            self.last_trigger_value = value # Stop any further sends of the same value until it changes
            label = f" ({name})" if name else ""
            self._log_data(f"TriggerBoxManager: sent {value}{label}")
            return True
        except Exception:
            self.connected = False
            self._log_state_change()
            return False

    # ------------------------------------------------------------------
    # Trigger operations
    # ------------------------------------------------------------------
    def send_event(self, value: int, name: Optional[str] = "event") -> bool:
        """Send a trigger value (1-254). Optionally, set a name for logging."""
        return self._send_value(value, name)

    def send_idle(self, name: Optional[str] = "idle") -> bool:
        """Send idle baseline (0)."""
        return self._send_value(self.idle_byte, name)

    def _send_reset(self, name: Optional[str] = "reset") -> bool:
        """Send reset byte (255), used in end_experiment."""
        return self._send_value(self.reset_byte, name)
    
    # ------------------------------------------------------------
    # Status
    # ------------------------------------------------------------
    def get_status(self) -> dict:
        # You can use "connected" to set up a Psychopy CodeComponent to monitor connection status
        return {
            "connected": self.connected, 
            "port": self.port_in_use,
            "last_trigger_value": self.last_trigger_value,
        }

__all__ = ["TriggerBoxManager"]
