from __future__ import annotations
import time
from typing import Optional
from psychopy import logging

try:
    import serial
    import serial.tools.list_ports
except Exception:  # pragma: no cover - environment dependent
    serial = None  # type: ignore

# Only the TriggerBox Extension can truly trigger external devices (with actiCHamp Plus mirror mode).
# This is still useful for logging triggers in a consistent way, without crashing if the COM port is missing.
# It also provides a single place to manage trigger values and names.

class TriggerBoxManager:
    """
    Manage a Brain Products TriggerBox over a virtual serial port.
    Specfically, the TriggerBox appears as a USB virtual COM port.

    Provides connection discovery, reconnection, and high/low trigger management
    (0-255) without crashing when the COM port is missing. Designed to be instantiated 
    once in a "Begin Experiment" code component and called from per-routine code.

    The ports of the TriggerBox activate like the bits in a byte. For example, if you
    want to activate port 0 and port 6, you would send a value of 65 (1 for port 0, 64 for port 6).

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

            tb.send_idle()

        This helps to clear out any residual triggers, and also acts as both a marker and connection insurance.
       
        Lastly, add this to "End Experiment":
       
            tb.end_experiment()
    
    2. TRIGGER CODE COMPONENTS (place BELOW the component you're triggering):
       
        Important: Component order matters! Each frame, Builder executes components
        from top to bottom. Your trigger code must come AFTER the component it
        synchronizes with, so both have updated status values.
       
        You can create multiple trigger components with different names and values.
        Values range from 0-255 (8-bit). I recommended staying consistent.

        You are expected to handle repeated values in Builder logic. That means you
        can call send_idle() in multiple end routines, or send the same trigger value
        twice in a row, without this class suppressing it.

        A value of 0 is still meaningful as an idle/reset marker for your own logic,
        even though BrainVision Recorder will not show it as a visible event.

        In BrainVision Recorder, if you want to send a value of 1 twice,
        you'll need to send 0 in between to reset the state.

        If you are sending different values, such as 1 and 2,
        you can send them back to back without needing to reset to 0.

        A common Builder-style pattern is to mirror component status with your own
        trigger state variable:

            # Begin Routine
            trigger_status = NOT_STARTED

            # Each Frame
            if cross_fixation.status == STARTED and trigger_status == NOT_STARTED:
                tb.send_event(*current_trigger_start)
                trigger_status = STARTED

            elif cross_fixation.status == STOPPED and trigger_status == STARTED:
                tb.send_event(*current_trigger_stop)
                trigger_status = STOPPED

        This keeps the trigger flow aligned with PsychoPy Builder semantics:
        STARTED means the event is active, STOPPED means the component has been told
        to stop, and NOT_STARTED means it has not fired yet.

        NOTE: Be careful on components that can force end the routine, as the each frame logic can be missed.
        In those cases, you may want to place the trigger end logic in "End Routine" instead, and just check the component status there.
    
    3. MONITORING CONNECTION (optional):
       
        To display connection status in your UI:
       
        In "Begin Routine" of a code component:
            tb.send_idle() # This will attempt to connect and can be used to trigger the warning if it fails
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
    """

    def __init__(
        self,
        *,        
        preferred_ports = ("COM3"), # Add common defaults if needed
        description_hint: str = "TriggerBox", # Description hint to identify the device in Device Manager
        baudrate: int = 9600, # Standard baudrate, though our TriggerBox doesn't actually care
        idle_byte: int = 0x00, # Idle baseline byte
        # reset_byte: int = 0xFF, # Experiment end byte
        reconnect_interval_sec: float = 1.0, # seconds between reconnection attempts
    ):
        self.preferred_ports = list(preferred_ports)
        self.description_hint = description_hint
        self.baudrate = baudrate
        self.idle_byte = idle_byte
        # self.reset_byte = reset_byte
        self.reconnect_interval_sec = reconnect_interval_sec

        self.serial_device = None
        self.port_in_use: Optional[str] = None
        self.connected: bool = False
        self.last_connection_state: bool = None # ensures first log fires
        self._last_reconnect_attempt: float = 0.0

        # Global trigger map
        # Note: TriggerBox utilizes 8 bits as output from the USB port.
        # You must take into account which ports you wish to set as "HIGH" (1). 
        # For example, if you want to fire both port 0 and port 6, 
        # you would need to send a value of 65 (1 for port 0, 64 for port 6).
        # Other values could work too, as long as both bits are set to 1 in the byte you send.
        # Markers are determined by what the software receiving the triggers is set up to recognize.
        self.TRIG = {
            "open_eyes_start": 1,
            "open_eyes_end": 2,
            "closed_eyes_start": 3,
            "closed_eyes_end": 4,
            "phoda_image_view_start": 5,
            "phoda_image_view_end": 6,
            "phoda_image_rate_start": 7,
            "phoda_image_rate_end": 8,
            # etc...
        }
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
    
    # TODO: Implement thisExp data saving of triggers in the same 
    # format as PsychoPy's data logging so it can be merged later.

    # TODO: Log manager that allows user to use logging or printing on object creation

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------
    def begin_experiment(self) -> None:
        """Placed at Begin Experiment to initialize connection."""
        self._connect_if_needed(initial=True)

    def end_experiment(self) -> None:
        """Placed at End Experiment to close connection."""
        self.send_idle(name="experiment_end")
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

        # Ensure connection (with throttled reconnect attempts)
        if not self.connected or not self.serial_device:
            self._connect_if_needed()

        # If still not connected, give up for now (will retry on next call)
        if not self.connected or not self.serial_device:
            return False
        
        # Send the byte
        try:
            self.serial_device.write(bytes([value]))
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

    # Reset byte could be bad in some contexts, so we may want to remove it for now until we have a clear use case.
    # def _send_reset(self, name: Optional[str] = "reset") -> bool:
    #     """Send reset byte (255), used in end_experiment."""
    #     return self._send_value(self.reset_byte, name)

    # ------------------------------------------------------------
    # Status
    # ------------------------------------------------------------
    def get_status(self) -> dict:
        # You can use "connected" to set up a Psychopy CodeComponent to monitor connection status
        return {
            "connected": self.connected, 
            "port": self.port_in_use,
        }

__all__ = ["TriggerBoxManager"]
