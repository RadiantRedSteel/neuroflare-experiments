"""TriggerBox helper for PsychoPy Builder code.

This module provides the TriggerBoxManager class for managing Brain Products
TriggerBox communication in PsychoPy Builder experiments.
"""

from __future__ import annotations

import time
from typing import Dict, Iterable, List, Optional, Set

from psychopy import logging

try:
    import serial
    import serial.tools.list_ports
except Exception:  # pragma: no cover - environment dependent
    serial = None  # type: ignore


class TriggerBoxManager:
    """Manage a Brain Products TriggerBox over a virtual serial port.

    Provides connection discovery, reconnection, and high/low trigger management
    without crashing when the COM port is missing. Designed to be instantiated once
    in a "Begin Experiment" code component and called from per-routine code.

    ===========================================================================
    USAGE IN PSYCHOPY BUILDER
    ===========================================================================
    
    1. SETUP CODE COMPONENT (place at the very start of your experiment):
       
        Create a Code Component in your first routine and add this to "Begin Experiment":
       
            from neuroflare.shared.trigger_box import TriggerBoxManager
            tb = TriggerBoxManager(preferred_ports=["COM3"], baudrate=9600)
            tb.begin_experiment()
       
        And add this to "End Experiment":
       
            tb.end_experiment()
    
    2. TRIGGER CODE COMPONENTS (place BELOW the component you're triggering):
       
        Important: Component order matters! Each frame, Builder executes components
        from top to bottom. Your trigger code must come AFTER the component it
        synchronizes with, so both have updated status values.
       
        You can create multiple trigger components with different names and values.
        Values range from 0-255 (8-bit). Choose unique trigger values for each event.
       
        In "Each Frame":
       
            if cross_fixation.status == STARTED:
                tb.start(name="fixation", value=1)
            if cross_fixation.status == STOPPED:
                tb.stop(name="fixation")
       
        Multiple simultaneous triggers are supported:
        
            if stimulus.status == STARTED:
                tb.start(name="stimulus", value=10)
            if audio.status == STARTED:
                tb.start(name="audio", value=20)
       
        Belt-and-suspenders cleanup in "End Routine":
       
            tb.stop(name="fixation")
            tb.stop(name="stimulus")
            tb.stop(name="audio")
            or simply:
                tb.stop_all()
    
    3. MONITORING CONNECTION (optional):
       
        To display connection status in your UI:
       
        In "Each Frame" of a code component:
            if tb.get_status()["warning_no_connection"]:
                warning_text = "No TriggerBox connection!"
            else:
                warning_text = ""

        Then display `$warning_text` in a Text component.

    ===========================================================================
    FEATURES
    ===========================================================================

    - Trigger values: 0-255 (8-bit); choose unique values for different events
    - Custom names: Name each trigger however you like (e.g., "fixation", "stimulus")
    - Multiple triggers: Send multiple simultaneous triggers with different values
    - Auto-discovery: Scans for TriggerBox by description if preferred port fails
    - Graceful degradation: Continues experiment if port is missing (logs warnings)
    - Reconnection: Attempts to reconnect every x seconds if disconnected
    - Deduplication: Tracks active triggers; only sends first start() per frame
    - Safety: Automatically sends 0xFF reset on experiment end
    """

    def __init__(
        self,
        *,
        preferred_ports: Optional[Iterable[str]] = None,
        description_hint: str = "TriggerBox",
        baudrate: int = 9600,
        stop_byte: int = 0x00,
        reset_byte: int = 0xFF,
        reconnect_interval_sec: float = 1.0, # seconds between reconnection attempts
    ) -> None:
        self.preferred_ports: List[str] = list(preferred_ports or ["COM3"])
        self.description_hint = description_hint
        self.baudrate = baudrate
        self.stop_byte = stop_byte
        self.reset_byte = reset_byte
        self.reconnect_interval_sec = reconnect_interval_sec

        self.serial = None
        self.port_in_use: Optional[str] = None
        self.connected: bool = False
        self.connection_warning: bool = False
        self.last_trigger_value: Optional[int] = None
        self.failure_count: int = 0
        self._last_reconnect_attempt: float = 0.0
        self._active: Dict[str, int] = {}
        self._failed_stops: Set[str] = set()  # Track trigger names that failed to stop

    # ------------------------------------------------------------------
    # Logging helpers (with immediate flush for real-time visibility)
    # ------------------------------------------------------------------
    def _log_error(self, msg: str) -> None:
        logging.error(msg)
        logging.flush()

    def _log_warning(self, msg: str) -> None:
        logging.warning(msg)
        logging.flush()

    def _log_info(self, msg: str) -> None:
        logging.info(msg)
        logging.flush()

    def _log_data(self, msg: str) -> None:
        logging.data(msg)
        logging.flush()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------
    def begin_experiment(self) -> None:
        """Attempt initial connection."""
        self._connect_if_needed(initial=True)

    def end_experiment(self) -> None:
        """Reset lines and close the port."""
        self.stop_all()
        if self.serial is not None:
            try:
                self.serial.close()
            except Exception:
                pass
        self.connected = False
        self.port_in_use = None
        self._log_info("TriggerBoxManager: experiment ended, port closed.")

    def _connect_if_needed(self, *, initial: bool = False) -> None:
        now = time.time()
        if not initial and (now - self._last_reconnect_attempt) < self.reconnect_interval_sec:
            return
        self._last_reconnect_attempt = now

        if serial is None:
            self._log_error("TriggerBoxManager: pyserial not available; cannot connect.")
            self.connected = False
            self.connection_warning = True
            return

        # Try preferred ports first
        candidates = list(self.preferred_ports)
        # Then scan for matching description
        try:
            for port in serial.tools.list_ports.comports():  # type: ignore[attr-defined]
                if self.description_hint.lower() in port.description.lower():
                    candidates.append(port.device)
        except Exception:
            pass

        seen = set()
        for port_name in candidates:
            if port_name in seen:
                continue
            seen.add(port_name)
            try:
                self._open_serial(port_name)
                self._log_info(f"TriggerBoxManager: connected on {port_name} (baud={self.baudrate}).")
                self.connected = True
                self.connection_warning = False
                self.port_in_use = port_name
                return
            except Exception as exc:
                self._log_warning(f"TriggerBoxManager: failed to open {port_name}: {exc}")

        # If we reach here, no port worked
        self.connected = False
        self.connection_warning = True
        self.port_in_use = None

    def _open_serial(self, port_name: str) -> None:
        if self.serial is not None:
            try:
                self.serial.close()
            except Exception:
                pass
        self.serial = serial.Serial(port=port_name, baudrate=self.baudrate, timeout=0)  # type: ignore[call-arg]

    # ------------------------------------------------------------------
    # Trigger operations
    # ------------------------------------------------------------------
    def send(self, value: int) -> bool:
        """Send a one-shot value (no tracking)."""
        return self._send_value(value)

    def start(self, *, name: str, value: int) -> bool:
        """Hold a value high until stop(). Warn on duplicate active values."""
        # Attempt reconnection if disconnected (even if trigger is in _active)
        if not self.connected or self.serial is None:
            self._connect_if_needed()
        
        # Clean up any triggers that failed to stop, if we're now connected
        if self.connected and self._failed_stops:
            for failed_name in list(self._failed_stops):
                if self._send_value(self.stop_byte):
                    self._failed_stops.discard(failed_name)
                    self._active.pop(failed_name, None)
                    break  # One success proves reconnection works; others will retry next frame
        
        # If this trigger is already active, don't resend (silently return)
        if name in self._active:
            return self.connected  # Return connection status instead of True
        
        # Check for value conflicts with OTHER triggers
        if value in self._active.values():
            other = [k for k, v in self._active.items() if v == value]
            self._log_warning(
                f"TriggerBoxManager: duplicate trigger value {value} requested by {name} (already active by {','.join(other)})"
            )
        
        # Send the value first
        success = self._send_value(value)
        # Only mark as active if send succeeded
        if success:
            self._active[name] = value
        return success

    def stop(self, name: str) -> bool:
        """Send stop byte for a named trigger if it was active."""
        if name not in self._active:
            return False
        success = self._send_value(self.stop_byte)
        if success:
            self._active.pop(name, None)
            self._failed_stops.discard(name)  # No longer failed
        else:
            self._failed_stops.add(name)  # Track as failed to stop
        return success

    def stop_all(self) -> bool:
        """Stop all active triggers and send reset byte."""
        success = self._send_value(self.reset_byte)
        if success:
            self._active.clear()
            self._failed_stops.clear()
        else:
            # If reset failed, track all active triggers as failed stops
            self._failed_stops.update(self._active.keys())
        return success

    # ------------------------------------------------------------------
    # Internal send with reconnection
    # ------------------------------------------------------------------
    def _send_value(self, value: int) -> bool:
        if not self.connected or self.serial is None:
            self._connect_if_needed()
        if not self.connected or self.serial is None:
            return False

        try:
            self.serial.write(bytes([value]))  # type: ignore[call-arg]
            self.last_trigger_value = value
            self._log_data(f"TriggerBoxManager: sent {value} on {self.port_in_use}")
            return True
        except Exception as exc:
            self.failure_count += 1
            self._log_warning(f"TriggerBoxManager: write failed ({exc}); attempting reconnect.")
            self.connected = False
            self._connect_if_needed()
            return False

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------
    def get_status(self) -> Dict[str, object]:
        return {
            "connected": self.connected,
            "port": self.port_in_use,
            "warning_no_connection": self.connection_warning,
            "last_trigger_value": self.last_trigger_value,
            "failure_count": self.failure_count,
            "active_triggers": dict(self._active),
        }


__all__ = ["TriggerBoxManager"]
