"""TriggerBox helper for PsychoPy Builder code.

Provides connection discovery, reconnection, and high/low trigger management
without crashing when the COM port is missing. Designed to be instantiated once
in a "Begin Experiment" code component and called from per-routine code.
"""

from __future__ import annotations

import time
from typing import Dict, Iterable, List, Optional

from psychopy import logging

try:
    import serial
    import serial.tools.list_ports
except Exception:  # pragma: no cover - environment dependent
    serial = None  # type: ignore


class TriggerBoxManager:
    """Manage a Brain Products TriggerBox over a virtual serial port."""

    def __init__(
        self,
        *,
        preferred_ports: Optional[Iterable[str]] = None,
        description_hint: str = "TriggerBox",
        baudrate: int = 9600,
        stop_byte: int = 0x00,
        reset_byte: int = 0xFF,
        reconnect_interval_sec: float = 1.0,
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

    def _connect_if_needed(self, *, initial: bool = False) -> None:
        now = time.time()
        if not initial and (now - self._last_reconnect_attempt) < self.reconnect_interval_sec:
            return
        self._last_reconnect_attempt = now

        if serial is None:
            logging.error("TriggerBoxManager: pyserial not available; cannot connect.")
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
                logging.info(f"TriggerBoxManager: connected on {port_name} (baud={self.baudrate}).")
                self.connected = True
                self.connection_warning = False
                self.port_in_use = port_name
                return
            except Exception as exc:
                logging.warning(f"TriggerBoxManager: failed to open {port_name}: {exc}")

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
        if value in self._active.values():
            other = [k for k, v in self._active.items() if v == value]
            logging.warning(
                "TriggerBoxManager: duplicate trigger value %s requested by %s (already active by %s)",
                value,
                name,
                ",".join(other),
            )
        self._active[name] = value
        return self._send_value(value)

    def stop(self, name: str) -> bool:
        """Send stop byte for a named trigger if it was active."""
        if name not in self._active:
            return False
        self._active.pop(name, None)
        return self._send_value(self.stop_byte)

    def stop_all(self) -> bool:
        """Stop all active triggers and send reset byte."""
        self._active.clear()
        return self._send_value(self.reset_byte)

    # ------------------------------------------------------------------
    # Internal send with reconnection
    # ------------------------------------------------------------------
    def _send_value(self, value: int) -> bool:
        if not self.connected or self.serial is None:
            self._connect_if_needed()
        if not self.connected or self.serial is None:
            self.failure_count += 1
            self.connection_warning = True
            logging.warning("TriggerBoxManager: send skipped; no connection.")
            return False

        try:
            self.serial.write(bytes([value]))  # type: ignore[call-arg]
            self.last_trigger_value = value
            logging.data(f"TriggerBoxManager: sent {value} on {self.port_in_use}")
            return True
        except Exception as exc:
            self.failure_count += 1
            logging.warning(f"TriggerBoxManager: write failed ({exc}); attempting reconnect.")
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
