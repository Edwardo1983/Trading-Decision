from __future__ import annotations

import logging
import socket
import struct
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

_NTP_PACKET = b"\x1b" + 47 * b"\0"
_NTP_DELTA = 2208988800


def query_ntp_offset(server: str, timeout_seconds: float) -> float:
    # NTP offset estimate using local send/receive midpoint.
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.settimeout(timeout_seconds)
        send_time = time.time()
        sock.sendto(_NTP_PACKET, (server, 123))
        payload, _ = sock.recvfrom(48)
        recv_time = time.time()

    if len(payload) < 48:
        raise RuntimeError(f"NTP payload too short from {server}")

    values = struct.unpack("!12I", payload[:48])
    transmit_seconds = values[10]
    transmit_fraction = values[11]
    server_unix = (transmit_seconds - _NTP_DELTA) + (transmit_fraction / 2**32)
    local_midpoint = (send_time + recv_time) / 2.0
    return server_unix - local_midpoint


@dataclass
class ClockSync:
    enabled: bool = False
    servers: List[str] = field(
        default_factory=lambda: ["time.google.com", "time.cloudflare.com", "pool.ntp.org"]
    )
    timeout_seconds: float = 1.5
    sync_interval_seconds: int = 600
    max_offset_seconds: float = 3.0
    _offset_seconds: float = 0.0
    _last_sync_mono: float = 0.0
    _last_error: str = ""
    _active_server: Optional[str] = None
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]]) -> "ClockSync":
        cfg = dict(config or {})
        servers = cfg.get("servers", ["time.google.com", "time.cloudflare.com", "pool.ntp.org"])
        if not isinstance(servers, list) or not servers:
            servers = ["time.google.com", "time.cloudflare.com", "pool.ntp.org"]
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            servers=[str(item).strip() for item in servers if str(item).strip()],
            timeout_seconds=float(cfg.get("timeout_seconds", 1.5)),
            sync_interval_seconds=max(30, int(cfg.get("sync_interval_seconds", 600))),
            max_offset_seconds=max(0.0, float(cfg.get("max_offset_seconds", 3.0))),
        )

    def sync_if_due(self) -> bool:
        if not self.enabled:
            return False
        now_mono = time.monotonic()
        with self._lock:
            due = (self._last_sync_mono <= 0) or (
                (now_mono - self._last_sync_mono) >= self.sync_interval_seconds
            )
        if due:
            return self.sync(force=True)
        return True

    def sync(self, force: bool = False) -> bool:
        if not self.enabled:
            return False
        if not force:
            now_mono = time.monotonic()
            with self._lock:
                if self._last_sync_mono > 0 and (
                    now_mono - self._last_sync_mono
                ) < self.sync_interval_seconds:
                    return True

        errors: List[str] = []
        for server in self.servers:
            try:
                offset = query_ntp_offset(server=server, timeout_seconds=self.timeout_seconds)
                with self._lock:
                    self._offset_seconds = float(offset)
                    self._last_sync_mono = time.monotonic()
                    self._last_error = ""
                    self._active_server = server
                if self.max_offset_seconds > 0 and abs(offset) > self.max_offset_seconds:
                    logger.warning(
                        "Clock offset vs NTP is %.3fs (> %.3fs tolerance) using %s",
                        offset,
                        self.max_offset_seconds,
                        server,
                    )
                else:
                    logger.info("Clock synced via NTP server %s (offset %.3fs)", server, offset)
                return True
            except Exception as exc:
                errors.append(f"{server}: {exc}")

        error_text = "; ".join(errors) if errors else "all NTP servers failed"
        with self._lock:
            self._last_error = error_text
            self._active_server = None
            self._last_sync_mono = time.monotonic()
        logger.warning("NTP sync failed. Falling back to local clock. Details: %s", error_text)
        return False

    def now_utc(self) -> datetime:
        offset = 0.0
        if self.enabled:
            self.sync_if_due()
            with self._lock:
                offset = self._offset_seconds
        return datetime.now(timezone.utc) + timedelta(seconds=offset)

    def now_tz(self, tz_name: str) -> datetime:
        return self.now_utc().astimezone(ZoneInfo(tz_name))

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "active_server": self._active_server,
                "offset_seconds": round(self._offset_seconds, 6),
                "last_error": self._last_error,
                "last_sync_mono": self._last_sync_mono,
                "max_offset_seconds": self.max_offset_seconds,
            }
