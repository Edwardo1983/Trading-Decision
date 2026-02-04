from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import math

from skyfield.api import load
from skyfield import almanac
from zoneinfo import ZoneInfo

from core.models import IndicatorResult, SignalState
from core.utils.paths import assets_dir
from indicators.base import IndicatorBase


@dataclass
class AstroCache:
    last_calc: Optional[datetime] = None
    next_new: Optional[datetime] = None
    next_full: Optional[datetime] = None


class AstroCalendarIndicator(IndicatorBase):
    name = "astro_calendar"
    category = "context"
    timeframes_required = ["1m"]

    def __init__(self, params: Dict, weight: float):
        super().__init__(params, weight)
        self._cache = AstroCache()
        self._eph = None
        self._ts = None

    def _load_ephemeris(self):
        if self._eph is None:
            eph_path = assets_dir() / "ephemeris" / "de421.bsp"
            if not eph_path.exists():
                eph_path = load("de421.bsp")
            else:
                eph_path = load(str(eph_path))
            self._eph = eph_path
            self._ts = load.timescale()

    def _phase_label(self, angle_deg: float) -> str:
        if angle_deg < 15 or angle_deg > 345:
            return "new"
        if 165 < angle_deg < 195:
            return "full"
        return "waxing" if angle_deg < 180 else "waning"

    def _compute_next_events(self, now: datetime) -> None:
        if self._cache.last_calc and (now - self._cache.last_calc) < timedelta(hours=6):
            return
        self._load_ephemeris()
        t0 = self._ts.from_datetime(now)
        t1 = self._ts.from_datetime(now + timedelta(days=30))
        f = almanac.moon_phases(self._eph)
        times, phases = almanac.find_discrete(t0, t1, f)
        next_new = None
        next_full = None
        for t, phase in zip(times, phases):
            if phase == 0 and next_new is None:
                next_new = t.utc_datetime()
            if phase == 2 and next_full is None:
                next_full = t.utc_datetime()
            if next_new and next_full:
                break
        self._cache.last_calc = now
        self._cache.next_new = next_new
        self._cache.next_full = next_full

    def _aspect_list(self, now: datetime, orb_deg: float, planets: List[str]) -> List[str]:
        if not planets:
            return []
        self._load_ephemeris()
        t = self._ts.from_datetime(now)
        moon = self._eph["moon"]
        earth = self._eph["earth"]
        moon_lon = earth.at(t).observe(moon).ecliptic_latlon()[1].degrees
        aspects = []
        targets = {
            "Mercury": "mercury",
            "Venus": "venus",
            "Mars": "mars",
            "Jupiter": "jupiter barycenter",
            "Saturn": "saturn barycenter",
        }
        for name in planets:
            key = targets.get(name)
            if not key:
                continue
            lon = earth.at(t).observe(self._eph[key]).ecliptic_latlon()[1].degrees
            diff = abs((moon_lon - lon + 180) % 360 - 180)
            if diff <= orb_deg:
                aspects.append(f"Moon-{name} conjunction")
            elif abs(diff - 180) <= orb_deg:
                aspects.append(f"Moon-{name} opposition")
            elif abs(diff - 90) <= orb_deg:
                aspects.append(f"Moon-{name} square")
            elif abs(diff - 120) <= orb_deg:
                aspects.append(f"Moon-{name} trine")
        return aspects

    def compute(self, candles_by_tf: Dict[str, List]) -> IndicatorResult:
        tz_name = self.params.get("timezone", "Europe/Bucharest")
        location = self.params.get("location", {"lat": 44.4268, "lon": 26.1025})
        now = datetime.now(ZoneInfo(tz_name))
        self._compute_next_events(now)
        self._load_ephemeris()
        t = self._ts.from_datetime(now)
        phase_angle = almanac.moon_phase(self._eph, t).degrees
        illumination = (1 - math.cos(math.radians(phase_angle))) / 2
        label = self._phase_label(phase_angle)

        window_hours = float(self.params.get("new_full_window_hours", 24))
        next_new = self._cache.next_new
        next_full = self._cache.next_full
        is_new_window = bool(next_new and abs((next_new - now).total_seconds()) <= window_hours * 3600)
        is_full_window = bool(next_full and abs((next_full - now).total_seconds()) <= window_hours * 3600)

        aspects_enabled = bool(self.params.get("aspects_enabled", False))
        orb = float(self.params.get("orb_degrees", 3))
        planets = list(self.params.get("planets", []))
        aspects = self._aspect_list(now, orb, planets) if aspects_enabled else []

        value = {
            "phase_angle": phase_angle,
            "illumination": illumination,
            "label": label,
            "next_new": next_new.isoformat() if next_new else None,
            "next_full": next_full.isoformat() if next_full else None,
            "is_new_window": is_new_window,
            "is_full_window": is_full_window,
            "aspects": aspects,
            "location": location,
        }
        state = SignalState.NEUTRAL
        confidence = 20.0
        reason = "experimental astro context"
        if is_new_window:
            state = SignalState.BUY
            confidence = 40.0
            reason = "new moon bias"
        elif is_full_window:
            state = SignalState.SELL
            confidence = 40.0
            reason = "full moon bias"

        return IndicatorResult(
            self.name,
            self.category,
            "1m",
            value,
            state,
            confidence,
            reason,
            self.weight,
            {"astro_window": is_new_window or is_full_window},
        )
