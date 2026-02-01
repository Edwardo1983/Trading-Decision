from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from core.models import Candle, IndicatorResult, SignalState
from core.utils.math_utils import atr
from indicators.base import IndicatorBase


def _pivots(candles: List[Candle], lookback: int) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    highs: List[Tuple[int, float]] = []
    lows: List[Tuple[int, float]] = []
    for i in range(lookback, len(candles) - lookback):
        window = candles[i - lookback : i + lookback + 1]
        center = candles[i]
        if center.high == max(c.high for c in window):
            highs.append((i, center.high))
        if center.low == min(c.low for c in window):
            lows.append((i, center.low))
    return highs, lows


class SmartMoneyIndicator(IndicatorBase):
    name = "smart_money"
    category = "structure"
    timeframes_required = ["1m"]

    def compute(self, candles_by_tf: Dict[str, List[Candle]]) -> IndicatorResult:
        candles = candles_by_tf.get(self.timeframes_required[0], [])
        lookback = int(self.params.get("lookback", 80))
        pivot_lookback = int(self.params.get("pivot_lookback", 3))
        atr_period = int(self.params.get("atr_period", 14))
        ob_impulse_mult = float(self.params.get("ob_impulse_mult", 1.8))
        min_gap_pct = float(self.params.get("min_gap_pct", 0.0005))
        sweep_lookback = int(self.params.get("sweep_lookback", 20))

        if len(candles) < max(lookback, atr_period * 3):
            return IndicatorResult(self.name, self.category, self.timeframes_required[0], {}, SignalState.NEUTRAL, 0, "insufficient data", self.weight)

        window = candles[-lookback:]
        highs = [c.high for c in window]
        lows = [c.low for c in window]
        closes = [c.close for c in window]

        atr_values = atr(highs, lows, closes, atr_period)
        atr_now = float(atr_values[-1]) if len(atr_values) else 0.0

        # Order block detection (simple + deterministic)
        bullish_ob: Optional[Dict[str, float]] = None
        bearish_ob: Optional[Dict[str, float]] = None
        impulse_threshold = atr_now * ob_impulse_mult if atr_now else 0.0

        for i in range(len(window) - 2, 2, -1):
            c = window[i]
            next_close = window[min(i + 2, len(window) - 1)].close
            # bullish OB: last bearish candle before strong up move
            if c.close < c.open and next_close - c.high > impulse_threshold:
                bullish_ob = {"top": c.high, "bottom": c.low, "index": i}
                break

        for i in range(len(window) - 2, 2, -1):
            c = window[i]
            next_close = window[min(i + 2, len(window) - 1)].close
            # bearish OB: last bullish candle before strong down move
            if c.close > c.open and c.low - next_close > impulse_threshold:
                bearish_ob = {"top": c.high, "bottom": c.low, "index": i}
                break

        # Fair value gap (3-candle)
        bullish_fvg: Optional[Dict[str, float]] = None
        bearish_fvg: Optional[Dict[str, float]] = None
        if len(window) >= 3:
            c1, c2, c3 = window[-3], window[-2], window[-1]
            if c1.high < c3.low:
                gap = (c3.low - c1.high) / max(c1.high, 1e-9)
                if gap >= min_gap_pct:
                    bullish_fvg = {"low": c1.high, "high": c3.low, "gap_pct": gap}
            if c1.low > c3.high:
                gap = (c1.low - c3.high) / max(c3.high, 1e-9)
                if gap >= min_gap_pct:
                    bearish_fvg = {"low": c3.high, "high": c1.low, "gap_pct": gap}

        # Liquidity sweeps
        piv_highs, piv_lows = _pivots(window, pivot_lookback)
        sweep_high = False
        sweep_low = False
        last = window[-1]
        if piv_highs:
            last_high = max(piv_highs[-sweep_lookback:], key=lambda x: x[0])[1] if sweep_lookback else piv_highs[-1][1]
            if last.high > last_high and last.close < last_high:
                sweep_high = True
        if piv_lows:
            last_low = max(piv_lows[-sweep_lookback:], key=lambda x: x[0])[1] if sweep_lookback else piv_lows[-1][1]
            if last.low < last_low and last.close > last_low:
                sweep_low = True

        # BOS / CHoCH (simple)
        bos = None
        choch = None
        if piv_highs and last.close > piv_highs[-1][1]:
            bos = "bullish"
        if piv_lows and last.close < piv_lows[-1][1]:
            bos = "bearish"
        if piv_highs and last.close < piv_highs[-1][1] and bos != "bearish":
            choch = "bearish"
        if piv_lows and last.close > piv_lows[-1][1] and bos != "bullish":
            choch = "bullish"

        score = 0
        if bullish_ob:
            score += 2
        if bullish_fvg:
            score += 1
        if sweep_low:
            score += 1
        if bos == "bullish":
            score += 1
        if bearish_ob:
            score -= 2
        if bearish_fvg:
            score -= 1
        if sweep_high:
            score -= 1
        if bos == "bearish":
            score -= 1

        state = SignalState.NEUTRAL
        reason = "no clear smart money bias"
        if score >= 2:
            state = SignalState.BUY
            reason = "bullish smart money signals"
        elif score <= -2:
            state = SignalState.SELL
            reason = "bearish smart money signals"

        confidence = min(100.0, 40.0 + abs(score) * 15.0)
        value = {
            "bullish_ob": bullish_ob,
            "bearish_ob": bearish_ob,
            "bullish_fvg": bullish_fvg,
            "bearish_fvg": bearish_fvg,
            "sweep_high": sweep_high,
            "sweep_low": sweep_low,
            "bos": bos,
            "choch": choch,
            "atr": round(atr_now, 4),
        }
        return IndicatorResult(self.name, self.category, self.timeframes_required[0], value, state, confidence, reason, self.weight)
