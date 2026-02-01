from __future__ import annotations

from typing import Dict, List

from core.models import Candle, MarketRegime
from core.utils.math_utils import atr, bollinger_bands, ema


def classify_regime(candles: List[Candle], config: Dict) -> MarketRegime:
    if not candles:
        return MarketRegime.UNKNOWN

    params = config.get("daily_regime", {})
    min_bars = int(params.get("min_bars", 60))
    if len(candles) < min_bars:
        return MarketRegime.UNKNOWN

    atr_period = int(params.get("atr_period", 14))
    bb_period = int(params.get("bb_period", 20))
    volume_ma = int(params.get("volume_ma", 20))

    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    closes = [c.close for c in candles]
    volumes = [c.volume for c in candles]

    atr_values = atr(highs, lows, closes, atr_period)
    current_atr = atr_values[-1]
    avg_atr = atr_values[-min_bars:].mean() if len(atr_values) >= min_bars else atr_values.mean()
    atr_ratio = current_atr / avg_atr if avg_atr else 1.0

    _, _, _, bb_width = bollinger_bands(closes, bb_period, 2.0)
    current_bb = bb_width[-1]

    vol_series = ema(volumes, volume_ma)
    vol_ratio = volumes[-1] / vol_series[-1] if vol_series[-1] else 1.0

    score = (atr_ratio * 0.5) + (current_bb * 10 * 0.3) + (vol_ratio * 0.2)

    thresholds = params.get("thresholds", {})
    aggressive = float(thresholds.get("aggressive", 1.6))
    moderate = float(thresholds.get("moderate", 1.1))
    chill = float(thresholds.get("chill", 0.7))

    if score >= aggressive:
        return MarketRegime.AGGRESSIVE
    if score >= moderate:
        return MarketRegime.MODERATE
    if score <= chill:
        return MarketRegime.CHILL
    return MarketRegime.MODERATE