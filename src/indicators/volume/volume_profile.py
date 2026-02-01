from __future__ import annotations

from typing import Dict, List
from collections import defaultdict

from core.models import Candle, IndicatorResult, SignalState
from indicators.base import IndicatorBase


class VolumeProfileIndicator(IndicatorBase):
    """
    Volume Profile indicator.
    Shows volume distribution at different price levels.

    Key Levels:
    - POC (Point of Control): Price level with highest volume
    - VAH (Value Area High): Upper bound of value area (70% of volume)
    - VAL (Value Area Low): Lower bound of value area

    Signals:
    - Price at POC: Strong support/resistance
    - Price outside value area: Potential breakout or mean reversion
    """
    name = "volume_profile"
    category = "volume"
    timeframes_required = ["1m"]

    def compute(self, candles_by_tf: Dict[str, List[Candle]]) -> IndicatorResult:
        candles = candles_by_tf.get(self.timeframes_required[0], [])
        lookback = int(self.params.get("lookback", 100))
        num_bins = int(self.params.get("num_bins", 50))
        value_area_pct = float(self.params.get("value_area_pct", 0.70))

        if len(candles) < lookback:
            return IndicatorResult(
                self.name, self.category, self.timeframes_required[0],
                {}, SignalState.NEUTRAL, 0, "insufficient data", self.weight
            )

        window = candles[-lookback:]
        current_price = candles[-1].close

        # Find price range
        price_high = max(c.high for c in window)
        price_low = min(c.low for c in window)
        price_range = price_high - price_low

        if price_range <= 0:
            return IndicatorResult(
                self.name, self.category, self.timeframes_required[0],
                {}, SignalState.NEUTRAL, 0, "no price range", self.weight
            )

        bin_size = price_range / num_bins

        # Build volume profile
        volume_at_price = defaultdict(float)
        for c in window:
            # Distribute candle volume across price levels it touched
            low_bin = int((c.low - price_low) / bin_size)
            high_bin = int((c.high - price_low) / bin_size)
            bins_touched = max(1, high_bin - low_bin + 1)
            volume_per_bin = c.volume / bins_touched

            for b in range(low_bin, min(high_bin + 1, num_bins)):
                price_level = price_low + (b + 0.5) * bin_size
                volume_at_price[price_level] += volume_per_bin

        # Find POC (highest volume level)
        poc_price = max(volume_at_price.keys(), key=lambda p: volume_at_price[p])
        poc_volume = volume_at_price[poc_price]

        # Calculate Value Area (70% of total volume)
        total_volume = sum(volume_at_price.values())
        target_volume = total_volume * value_area_pct

        # Expand from POC until we capture 70% of volume
        sorted_by_volume = sorted(volume_at_price.items(), key=lambda x: x[1], reverse=True)
        accumulated_volume = 0
        value_area_prices = []

        for price, vol in sorted_by_volume:
            accumulated_volume += vol
            value_area_prices.append(price)
            if accumulated_volume >= target_volume:
                break

        vah = max(value_area_prices) if value_area_prices else price_high
        val = min(value_area_prices) if value_area_prices else price_low

        # Determine position relative to value area
        above_vah = current_price > vah
        below_val = current_price < val
        in_value_area = val <= current_price <= vah

        # Distance to POC
        distance_to_poc = abs(current_price - poc_price)
        distance_to_poc_pct = (distance_to_poc / poc_price) * 100

        state = SignalState.NEUTRAL
        reason = "in value area"
        confidence = 40.0

        if distance_to_poc_pct < 0.3:
            # At POC - strong level
            if current_price > poc_price:
                state = SignalState.BUY
                reason = "at POC (high volume support)"
            else:
                state = SignalState.SELL
                reason = "at POC (high volume resistance)"
            confidence = 70.0
        elif above_vah:
            state = SignalState.BUY
            reason = "breakout above value area"
            confidence = 65.0
        elif below_val:
            state = SignalState.SELL
            reason = "breakdown below value area"
            confidence = 65.0
        elif current_price > poc_price:
            state = SignalState.BUY
            reason = "above POC in value area"
            confidence = 50.0
        else:
            state = SignalState.SELL
            reason = "below POC in value area"
            confidence = 50.0

        value = {
            "poc": round(poc_price, 4),
            "vah": round(vah, 4),
            "val": round(val, 4),
            "price": current_price,
            "above_vah": above_vah,
            "below_val": below_val,
            "in_value_area": in_value_area,
            "distance_to_poc_pct": round(distance_to_poc_pct, 3),
        }
        return IndicatorResult(
            self.name, self.category, self.timeframes_required[0],
            value, state, min(100, confidence), reason, self.weight
        )
