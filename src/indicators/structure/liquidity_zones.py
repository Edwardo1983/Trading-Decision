from __future__ import annotations

from typing import Dict, List, Tuple

from core.models import Candle, IndicatorResult, SignalState
from indicators.base import IndicatorBase


def _find_swing_points(candles: List[Candle], lookback: int) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """Find swing highs and lows with their indices."""
    swing_highs = []
    swing_lows = []

    for i in range(lookback, len(candles) - lookback):
        window = candles[i - lookback : i + lookback + 1]
        center = candles[i]

        is_swing_high = all(center.high >= c.high for c in window)
        is_swing_low = all(center.low <= c.low for c in window)

        if is_swing_high:
            swing_highs.append((i, center.high))
        if is_swing_low:
            swing_lows.append((i, center.low))

    return swing_highs, swing_lows


class LiquidityZonesIndicator(IndicatorBase):
    """
    Liquidity Zones detection indicator.

    Identifies areas where stop losses are likely clustered:
    - Above swing highs (buy-side liquidity)
    - Below swing lows (sell-side liquidity)

    Smart money often targets these zones before reversing.

    Signals:
    - Price approaching liquidity zone: Potential reversal
    - Price swept liquidity: Confirmation of reversal
    """
    name = "liquidity_zones"
    category = "structure"
    timeframes_required = ["1m"]

    def compute(self, candles_by_tf: Dict[str, List[Candle]]) -> IndicatorResult:
        candles = candles_by_tf.get(self.timeframes_required[0], [])
        lookback = int(self.params.get("lookback", 50))
        pivot_lookback = int(self.params.get("pivot_lookback", 3))
        sweep_tolerance_pct = float(self.params.get("sweep_tolerance_pct", 0.001))  # 0.1%

        if len(candles) < lookback:
            return IndicatorResult(
                self.name, self.category, self.timeframes_required[0],
                {}, SignalState.NEUTRAL, 0, "insufficient data", self.weight
            )

        window = candles[-lookback:]
        current_candle = candles[-1]
        current_price = current_candle.close
        current_high = current_candle.high
        current_low = current_candle.low

        swing_highs, swing_lows = _find_swing_points(window, pivot_lookback)

        # Buy-side liquidity (above swing highs)
        buy_side_liquidity = []
        for idx, high in swing_highs:
            # Check if this high hasn't been taken out yet (excluding current candle)
            not_taken = all(c.high < high for c in window[idx + 1:-1]) if idx < len(window) - 2 else True
            if not_taken:
                buy_side_liquidity.append(high)

        # Sell-side liquidity (below swing lows)
        sell_side_liquidity = []
        for idx, low in swing_lows:
            not_taken = all(c.low > low for c in window[idx + 1:-1]) if idx < len(window) - 2 else True
            if not_taken:
                sell_side_liquidity.append(low)

        # Find nearest liquidity zones
        nearest_buy_liq = min([h for h in buy_side_liquidity if h > current_price], default=None)
        nearest_sell_liq = max([l for l in sell_side_liquidity if l < current_price], default=None)

        # Detect liquidity sweeps
        buy_side_swept = False
        sell_side_swept = False

        for high in buy_side_liquidity:
            if current_high > high and current_price < high:
                buy_side_swept = True
                break

        for low in sell_side_liquidity:
            if current_low < low and current_price > low:
                sell_side_swept = True
                break

        # Calculate proximity
        distance_to_buy_liq = float('inf')
        distance_to_sell_liq = float('inf')

        if nearest_buy_liq:
            distance_to_buy_liq = (nearest_buy_liq - current_price) / current_price

        if nearest_sell_liq:
            distance_to_sell_liq = (current_price - nearest_sell_liq) / current_price

        approaching_buy_liq = distance_to_buy_liq < 0.005  # Within 0.5%
        approaching_sell_liq = distance_to_sell_liq < 0.005

        state = SignalState.NEUTRAL
        reason = "no liquidity event"
        confidence = 40.0

        if sell_side_swept:
            state = SignalState.BUY
            reason = "sell-side liquidity swept (bullish reversal)"
            confidence = 80.0
        elif buy_side_swept:
            state = SignalState.SELL
            reason = "buy-side liquidity swept (bearish reversal)"
            confidence = 80.0
        elif approaching_sell_liq:
            state = SignalState.BUY
            reason = f"approaching sell-side liquidity ({nearest_sell_liq:.2f})"
            confidence = 60.0
        elif approaching_buy_liq:
            state = SignalState.SELL
            reason = f"approaching buy-side liquidity ({nearest_buy_liq:.2f})"
            confidence = 60.0
        elif nearest_sell_liq and distance_to_sell_liq < distance_to_buy_liq:
            state = SignalState.BUY
            reason = "closer to sell-side liquidity"
            confidence = 45.0
        elif nearest_buy_liq:
            state = SignalState.SELL
            reason = "closer to buy-side liquidity"
            confidence = 45.0

        value = {
            "buy_side_liquidity": [round(h, 4) for h in sorted(buy_side_liquidity, reverse=True)[:3]],
            "sell_side_liquidity": [round(l, 4) for l in sorted(sell_side_liquidity)[:3]],
            "nearest_buy_liq": round(nearest_buy_liq, 4) if nearest_buy_liq else None,
            "nearest_sell_liq": round(nearest_sell_liq, 4) if nearest_sell_liq else None,
            "price": current_price,
            "buy_side_swept": buy_side_swept,
            "sell_side_swept": sell_side_swept,
            "approaching_buy_liq": approaching_buy_liq,
            "approaching_sell_liq": approaching_sell_liq,
        }
        return IndicatorResult(
            self.name, self.category, self.timeframes_required[0],
            value, state, min(100, confidence), reason, self.weight
        )
