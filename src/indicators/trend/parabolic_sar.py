from __future__ import annotations

from typing import Dict, List

from core.models import Candle, IndicatorResult, SignalState
from indicators.base import IndicatorBase


class ParabolicSARIndicator(IndicatorBase):
    """
    Parabolic SAR (Stop and Reverse).
    A trend-following indicator that provides potential entry and exit points.

    The SAR trails price as the trend extends:
    - In an uptrend, SAR is below price
    - In a downtrend, SAR is above price
    - When price crosses SAR, trend reverses

    Parameters:
    - start: Initial acceleration factor (default 0.02)
    - increment: AF increment (default 0.02)
    - maximum: Maximum AF (default 0.2)
    """
    name = "parabolic_sar"
    category = "trend"
    timeframes_required = ["1m"]

    def compute(self, candles_by_tf: Dict[str, List[Candle]]) -> IndicatorResult:
        candles = candles_by_tf.get(self.timeframes_required[0], [])
        start = float(self.params.get("start", 0.02))
        increment = float(self.params.get("increment", 0.02))
        maximum = float(self.params.get("maximum", 0.2))

        if len(candles) < 10:
            return IndicatorResult(
                self.name, self.category, self.timeframes_required[0],
                {}, SignalState.NEUTRAL, 0, "insufficient data", self.weight
            )

        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        closes = [c.close for c in candles]

        # Initialize
        sar = [0.0] * len(candles)
        ep = [0.0] * len(candles)  # Extreme point
        af = [start] * len(candles)  # Acceleration factor
        trend = [1] * len(candles)  # 1 for uptrend, -1 for downtrend

        # Determine initial trend from first two candles
        if closes[1] > closes[0]:
            trend[0] = trend[1] = 1
            sar[0] = sar[1] = lows[0]
            ep[0] = ep[1] = highs[1]
        else:
            trend[0] = trend[1] = -1
            sar[0] = sar[1] = highs[0]
            ep[0] = ep[1] = lows[1]

        for i in range(2, len(candles)):
            if trend[i - 1] == 1:  # Uptrend
                sar[i] = sar[i - 1] + af[i - 1] * (ep[i - 1] - sar[i - 1])
                sar[i] = min(sar[i], lows[i - 1], lows[i - 2])

                if lows[i] < sar[i]:
                    # Trend reversal to downtrend
                    trend[i] = -1
                    sar[i] = ep[i - 1]
                    ep[i] = lows[i]
                    af[i] = start
                else:
                    trend[i] = 1
                    if highs[i] > ep[i - 1]:
                        ep[i] = highs[i]
                        af[i] = min(af[i - 1] + increment, maximum)
                    else:
                        ep[i] = ep[i - 1]
                        af[i] = af[i - 1]
            else:  # Downtrend
                sar[i] = sar[i - 1] + af[i - 1] * (ep[i - 1] - sar[i - 1])
                sar[i] = max(sar[i], highs[i - 1], highs[i - 2])

                if highs[i] > sar[i]:
                    # Trend reversal to uptrend
                    trend[i] = 1
                    sar[i] = ep[i - 1]
                    ep[i] = highs[i]
                    af[i] = start
                else:
                    trend[i] = -1
                    if lows[i] < ep[i - 1]:
                        ep[i] = lows[i]
                        af[i] = min(af[i - 1] + increment, maximum)
                    else:
                        ep[i] = ep[i - 1]
                        af[i] = af[i - 1]

        current_sar = sar[-1]
        current_trend = trend[-1]
        prev_trend = trend[-2]
        current_price = closes[-1]

        # Detect reversal
        trend_reversal = current_trend != prev_trend
        bullish_reversal = trend_reversal and current_trend == 1
        bearish_reversal = trend_reversal and current_trend == -1

        state = SignalState.NEUTRAL
        reason = "SAR neutral"
        confidence = 40.0

        if bullish_reversal:
            state = SignalState.BUY
            reason = "SAR bullish reversal"
            confidence = 80.0
        elif bearish_reversal:
            state = SignalState.SELL
            reason = "SAR bearish reversal"
            confidence = 80.0
        elif current_trend == 1:
            state = SignalState.BUY
            reason = "SAR uptrend"
            confidence = 60.0
        else:
            state = SignalState.SELL
            reason = "SAR downtrend"
            confidence = 60.0

        value = {
            "sar": round(current_sar, 4),
            "trend": "up" if current_trend == 1 else "down",
            "price": current_price,
            "af": round(af[-1], 4),
            "ep": round(ep[-1], 4),
            "reversal": trend_reversal,
        }
        return IndicatorResult(
            self.name, self.category, self.timeframes_required[0],
            value, state, min(100, confidence), reason, self.weight
        )
