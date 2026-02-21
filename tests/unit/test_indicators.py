from __future__ import annotations

import pytest

from core.models import SignalState
from indicators.price_action.candle_efficiency import CandleEfficiencyIndicator
from indicators.price_action.prev_candle_break import PreviousCandleBreakIndicator
from indicators.trend.ema_bias import EMABiasIndicator
from indicators.trend.market_structure import MarketStructureIndicator
from indicators.structure.smart_money import SmartMoneyIndicator
from indicators.trend.adx import ADXIndicator
from indicators.trend.supertrend import SupertrendIndicator
from indicators.momentum.rsi_state import RSIStateIndicator
from indicators.momentum.roc_impulse import ROCImpulseIndicator
from indicators.momentum.macd import MACDIndicator
from indicators.momentum.stochastic_rsi import StochasticRSIIndicator
from indicators.volatility.atr_regime import ATRRegimeIndicator
from indicators.volatility.bb_squeeze_expand import BBSqueezeExpandIndicator
from indicators.volume.vwap_bias import VWAPBiasIndicator
from indicators.volume.obv_flow import OBVFlowIndicator
from indicators.volume.cvd import CVDIndicator
from indicators.context.htf_conflict_detector import HTFConflictDetector
from indicators.context.levels_daily_weekly import LevelsDailyWeeklyIndicator
from indicators.context.darvas_box import DarvasBoxIndicator
from tests.utils import make_candles


def test_candle_efficiency_buy():
    candles = make_candles([100, 105, 110])
    ind = CandleEfficiencyIndicator({"bullish_ratio": 0.5}, 1.0)
    res = ind.compute({"1m": candles})
    assert res.state in (SignalState.BUY, SignalState.NEUTRAL)


def test_prev_candle_break():
    candles = make_candles([100, 101, 110])
    ind = PreviousCandleBreakIndicator({}, 1.0)
    res = ind.compute({"1m": candles})
    assert res.state in (SignalState.BUY, SignalState.NEUTRAL, SignalState.SELL)


def test_ema_bias_trend():
    candles = make_candles(list(range(100, 180)))
    ind = EMABiasIndicator({"period": 10}, 1.0)
    res = ind.compute({"1m": candles})
    assert res.state in (SignalState.BUY, SignalState.NEUTRAL, SignalState.SELL)


def test_market_structure():
    prices = [100, 105, 102, 108, 104, 112, 108, 116, 112, 120, 118, 122]
    candles = make_candles(prices)
    ind = MarketStructureIndicator({"lookback": 10, "pivot_lookback": 1}, 1.0)
    res = ind.compute({"1m": candles})
    assert res.state in (SignalState.BUY, SignalState.SELL, SignalState.NEUTRAL)


def test_smart_money_indicator():
    prices = [100, 101, 99, 102, 98, 104, 97, 106, 105, 107, 103, 108, 102, 110, 109, 111, 108, 112, 107, 113]
    candles = make_candles(prices)
    ind = SmartMoneyIndicator({"lookback": 15, "pivot_lookback": 1, "atr_period": 5}, 1.0)
    res = ind.compute({"1m": candles})
    assert "bos" in res.value


def test_adx_indicator():
    candles = make_candles([100 + i for i in range(80)])
    ind = ADXIndicator({"period": 14, "threshold": 20}, 1.0)
    res = ind.compute({"1m": candles})
    assert "adx" in res.value


def test_supertrend_indicator():
    candles = make_candles([100 + i for i in range(80)])
    ind = SupertrendIndicator({"period": 10, "multiplier": 3.0}, 1.0)
    res = ind.compute({"1m": candles})
    assert res.state in (SignalState.BUY, SignalState.SELL)


def test_rsi_state():
    candles = make_candles([100 + i for i in range(30)])
    ind = RSIStateIndicator({"period": 14}, 1.0)
    res = ind.compute({"1m": candles})
    assert res.state in (SignalState.BUY, SignalState.SELL, SignalState.NEUTRAL)


def test_roc_impulse():
    candles = make_candles([100, 102, 105, 107, 110, 115, 117, 120, 123, 130])
    ind = ROCImpulseIndicator({"period": 3, "threshold": 0.5}, 1.0)
    res = ind.compute({"1m": candles})
    assert res.state in (SignalState.BUY, SignalState.SELL, SignalState.NEUTRAL)


def test_macd_indicator():
    candles = make_candles([100 + i for i in range(80)])
    ind = MACDIndicator({"fast": 12, "slow": 26, "signal": 9}, 1.0)
    res = ind.compute({"1m": candles})
    assert res.state in (SignalState.BUY, SignalState.SELL, SignalState.NEUTRAL)


def test_stochastic_rsi_indicator():
    candles = make_candles([100 + (i % 5) for i in range(120)])
    ind = StochasticRSIIndicator({"rsi_period": 14, "stoch_period": 14}, 1.0)
    res = ind.compute({"1m": candles})
    assert res.state in (SignalState.BUY, SignalState.SELL, SignalState.NEUTRAL)


def test_atr_regime():
    candles = make_candles([100 + i for i in range(60)])
    ind = ATRRegimeIndicator({"short": 5, "long": 10}, 1.0)
    res = ind.compute({"1m": candles})
    assert "ratio" in res.value


def test_bb_squeeze_expand():
    candles = make_candles([100] * 30)
    ind = BBSqueezeExpandIndicator({"period": 10, "squeeze_threshold": 0.2}, 1.0)
    res = ind.compute({"1m": candles})
    assert res.state in (SignalState.BUY, SignalState.SELL, SignalState.NEUTRAL)


def test_vwap_bias():
    candles = make_candles([100, 101, 103, 105, 107, 109])
    ind = VWAPBiasIndicator({"distance_threshold": 0.0001}, 1.0)
    res = ind.compute({"1m": candles})
    assert res.state in (SignalState.BUY, SignalState.SELL, SignalState.NEUTRAL)


def test_obv_flow():
    candles = make_candles([100, 102, 104, 103, 105, 107, 110])
    ind = OBVFlowIndicator({"slope_lookback": 3}, 1.0)
    res = ind.compute({"1m": candles})
    assert res.state in (SignalState.BUY, SignalState.SELL, SignalState.NEUTRAL)


def test_cvd_indicator():
    candles = make_candles([100 + i for i in range(80)])
    ind = CVDIndicator({"period": 10, "slope_lookback": 5}, 1.0)
    res = ind.compute({"1m": candles})
    assert res.state in (SignalState.BUY, SignalState.SELL, SignalState.NEUTRAL)


def test_htf_conflict():
    ltf = make_candles([100, 102, 104, 106, 108, 110, 112, 114])
    htf = make_candles([200, 198, 196, 194, 192, 190, 188, 186])
    ind = HTFConflictDetector({"ltf": "5m", "htf": "1h", "ema_period": 3}, 1.0)
    res = ind.compute({"5m": ltf, "1h": htf})
    assert res.state in (SignalState.NO_TRADE, SignalState.NEUTRAL)


def test_levels_daily_weekly():
    candles = make_candles([100 + i for i in range(30)])
    ind = LevelsDailyWeeklyIndicator({"distance_pct": 0.5}, 1.0)
    res = ind.compute({"1m": candles})
    assert "day_high" in res.value


def test_darvas_box():
    prices = [100, 105, 102, 108, 104, 110, 106, 112, 108, 115, 110, 118, 112, 120, 116, 122, 118, 124, 120, 126, 122, 128, 124, 130, 126]
    candles = make_candles(prices)
    ind = DarvasBoxIndicator({"pivot_lookback": 1, "min_bars_in_box": 2}, 1.0)
    res = ind.compute({"1m": candles})
    assert "box_top" in res.value
    assert "breakout_confirmed" in res.value
    assert "confirmation_bars" in res.value


def test_astro_calendar_optional():
    skyfield = pytest.importorskip("skyfield")
    # Skip if ephemeris missing to avoid network
    from core.utils.paths import assets_dir
    eph = assets_dir() / "ephemeris" / "de421.bsp"
    if not eph.exists():
        pytest.skip("Ephemeris not present")
    from indicators.context.astro_calendar import AstroCalendarIndicator
    ind = AstroCalendarIndicator({"aspects_enabled": False, "allow_directional_bias": False}, 1.0)
    res = ind.compute({"1m": []})
    assert res.state == SignalState.NEUTRAL
    assert res.meta.get("experimental") is True


# New indicator tests

def test_sma_indicator():
    from indicators.trend.sma import SMAIndicator
    candles = make_candles([100 + i for i in range(250)])
    ind = SMAIndicator({"periods": [20, 50]}, 1.0)
    res = ind.compute({"1m": candles})
    assert "sma_20" in res.value
    assert res.state in (SignalState.BUY, SignalState.SELL, SignalState.NEUTRAL)


def test_vwma_indicator():
    from indicators.trend.vwma import VWMAIndicator
    candles = make_candles([100 + i for i in range(50)])
    ind = VWMAIndicator({"period": 20}, 1.0)
    res = ind.compute({"1m": candles})
    assert "vwma" in res.value


def test_ichimoku_indicator():
    from indicators.trend.ichimoku import IchimokuIndicator
    candles = make_candles([100 + (i % 10) for i in range(120)])
    ind = IchimokuIndicator({"tenkan_period": 9, "kijun_period": 26}, 1.0)
    res = ind.compute({"1m": candles})
    assert "tenkan" in res.value


def test_parabolic_sar_indicator():
    from indicators.trend.parabolic_sar import ParabolicSARIndicator
    candles = make_candles([100 + i for i in range(50)])
    ind = ParabolicSARIndicator({"start": 0.02, "increment": 0.02, "maximum": 0.2}, 1.0)
    res = ind.compute({"1m": candles})
    assert "sar" in res.value


def test_cci_indicator():
    from indicators.momentum.cci import CCIIndicator
    candles = make_candles([100 + (i % 20) for i in range(80)])
    ind = CCIIndicator({"period": 20, "overbought": 100, "oversold": -100}, 1.0)
    res = ind.compute({"1m": candles})
    assert "cci" in res.value


def test_williams_r_indicator():
    from indicators.momentum.williams_r import WilliamsRIndicator
    candles = make_candles([100 + (i % 15) for i in range(50)])
    ind = WilliamsRIndicator({"period": 14}, 1.0)
    res = ind.compute({"1m": candles})
    assert "williams_r" in res.value


def test_mfi_indicator():
    from indicators.volume.mfi import MFIIndicator
    candles = make_candles([100 + i for i in range(50)])
    ind = MFIIndicator({"period": 14}, 1.0)
    res = ind.compute({"1m": candles})
    assert "mfi" in res.value


def test_cmf_indicator():
    from indicators.volume.cmf import CMFIndicator
    candles = make_candles([100 + i for i in range(50)])
    ind = CMFIndicator({"period": 20}, 1.0)
    res = ind.compute({"1m": candles})
    assert "cmf" in res.value


def test_volume_profile_indicator():
    from indicators.volume.volume_profile import VolumeProfileIndicator
    candles = make_candles([100 + (i % 10) for i in range(150)])
    ind = VolumeProfileIndicator({"lookback": 100, "num_bins": 20}, 1.0)
    res = ind.compute({"1m": candles})
    assert "poc" in res.value


def test_volume_oscillator_indicator():
    from indicators.volume.volume_oscillator import VolumeOscillatorIndicator
    candles = make_candles([100 + i for i in range(50)])
    ind = VolumeOscillatorIndicator({"short_period": 5, "long_period": 20}, 1.0)
    res = ind.compute({"1m": candles})
    assert "vo" in res.value


def test_keltner_channels_indicator():
    from indicators.volatility.keltner_channels import KeltnerChannelsIndicator
    candles = make_candles([100 + (i % 10) for i in range(50)])
    ind = KeltnerChannelsIndicator({"period": 20, "atr_period": 10, "multiplier": 1.5}, 1.0)
    res = ind.compute({"1m": candles})
    assert "upper" in res.value


def test_donchian_channels_indicator():
    from indicators.volatility.donchian_channels import DonchianChannelsIndicator
    candles = make_candles([100 + (i % 10) for i in range(50)])
    ind = DonchianChannelsIndicator({"period": 20}, 1.0)
    res = ind.compute({"1m": candles})
    assert "upper" in res.value


def test_pivot_points_indicator():
    from indicators.structure.pivot_points import PivotPointsIndicator
    candles = make_candles([100 + i for i in range(50)])
    ind = PivotPointsIndicator({"type": "standard", "lookback": 24}, 1.0)
    res = ind.compute({"1m": candles})
    assert "pivot" in res.value


def test_fibonacci_indicator():
    from indicators.structure.fibonacci import FibonacciIndicator
    candles = make_candles([100 + (i % 20) for i in range(80)])
    ind = FibonacciIndicator({"lookback": 50}, 1.0)
    res = ind.compute({"1m": candles})
    assert "swing_high" in res.value


def test_support_resistance_indicator():
    from indicators.structure.support_resistance import SupportResistanceIndicator
    prices = [100, 105, 102, 108, 104, 110, 106, 112, 108, 115, 110, 118]
    candles = make_candles(prices * 10)
    ind = SupportResistanceIndicator({"lookback": 100, "pivot_lookback": 2}, 1.0)
    res = ind.compute({"1m": candles})
    assert "price" in res.value


def test_liquidity_zones_indicator():
    from indicators.structure.liquidity_zones import LiquidityZonesIndicator
    prices = [100, 105, 102, 108, 104, 110, 106, 112, 108, 115, 110, 118]
    candles = make_candles(prices * 5)
    ind = LiquidityZonesIndicator({"lookback": 50, "pivot_lookback": 2}, 1.0)
    res = ind.compute({"1m": candles})
    assert "price" in res.value


def test_pattern_detector():
    from ml.pattern_detector import PatternDetector
    candles = make_candles([100, 105, 102, 108, 104, 110, 106, 112, 108, 115, 110, 118])
    detector = PatternDetector(lookback=12)
    patterns = detector.detect_all(candles)
    summary = detector.get_summary(candles)
    assert "patterns" in summary
    assert "bias" in summary


def test_pattern_detector_indicator():
    from indicators.price_action.pattern_detector import PatternDetectorIndicator
    candles = make_candles([100, 101, 99, 102, 98, 103, 100])
    ind = PatternDetectorIndicator({"lookback": 10}, 1.0)
    res = ind.compute({"1m": candles})
    assert res.state in (SignalState.BUY, SignalState.SELL, SignalState.NEUTRAL)


def test_sentiment_indicator():
    from indicators.context.sentiment_indicator import SentimentIndicator
    sentiment = {"fear_greed": 20, "long_short_ratio": 1.0, "funding_rate": -0.002}
    ind = SentimentIndicator({}, 1.0)
    res = ind.compute({"sentiment": [sentiment]})
    assert res.state in (SignalState.BUY, SignalState.SELL, SignalState.NEUTRAL)


def test_divergence_detector():
    from indicators.momentum.divergence_detector import DivergenceDetector
    candles = make_candles([100 + (i % 5) for i in range(120)])
    ind = DivergenceDetector({"lookback": 60}, 1.0)
    res = ind.compute({"1m": candles})
    assert res.state in (SignalState.BUY, SignalState.SELL, SignalState.NEUTRAL)
