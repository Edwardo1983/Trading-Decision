from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from threading import Thread
from typing import Any, Dict, List, Optional

from core.models import Candle, EngineState, IndicatorResult, MarketRegime, RunnerSnapshot
from core.utils.config_loader import load_config
from core.utils.paths import project_root
from core.utils.ring_buffer import RingBuffer
from core.utils.time_utils import now_tz
from core.utils.sound_alert import play_sound
from data.binance.mapper import ws_to_candle as binance_ws_to_candle
from data.binance.ws import BinanceWSClient
from data.mexc.mapper import ws_to_candle as mexc_ws_to_candle
from data.mexc.ws import MexcWSClient
from data.sentiment import SentimentClient
from data.provider import BinanceProvider, MexcProvider
from data.export.csv_logger import CSVMinuteLogger
from engine.aggregator import aggregate
from engine.daily_regime import classify_regime
from engine.event_bus import EventBus
from engine.state import RunnerState
from indicators.registry import IndicatorRegistry, load_indicators
from ml.inference import load_model_or_none, run_live_inference

logger = logging.getLogger(__name__)

try:
    from ml.claude_analyzer import PromptGenerator
except Exception:  # pragma: no cover - optional dependency path issues should not break runner
    PromptGenerator = None


class Runner:
    def __init__(self, config: Dict):
        self.config = config
        self.state = RunnerState()
        self.event_bus = EventBus()
        self._running = False
        self._thread: Optional[Thread] = None

        app = config.get("app", {})
        self.symbol = app.get("symbol", "BTCUSDT")
        self.timeframes = app.get("timeframes", ["1m"])
        self.refresh_seconds = int(app.get("refresh_seconds", 60))
        self.buffer_size = int(app.get("buffer_size", 500))
        self.timezone = app.get("timezone", "UTC")

        data = config.get("data", {})
        self.rest_limit = int(data.get("rest_limit", 200))

        self.buffers: Dict[str, RingBuffer[Candle]] = {
            tf: RingBuffer(self.buffer_size) for tf in self.timeframes
        }

        self._ws_client: Optional[object] = None
        self._ws_task: Optional[asyncio.Task] = None

        data_cfg = config.get("data", {})
        self.provider_name = str(data_cfg.get("provider", "auto")).lower()
        binance_cfg = config.get("binance", {})
        mexc_cfg = config.get("mexc", {})
        market_type = app.get("market_type", "spot")
        testnet = bool(binance_cfg.get("testnet", False))
        self._binance_provider = BinanceProvider(
            api_key=binance_cfg.get("api_key", ""),
            api_secret=binance_cfg.get("api_secret", ""),
            market_type=market_type,
            testnet=testnet,
        )
        self._mexc_provider = MexcProvider(
            api_key=mexc_cfg.get("api_key", ""),
            api_secret=mexc_cfg.get("api_secret", ""),
        )
        self.provider = self._binance_provider if self.provider_name in ("auto", "binance") else self._mexc_provider

        self.sentiment_client: Optional[SentimentClient] = None

        load_indicators()
        indicator_params = config.get("indicator_params", {})
        astro_cfg = config.get("astro", {})
        if astro_cfg:
            indicator_params = dict(indicator_params)
            indicator_params.setdefault("astro_calendar", {}).update(astro_cfg)
        self.indicators = IndicatorRegistry.create_all(
            params=indicator_params,
            weights=config.get("indicator_weights", {}),
        )

        csv_cfg = config.get("csv", {})
        csv_base_path = Path(str(csv_cfg.get("base_path", "logs")))
        if not csv_base_path.is_absolute():
            csv_base_path = project_root() / csv_base_path
        self.csv_logger = CSVMinuteLogger(
            base_path=str(csv_base_path),
            timezone=self.timezone,
            rotate_daily=bool(csv_cfg.get("rotate_daily", True)),
        )

        prompt_cfg = config.get("prompt_generator", {})
        self.prompt_generator: Optional[Any] = None
        self.prompt_interval_minutes = int(prompt_cfg.get("interval_minutes", 60))
        self.prompt_auto_save = bool(prompt_cfg.get("auto_save", True))
        if bool(prompt_cfg.get("enabled", False)):
            if PromptGenerator is None:
                logger.warning("Prompt generator enabled, but PromptGenerator could not be imported.")
            else:
                output_dir = Path(str(prompt_cfg.get("output_dir", "prompts")))
                if not output_dir.is_absolute():
                    output_dir = project_root() / output_dir
                output_dir = output_dir / self.symbol
                lookback = int(prompt_cfg.get("lookback_candles", 21))
                targets = list(prompt_cfg.get("targets", ["claude", "codex"]))
                self.prompt_generator = PromptGenerator(output_dir=str(output_dir), lookback=lookback, targets=targets)

        ml_cfg = config.get("ml", {})
        self.ml_enabled = bool(ml_cfg.get("enabled", False))
        self.ml_lookback = int(ml_cfg.get("lookback", 21))
        self.ml_event_threshold = float(ml_cfg.get("event_confidence", 0.7))
        self.ml_model_path = Path(str(ml_cfg.get("model_path", "assets/models/ml_signal_model.npz")))
        if not self.ml_model_path.is_absolute():
            self.ml_model_path = project_root() / self.ml_model_path
        self.ml_indicator_names = list(config.get("csv", {}).get("include_indicators", []))
        if not self.ml_indicator_names:
            self.ml_indicator_names = sorted(config.get("indicator_weights", {}).keys())
        self.ml_model = load_model_or_none(self.ml_model_path) if self.ml_enabled else None
        self._ml_missing_warned = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = Thread(target=self.run_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._ws_client:
            self._ws_client.stop()

    def run_forever(self, stop_flag: Optional[Path] = None) -> None:
        self.state.state = EngineState.RUNNING
        try:
            asyncio.run(self._run_loop(stop_flag))
        except Exception as exc:
            logger.exception("Runner error: %s", exc)
            self.state.state = EngineState.ERROR
            self.state.errors.append(str(exc))
        finally:
            self.state.state = EngineState.STOPPED
            self._running = False

    async def _run_loop(self, stop_flag: Optional[Path]) -> None:
        await self._bootstrap_buffers()
        await self._start_ws()

        while self._running:
            if stop_flag and stop_flag.exists():
                self.event_bus.publish("info", "Stop flag detected")
                break
            try:
                await self._update_from_rest()
                await self._compute_cycle()
            except Exception as exc:
                logger.exception("Loop error: %s", exc)
                self.event_bus.publish("error", "Loop error", {"error": str(exc)})
                self.state.errors.append(str(exc))
            await self._sleep_until_next_cycle(stop_flag)

        await self._stop_ws()

    async def _sleep_until_next_cycle(self, stop_flag: Optional[Path]) -> None:
        sleep_seconds = max(1, int(self.refresh_seconds))
        for _ in range(sleep_seconds):
            if not self._running:
                break
            if stop_flag and stop_flag.exists():
                break
            await asyncio.sleep(1)

    async def _bootstrap_buffers(self) -> None:
        await self._select_provider()
        self._init_sentiment()
        await self._update_from_rest(initial=True)
        self.event_bus.publish("info", "Buffers initialized")

    async def _select_provider(self) -> None:
        if self.provider_name != "auto":
            return
        try:
            if await self.provider.ping():
                return
        except Exception:
            pass
        try:
            if await self._mexc_provider.ping():
                self.provider = self._mexc_provider
                self.event_bus.publish("info", "Provider switched to MEXC")
                return
        except Exception:
            pass
        self.event_bus.publish("warning", "Provider auto: using Binance (fallback)")

    def _init_sentiment(self) -> None:
        sentiment_cfg = self.config.get("sentiment", {})
        if sentiment_cfg.get("enabled", True):
            self.sentiment_client = SentimentClient(
                provider=self.provider,
                refresh_seconds=int(sentiment_cfg.get("refresh_seconds", 300)),
                long_short_period=str(sentiment_cfg.get("long_short_period", "5m")),
                fear_greed_enabled=bool(sentiment_cfg.get("fear_greed_enabled", True)),
                spot_trade_depth=int(sentiment_cfg.get("spot_trade_depth", 0)),
            )

    async def _update_from_rest(self, initial: bool = False) -> None:
        tasks = []
        for tf in self.timeframes:
            tasks.append(self.provider.get_candles(self.symbol, tf, limit=self.rest_limit))
        results = await asyncio.gather(*tasks)

        for tf, candles in zip(self.timeframes, results):
            buffer = self.buffers[tf]
            if initial:
                buffer.extend(candles)
            else:
                # Append only new candles
                last_ts = buffer.last().timestamp if buffer.last() else None
                for c in candles:
                    if last_ts is None or c.timestamp > last_ts:
                        buffer.append(c)

    async def _start_ws(self) -> None:
        app = self.config.get("app", {})
        if not app.get("use_ws", True):
            return
        if not getattr(self.provider, "supports_ws", False):
            return
        ws_tf = app.get("ws_timeframe", "1m")
        if ws_tf not in self.timeframes:
            return
        provider_name = getattr(self.provider, "name", "binance")
        reconnect = self.config.get("data", {}).get("ws_reconnect_seconds", 5)
        if provider_name == "mexc":
            self._ws_client = MexcWSClient(self.symbol, ws_tf, reconnect_seconds=reconnect)
            self._ws_task = asyncio.create_task(self._ws_listener(ws_tf, provider_name))
            return
        market_type = self.config.get("app", {}).get("market_type", "spot")
        self._ws_client = BinanceWSClient(
            self.symbol,
            ws_tf,
            reconnect_seconds=reconnect,
            market_type=market_type,
        )
        self._ws_task = asyncio.create_task(self._ws_listener(ws_tf, "binance"))

    async def _stop_ws(self) -> None:
        if self._ws_client:
            self._ws_client.stop()
        if self._ws_task:
            self._ws_task.cancel()

    async def _ws_listener(self, timeframe: str, provider_name: str) -> None:
        if not self._ws_client:
            return
        if provider_name == "mexc":
            async for payload in self._ws_client.listen():
                candle = mexc_ws_to_candle(payload)
                if candle is None:
                    continue
                buffer = self.buffers[timeframe]
                last = buffer.last()
                if last and candle.timestamp == last.timestamp:
                    buffer.replace_last(candle)
                else:
                    buffer.append(candle)
                    self.event_bus.publish("info", "WS candle update", {"tf": timeframe, "provider": "mexc"})
            return
        async for payload in self._ws_client.listen():
            candle = binance_ws_to_candle(payload)
            # Binance marks closed candles with x=true
            closed = payload.get("k", {}).get("x", False)
            if closed:
                buffer = self.buffers[timeframe]
                last = buffer.last()
                if last and candle.timestamp < last.timestamp:
                    continue
                action = "append"
                if last and candle.timestamp == last.timestamp:
                    buffer.replace_last(candle)
                    action = "replace"
                else:
                    buffer.append(candle)
                self.event_bus.publish(
                    "info",
                    "WS candle closed",
                    {"tf": timeframe, "provider": "binance", "action": action},
                )

    def _resolve_regime_timeframe(self, candles_by_tf: Dict[str, List[Candle]]) -> str:
        regime_cfg = self.config.get("daily_regime", {})
        preferred = str(regime_cfg.get("timeframe", "")).strip()
        if preferred and preferred in candles_by_tf:
            return preferred
        if "1h" in candles_by_tf:
            return "1h"
        if self.timeframes:
            fallback = self.timeframes[-1]
            if fallback in candles_by_tf:
                return fallback
        return next(iter(candles_by_tf.keys()), "1m")

    async def _compute_cycle(self) -> None:
        candles_by_tf = {tf: buffer.to_list() for tf, buffer in self.buffers.items()}
        results: List[IndicatorResult] = []
        if self.sentiment_client:
            try:
                self.state.sentiment = await self.sentiment_client.refresh(self.symbol)
            except Exception as exc:
                logger.debug("Sentiment refresh failed: %s", exc)
        if self.state.sentiment:
            candles_by_tf["sentiment"] = [self.state.sentiment]
        for indicator in self.indicators:
            try:
                results.append(indicator.compute(candles_by_tf))
            except Exception as exc:
                logger.debug("Indicator error %s: %s", indicator.name, exc)
                self.event_bus.publish("error", f"Indicator {indicator.name} failed", {"error": str(exc)})

        base_tf = self.timeframes[0]
        regime_tf = self._resolve_regime_timeframe(candles_by_tf)
        market_regime = classify_regime(candles_by_tf.get(regime_tf, []), self.config)
        agg = aggregate(results, self.config, market_regime)

        self.state.last_update = now_tz(self.timezone)
        self.state.indicators = results
        self.state.aggregate = agg
        self.state.market_regime = market_regime
        self.state.ml_result = self._maybe_run_ml(candles_by_tf, results)

        if agg.alignment:
            self.event_bus.publish("info", "Alignment reached", {"state": agg.final_state.value})

        alerts = self.config.get("alerts", {})
        if alerts.get("enabled", False) and agg:
            max_conf = max(agg.buy_pct, agg.sell_pct)
            if max_conf >= float(alerts.get("min_confidence", 70)):
                play_sound(alerts.get("sound_file", ""))

        if candles_by_tf.get(base_tf):
            last = candles_by_tf[base_tf][-1]
            self.state.last_ohlcv = {
                "open": last.open,
                "high": last.high,
                "low": last.low,
                "close": last.close,
                "volume": last.volume,
            }

        # CSV logging
        csv_cfg = self.config.get("csv", {})
        if csv_cfg.get("enabled", True) and candles_by_tf.get(base_tf):
            self.csv_logger.log(
                symbol=self.symbol,
                timeframe=base_tf,
                ohlcv=self.state.last_ohlcv,
                indicators=results,
                aggregate=agg,
                include_indicators=csv_cfg.get("include_indicators", []),
                market_regime=market_regime,
                sentiment=self.state.sentiment,
                ml_result=self.state.ml_result,
            )

        self._maybe_generate_prompt(candles_by_tf, results, market_regime)

    def _maybe_run_ml(
        self,
        candles_by_tf: Dict[str, List[Candle]],
        indicators: List[IndicatorResult],
    ) -> Dict[str, Any]:
        if not self.ml_enabled:
            return {}
        if self.ml_model is None:
            if not self._ml_missing_warned:
                self._ml_missing_warned = True
                self.event_bus.publish(
                    "warning",
                    "ML enabled but model file missing/unreadable",
                    {"path": str(self.ml_model_path)},
                )
            return {}
        base_tf = self.timeframes[0]
        candles = candles_by_tf.get(base_tf, [])
        closes = [c.close for c in candles]
        if len(closes) < 3:
            return {}
        try:
            result = run_live_inference(
                closes=closes,
                indicators=indicators,
                indicator_names=self.ml_indicator_names,
                model=self.ml_model,
                lookback=self.ml_lookback,
            )
            payload = {
                "label": result.label,
                "confidence": round(float(result.confidence), 4),
                "probability_up": round(float(result.probability_up), 4),
                "buy_threshold": round(float(result.buy_threshold), 4),
                "sell_threshold": round(float(result.sell_threshold), 4),
            }
            if result.confidence >= self.ml_event_threshold:
                self.event_bus.publish("info", "ML advisory update", payload)
            return payload
        except Exception as exc:
            self.event_bus.publish("error", "ML inference failed", {"error": str(exc)})
            return {}

    def _extract_patterns(self, indicators: List[IndicatorResult]) -> List[Dict[str, Any]]:
        pattern_indicator = next((item for item in indicators if item.name == "pattern_detector"), None)
        if pattern_indicator is None:
            return []
        value = pattern_indicator.value
        if isinstance(value, dict):
            patterns = value.get("patterns")
            if isinstance(patterns, list):
                return [item for item in patterns if isinstance(item, dict)]
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        return []

    def _maybe_generate_prompt(
        self,
        candles_by_tf: Dict[str, List[Candle]],
        indicators: List[IndicatorResult],
        market_regime: MarketRegime,
    ) -> None:
        if not self.prompt_generator:
            return
        if not self.prompt_generator.should_refresh(self.prompt_interval_minutes):
            return
        base_tf = self.timeframes[0]
        candles = candles_by_tf.get(base_tf, [])
        if not candles:
            return
        try:
            prompts = self.prompt_generator.generate_prompt_bundle(
                symbol=self.symbol,
                candles=candles,
                indicators=indicators,
                sentiment=self.state.sentiment,
                market_regime=market_regime,
                day_classification=market_regime.value,
                patterns=self._extract_patterns(indicators),
                save_to_file=self.prompt_auto_save,
            )
            latest_path = self.prompt_generator.output_dir / "latest_prompt.txt"
            self.event_bus.publish(
                "info",
                "Prompt generated for Claude/Codex",
                {
                    "path": str(latest_path),
                    "targets": list(prompts.keys()),
                },
            )
        except Exception as exc:
            logger.exception("Prompt generation failed: %s", exc)
            self.event_bus.publish("error", "Prompt generation failed", {"error": str(exc)})

    def get_snapshot(self) -> RunnerSnapshot:
        return RunnerSnapshot(
            state=self.state.state,
            symbol=self.symbol,
            timeframes=self.timeframes,
            last_update=self.state.last_update,
            indicators=self.state.indicators,
            aggregate=self.state.aggregate,
            market_regime=self.state.market_regime,
            sentiment=self.state.sentiment,
            events=self.event_bus.recent(),
            errors=self.state.errors,
            ml_result=self.state.ml_result,
            last_ohlcv=self.state.last_ohlcv,
        )


def create_runner_from_config() -> Runner:
    return Runner(load_config())
