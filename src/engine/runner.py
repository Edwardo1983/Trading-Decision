from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import Dict, List, Optional

from core.models import Candle, EngineState, IndicatorResult, MarketRegime, RunnerSnapshot
from core.utils.config_loader import load_config
from core.utils.ring_buffer import RingBuffer
from core.utils.time_utils import now_tz
from core.utils.sound_alert import play_sound
from data.binance.client import BinanceRESTClient
from data.binance.mapper import kline_to_candle, ws_to_candle
from data.binance.ws import BinanceWSClient
from data.sentiment import SentimentClient
from data.export.csv_logger import CSVMinuteLogger
from engine.aggregator import aggregate
from engine.daily_regime import classify_regime
from engine.event_bus import EventBus
from engine.state import RunnerState
from indicators.registry import IndicatorRegistry, load_indicators

logger = logging.getLogger(__name__)


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

        self._ws_client: Optional[BinanceWSClient] = None
        self._ws_task: Optional[asyncio.Task] = None

        api_key = config.get("binance", {}).get("api_key", "")
        api_secret = config.get("binance", {}).get("api_secret", "")
        market_type = app.get("market_type", "spot")
        testnet = bool(config.get("binance", {}).get("testnet", False))
        self.rest_client = BinanceRESTClient(api_key, api_secret, market_type, testnet)

        sentiment_cfg = config.get("sentiment", {})
        self.sentiment_client: Optional[SentimentClient] = None
        if sentiment_cfg.get("enabled", True):
            self.sentiment_client = SentimentClient(
                rest_client=self.rest_client,
                refresh_seconds=int(sentiment_cfg.get("refresh_seconds", 300)),
                long_short_period=str(sentiment_cfg.get("long_short_period", "5m")),
                fear_greed_enabled=bool(sentiment_cfg.get("fear_greed_enabled", True)),
                spot_trade_depth=int(sentiment_cfg.get("spot_trade_depth", 0)),
            )

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
        self.csv_logger = CSVMinuteLogger(
            base_path=csv_cfg.get("base_path", "logs"),
            timezone=self.timezone,
            rotate_daily=bool(csv_cfg.get("rotate_daily", True)),
        )

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
            await asyncio.sleep(self.refresh_seconds)

        await self._stop_ws()

    async def _bootstrap_buffers(self) -> None:
        await self._update_from_rest(initial=True)
        self.event_bus.publish("info", "Buffers initialized")

    async def _update_from_rest(self, initial: bool = False) -> None:
        tasks = []
        for tf in self.timeframes:
            tasks.append(self.rest_client.get_klines(self.symbol, tf, limit=self.rest_limit))
        results = await asyncio.gather(*tasks)

        for tf, klines in zip(self.timeframes, results):
            candles = [kline_to_candle(k) for k in klines]
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
        ws_tf = app.get("ws_timeframe", "1m")
        if ws_tf not in self.timeframes:
            return
        market_type = self.config.get("app", {}).get("market_type", "spot")
        self._ws_client = BinanceWSClient(
            self.symbol,
            ws_tf,
            reconnect_seconds=self.config.get("data", {}).get("ws_reconnect_seconds", 5),
            market_type=market_type,
        )
        self._ws_task = asyncio.create_task(self._ws_listener(ws_tf))

    async def _stop_ws(self) -> None:
        if self._ws_client:
            self._ws_client.stop()
        if self._ws_task:
            self._ws_task.cancel()

    async def _ws_listener(self, timeframe: str) -> None:
        if not self._ws_client:
            return
        async for payload in self._ws_client.listen():
            candle = ws_to_candle(payload)
            # Binance marks closed candles with x=true
            closed = payload.get("k", {}).get("x", False)
            if closed:
                self.buffers[timeframe].append(candle)
                self.event_bus.publish("info", "WS candle closed", {"tf": timeframe})

    async def _compute_cycle(self) -> None:
        candles_by_tf = {tf: buffer.to_list() for tf, buffer in self.buffers.items()}
        results: List[IndicatorResult] = []
        if self.sentiment_client:
            try:
                self.state.sentiment = await self.sentiment_client.refresh(self.symbol)
            except Exception as exc:
                logger.debug("Sentiment refresh failed: %s", exc)
        for indicator in self.indicators:
            try:
                results.append(indicator.compute(candles_by_tf))
            except Exception as exc:
                logger.debug("Indicator error %s: %s", indicator.name, exc)
                self.event_bus.publish("error", f"Indicator {indicator.name} failed", {"error": str(exc)})

        base_tf = self.timeframes[0]
        market_regime = classify_regime(candles_by_tf.get(base_tf, []), self.config)
        agg = aggregate(results, self.config, market_regime)

        self.state.last_update = now_tz(self.timezone)
        self.state.indicators = results
        self.state.aggregate = agg
        self.state.market_regime = market_regime

        if agg.alignment:
            self.event_bus.publish("info", "Alignment reached", {"state": agg.final_state.value})

        alerts = self.config.get("alerts", {})
        if alerts.get("enabled", False) and agg:
            max_conf = max(agg.buy_pct, agg.sell_pct)
            if max_conf >= float(alerts.get("min_confidence", 70)):
                play_sound(alerts.get("sound_file", ""))

        # CSV logging
        csv_cfg = self.config.get("csv", {})
        if csv_cfg.get("enabled", True) and candles_by_tf.get(base_tf):
            last = candles_by_tf[base_tf][-1]
            ohlcv = {
                "open": last.open,
                "high": last.high,
                "low": last.low,
                "close": last.close,
                "volume": last.volume,
            }
            self.csv_logger.log(
                symbol=self.symbol,
                timeframe=base_tf,
                ohlcv=ohlcv,
                indicators=results,
                aggregate=agg,
                include_indicators=csv_cfg.get("include_indicators", []),
                market_regime=market_regime,
                sentiment=self.state.sentiment,
            )

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
        )


def create_runner_from_config() -> Runner:
    return Runner(load_config())
