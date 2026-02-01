from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator, List

import websockets

logger = logging.getLogger(__name__)


def _tf_to_interval(timeframe: str) -> str:
    mapping = {
        "1m": "Min1",
        "5m": "Min5",
        "15m": "Min15",
        "30m": "Min30",
        "1h": "Min60",
        "4h": "Hour4",
        "8h": "Hour8",
        "1d": "Day1",
        "1w": "Week1",
        "1M": "Month1",
    }
    return mapping.get(timeframe, "Min1")


class MexcWSClient:
    def __init__(self, symbol: str, interval: str, reconnect_seconds: int = 5, ping_seconds: int = 20):
        self.symbol = symbol.upper()
        self.interval = _tf_to_interval(interval)
        self.reconnect_seconds = reconnect_seconds
        self.ping_seconds = ping_seconds
        self._running = False

    def _url(self) -> str:
        return "wss://wbs-api.mexc.com/ws"

    def _subscribe_payload(self) -> dict:
        channel = f"spot@public.kline.v3.api.pb@{self.symbol}@{self.interval}"
        return {
            "method": "SUBSCRIPTION",
            "params": [channel],
        }

    async def _ping_loop(self, ws: websockets.WebSocketClientProtocol) -> None:
        while self._running:
            try:
                await ws.send(json.dumps({"method": "PING"}))
            except Exception:
                return
            await asyncio.sleep(self.ping_seconds)

    async def listen(self) -> AsyncIterator[dict]:
        self._running = True
        while self._running:
            try:
                async with websockets.connect(self._url(), ping_interval=20, ping_timeout=20) as ws:
                    payload = self._subscribe_payload()
                    await ws.send(json.dumps(payload))
                    logger.info("MEXC WS subscribed: %s", payload)
                    ping_task = asyncio.create_task(self._ping_loop(ws))
                    try:
                        async for msg in ws:
                            if not self._running:
                                break
                            if isinstance(msg, bytes):
                                try:
                                    msg = msg.decode("utf-8")
                                except Exception:
                                    logger.debug("MEXC WS binary frame received")
                                    continue
                            try:
                                data = json.loads(msg)
                            except Exception:
                                continue
                            yield data
                    finally:
                        ping_task.cancel()
            except Exception as exc:
                logger.warning("MEXC WS disconnected: %s", exc)
                await asyncio.sleep(self.reconnect_seconds)

    def stop(self) -> None:
        self._running = False
