from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator, Optional

import websockets

logger = logging.getLogger(__name__)


class BinanceWSClient:
    def __init__(self, symbol: str, interval: str, reconnect_seconds: int = 5, market_type: str = "spot"):
        self.symbol = symbol.lower()
        self.interval = interval
        self.reconnect_seconds = reconnect_seconds
        self.market_type = market_type
        self._running = False

    def _url(self) -> str:
        base = "wss://stream.binance.com:9443"
        if self.market_type == "futures":
            base = "wss://fstream.binance.com"
        return f"{base}/ws/{self.symbol}@kline_{self.interval}"

    async def listen(self) -> AsyncIterator[dict]:
        self._running = True
        while self._running:
            try:
                async with websockets.connect(self._url(), ping_interval=20, ping_timeout=20) as ws:
                    logger.info("WS connected: %s", self._url())
                    async for msg in ws:
                        if not self._running:
                            break
                        try:
                            payload = json.loads(msg)
                        except json.JSONDecodeError as exc:
                            logger.warning("WS message decode error: %s", exc)
                            continue
                        if isinstance(payload, dict):
                            yield payload
                        else:
                            logger.warning("WS message not a JSON object: %s", payload)
            except Exception as exc:
                logger.warning("WS disconnected: %s", exc)
                await asyncio.sleep(self.reconnect_seconds)

    def stop(self) -> None:
        self._running = False
