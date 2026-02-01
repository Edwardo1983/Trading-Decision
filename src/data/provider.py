from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Protocol

from core.models import Candle
from data.binance.client import BinanceRESTClient
from data.binance.mapper import kline_to_candle as binance_kline_to_candle
from data.mexc.client import MexcRESTClient
from data.mexc.mapper import kline_to_candle as mexc_kline_to_candle

logger = logging.getLogger(__name__)


class MarketDataProvider(Protocol):
    name: str
    market_type: str
    supports_ws: bool

    async def ping(self) -> bool: ...
    async def get_candles(self, symbol: str, interval: str, limit: int = 200) -> List[Candle]: ...
    async def get_ticker_24h(self, symbol: str) -> Dict[str, Any]: ...
    async def get_recent_trades(self, symbol: str, limit: int = 500) -> List[Dict[str, Any]]: ...
    async def get_funding_rate(self, symbol: str) -> Optional[float]: ...
    async def get_open_interest(self, symbol: str) -> Optional[float]: ...
    async def get_long_short_ratio(self, symbol: str, period: str = "5m") -> Optional[float]: ...


class BinanceProvider:
    name = "binance"

    def __init__(self, api_key: str = "", api_secret: str = "", market_type: str = "spot", testnet: bool = False):
        self.market_type = market_type
        self.supports_ws = True
        self.client = BinanceRESTClient(api_key, api_secret, market_type, testnet)

    async def ping(self) -> bool:
        return await self.client.ping()

    async def get_candles(self, symbol: str, interval: str, limit: int = 200) -> List[Candle]:
        klines = await self.client.get_klines(symbol, interval, limit=limit)
        return [binance_kline_to_candle(k) for k in klines]

    async def get_ticker_24h(self, symbol: str) -> Dict[str, Any]:
        return await self.client.get_ticker_24h(symbol)

    async def get_recent_trades(self, symbol: str, limit: int = 500) -> List[Dict[str, Any]]:
        return await self.client.get_recent_trades(symbol, limit=limit)

    async def get_funding_rate(self, symbol: str) -> Optional[float]:
        return await self.client.get_funding_rate(symbol)

    async def get_open_interest(self, symbol: str) -> Optional[float]:
        return await self.client.get_open_interest(symbol)

    async def get_long_short_ratio(self, symbol: str, period: str = "5m") -> Optional[float]:
        return await self.client.get_long_short_ratio(symbol, period=period)


class MexcProvider:
    name = "mexc"

    def __init__(self, api_key: str = "", api_secret: str = ""):
        self.market_type = "spot"
        self.supports_ws = False
        self.client = MexcRESTClient(api_key, api_secret)

    async def ping(self) -> bool:
        return await self.client.ping()

    async def get_candles(self, symbol: str, interval: str, limit: int = 200) -> List[Candle]:
        klines = await self.client.get_klines(symbol, interval, limit=limit)
        return [mexc_kline_to_candle(k) for k in klines]

    async def get_ticker_24h(self, symbol: str) -> Dict[str, Any]:
        return await self.client.get_ticker_24h(symbol)

    async def get_recent_trades(self, symbol: str, limit: int = 500) -> List[Dict[str, Any]]:
        return await self.client.get_recent_trades(symbol, limit=limit)

    async def get_funding_rate(self, symbol: str) -> Optional[float]:
        return None

    async def get_open_interest(self, symbol: str) -> Optional[float]:
        return None

    async def get_long_short_ratio(self, symbol: str, period: str = "5m") -> Optional[float]:
        return None


async def create_provider(config: Dict) -> MarketDataProvider:
    data_cfg = config.get("data", {})
    provider = str(data_cfg.get("provider", "auto")).lower()

    binance_cfg = config.get("binance", {})
    app_cfg = config.get("app", {})
    market_type = app_cfg.get("market_type", "spot")
    testnet = bool(binance_cfg.get("testnet", False))
    binance = BinanceProvider(
        api_key=binance_cfg.get("api_key", ""),
        api_secret=binance_cfg.get("api_secret", ""),
        market_type=market_type,
        testnet=testnet,
    )

    mexc_cfg = config.get("mexc", {})
    mexc = MexcProvider(
        api_key=mexc_cfg.get("api_key", ""),
        api_secret=mexc_cfg.get("api_secret", ""),
    )

    if provider == "binance":
        return binance
    if provider == "mexc":
        return mexc

    # auto: prefer Binance if reachable
    if await binance.ping():
        logger.info("Provider auto: using Binance")
        return binance
    if await mexc.ping():
        logger.info("Provider auto: using MEXC")
        return mexc
    logger.warning("Provider auto: both ping failed, defaulting to Binance")
    return binance
