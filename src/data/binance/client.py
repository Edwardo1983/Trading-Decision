from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class BinanceRESTClient:
    def __init__(self, api_key: str = "", api_secret: str = "", market_type: str = "spot", testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.market_type = market_type
        self.testnet = testnet
        self.base_url = self._resolve_base_url()
        self.klines_path = "/fapi/v1/klines" if self.market_type == "futures" else "/api/v3/klines"

    def _resolve_base_url(self) -> str:
        if self.market_type == "futures":
            return "https://testnet.binancefuture.com" if self.testnet else "https://fapi.binance.com"
        return "https://testnet.binance.vision" if self.testnet else "https://api.binance.com"

    async def _get(self, path: str, params: Optional[Dict[str, Any]] = None, retries: int = 3, backoff: float = 2.0) -> Any:
        url = f"{self.base_url}{path}"
        headers = {}
        if self.api_key:
            headers["X-MBX-APIKEY"] = self.api_key
        attempt = 0
        while True:
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.get(url, params=params, headers=headers)
                    resp.raise_for_status()
                    return resp.json()
            except Exception as exc:
                attempt += 1
                if attempt > retries:
                    logger.error("REST error %s %s: %s", path, params, exc)
                    raise
                await asyncio.sleep(backoff * attempt)

    async def get_klines(self, symbol: str, interval: str, limit: int = 200) -> List[List[Any]]:
        return await self._get(self.klines_path, params={"symbol": symbol, "interval": interval, "limit": limit})

    async def ping(self) -> bool:
        try:
            await self._get("/api/v3/ping")
            return True
        except Exception:
            return False

    async def get_ticker_24h(self, symbol: str) -> Dict[str, Any]:
        path = "/fapi/v1/ticker/24hr" if self.market_type == "futures" else "/api/v3/ticker/24hr"
        data = await self._get(path, params={"symbol": symbol})
        if isinstance(data, dict):
            return data
        return {}

    async def get_recent_trades(self, symbol: str, limit: int = 500) -> List[Dict[str, Any]]:
        path = "/api/v3/trades" if self.market_type == "spot" else "/fapi/v1/trades"
        data = await self._get(path, params={"symbol": symbol, "limit": min(limit, 1000)})
        if isinstance(data, list):
            return data
        return []

    async def get_funding_rate(self, symbol: str) -> Optional[float]:
        if self.market_type != "futures":
            return None
        data = await self._get("/fapi/v1/fundingRate", params={"symbol": symbol, "limit": 1})
        if isinstance(data, list) and data:
            try:
                return float(data[0].get("fundingRate", 0))
            except Exception:
                return None
        return None

    async def get_open_interest(self, symbol: str) -> Optional[float]:
        if self.market_type != "futures":
            return None
        data = await self._get("/fapi/v1/openInterest", params={"symbol": symbol})
        try:
            return float(data.get("openInterest", 0))
        except Exception:
            return None

    async def get_long_short_ratio(self, symbol: str, period: str = "5m") -> Optional[float]:
        if self.market_type != "futures":
            return None
        data = await self._get(
            "/futures/data/globalLongShortAccountRatio",
            params={"symbol": symbol, "period": period, "limit": 1},
        )
        if isinstance(data, list) and data:
            try:
                return float(data[0].get("longShortRatio", 0))
            except Exception:
                return None
        return None
