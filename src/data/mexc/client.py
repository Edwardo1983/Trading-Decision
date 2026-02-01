from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class MexcRESTClient:
    def __init__(self, api_key: str = "", api_secret: str = "", base_url: str = "https://api.mexc.com"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")

    async def _get(self, path: str, params: Optional[Dict[str, Any]] = None, retries: int = 3, backoff: float = 2.0) -> Any:
        url = f"{self.base_url}{path}"
        headers = {}
        if self.api_key:
            headers["X-MEXC-APIKEY"] = self.api_key
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
                    logger.error("MEXC REST error %s %s: %s", path, params, exc)
                    raise
                await asyncio.sleep(backoff * attempt)

    async def ping(self) -> bool:
        try:
            await self._get("/api/v3/ping")
            return True
        except Exception:
            return False

    async def get_klines(self, symbol: str, interval: str, limit: int = 200) -> List[List[Any]]:
        return await self._get("/api/v3/klines", params={"symbol": symbol, "interval": interval, "limit": limit})

    async def get_ticker_24h(self, symbol: str) -> Dict[str, Any]:
        data = await self._get("/api/v3/ticker/24hr", params={"symbol": symbol})
        if isinstance(data, dict):
            return data
        return {}

    async def get_recent_trades(self, symbol: str, limit: int = 500) -> List[Dict[str, Any]]:
        data = await self._get("/api/v3/trades", params={"symbol": symbol, "limit": min(limit, 1000)})
        if isinstance(data, list):
            return data
        return []
