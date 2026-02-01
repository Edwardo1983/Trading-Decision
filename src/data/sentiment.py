from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import httpx

from data.binance.client import BinanceRESTClient

logger = logging.getLogger(__name__)


class SentimentClient:
    def __init__(
        self,
        rest_client: BinanceRESTClient,
        refresh_seconds: int = 300,
        long_short_period: str = "5m",
        fear_greed_enabled: bool = True,
        spot_trade_depth: int = 0,
    ):
        self.rest_client = rest_client
        self.refresh_seconds = refresh_seconds
        self.long_short_period = long_short_period
        self.fear_greed_enabled = fear_greed_enabled
        self.spot_trade_depth = spot_trade_depth
        self._last_update: Optional[datetime] = None
        self._cache: Dict[str, Any] = {}

    def _stale(self) -> bool:
        if self._last_update is None:
            return True
        return datetime.utcnow() - self._last_update >= timedelta(seconds=self.refresh_seconds)

    async def _fetch_fear_greed(self) -> Optional[int]:
        if not self.fear_greed_enabled:
            return None
        url = "https://api.alternative.me/fng/"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()
            if data.get("data"):
                return int(data["data"][0]["value"])
        except Exception as exc:
            logger.debug("Fear/Greed fetch failed: %s", exc)
        return None

    async def refresh(self, symbol: str) -> Dict[str, Any]:
        if not self._stale():
            return dict(self._cache)

        ticker_24h = await self.rest_client.get_ticker_24h(symbol)
        volume = float(ticker_24h.get("volume", 0) or 0)
        taker_buy = float(ticker_24h.get("takerBuyBaseAssetVolume", 0) or 0)
        taker_sell = max(volume - taker_buy, 0.0)
        buy_sell_ratio = (taker_buy / taker_sell) if taker_sell > 0 else None
        taker_buy_pct = (taker_buy / volume) if volume > 0 else None

        trade_buy_volume = None
        trade_sell_volume = None
        if self.rest_client.market_type == "spot" and self.spot_trade_depth > 0:
            trades = await self.rest_client.get_recent_trades(symbol, limit=self.spot_trade_depth)
            buy = 0.0
            sell = 0.0
            for trade in trades:
                qty = float(trade.get("qty", 0) or 0)
                is_buyer_maker = trade.get("isBuyerMaker", False)
                # if buyer is maker, seller is taker (sell pressure)
                if is_buyer_maker:
                    sell += qty
                else:
                    buy += qty
            trade_buy_volume = buy
            trade_sell_volume = sell
            if sell > 0:
                buy_sell_ratio = buy / sell

        funding = await self.rest_client.get_funding_rate(symbol)
        open_interest = await self.rest_client.get_open_interest(symbol)
        long_short = await self.rest_client.get_long_short_ratio(symbol, period=self.long_short_period)
        if long_short is None and buy_sell_ratio is not None:
            long_short = buy_sell_ratio

        fear_greed = await self._fetch_fear_greed()

        self._cache = {
            "funding_rate": funding,
            "open_interest": open_interest,
            "long_short_ratio": long_short,
            "fear_greed": fear_greed,
            "taker_buy_volume": taker_buy,
            "taker_sell_volume": taker_sell,
            "buy_sell_ratio": buy_sell_ratio,
            "taker_buy_pct": taker_buy_pct,
            "trade_buy_volume": trade_buy_volume,
            "trade_sell_volume": trade_sell_volume,
            "price_change_pct": ticker_24h.get("priceChangePercent"),
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._last_update = datetime.utcnow()
        return dict(self._cache)
