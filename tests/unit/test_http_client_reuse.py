from __future__ import annotations

import asyncio

from data.binance.client import BinanceRESTClient
from data.mexc.client import MexcRESTClient
from data.sentiment import SentimentClient


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


class _FakeAsyncClient:
    instances = 0

    def __init__(self, *args, **kwargs):
        type(self).instances += 1
        self.requests = []
        self.is_closed = False

    async def get(self, url, params=None, headers=None):
        self.requests.append((url, params, headers))
        if "ping" in url:
            return _FakeResponse({})
        if "ticker/24hr" in url:
            return _FakeResponse({"volume": "10", "takerBuyBaseAssetVolume": "4", "priceChangePercent": "1.2"})
        if "fundingRate" in url:
            return _FakeResponse([{"fundingRate": "0.01"}])
        if "openInterest" in url:
            return _FakeResponse({"openInterest": "123"})
        if "globalLongShortAccountRatio" in url:
            return _FakeResponse([{"longShortRatio": "1.5"}])
        if "klines" in url:
            return _FakeResponse([[1, 1, 1, 1, 1, 1]])
        if "trades" in url:
            return _FakeResponse([{"qty": "1", "isBuyerMaker": False}])
        if "fng" in url:
            return _FakeResponse({"data": [{"value": "52"}]})
        return _FakeResponse({})

    async def aclose(self):
        self.is_closed = True


class _FakeProvider:
    market_type = "spot"

    async def get_ticker_24h(self, symbol: str):
        return {"volume": "10", "takerBuyBaseAssetVolume": "4", "priceChangePercent": "1.2"}

    async def get_funding_rate(self, symbol: str):
        return None

    async def get_open_interest(self, symbol: str):
        return None

    async def get_long_short_ratio(self, symbol: str, period: str = "5m"):
        return None

    async def get_recent_trades(self, symbol: str, limit: int = 500):
        return [{"qty": "1", "isBuyerMaker": False}]


def test_binance_rest_client_reuses_single_async_client(monkeypatch):
    _FakeAsyncClient.instances = 0
    monkeypatch.setattr("data.binance.client.httpx.AsyncClient", _FakeAsyncClient)

    client = BinanceRESTClient(api_key="key", market_type="spot", testnet=False)

    async def run():
        await client.get_ticker_24h("BTCUSDC")
        await client.get_ticker_24h("BTCUSDC")

    asyncio.run(run())

    assert _FakeAsyncClient.instances == 1


def test_mexc_rest_client_reuses_single_async_client(monkeypatch):
    _FakeAsyncClient.instances = 0
    monkeypatch.setattr("data.mexc.client.httpx.AsyncClient", _FakeAsyncClient)

    client = MexcRESTClient(api_key="key")

    async def run():
        await client.get_ticker_24h("BTCUSDC")
        await client.get_ticker_24h("BTCUSDC")

    asyncio.run(run())

    assert _FakeAsyncClient.instances == 1


def test_sentiment_client_reuses_fear_greed_http_client(monkeypatch):
    _FakeAsyncClient.instances = 0
    monkeypatch.setattr("data.sentiment.httpx.AsyncClient", _FakeAsyncClient)
    provider = _FakeProvider()
    client = SentimentClient(provider=provider, refresh_seconds=0, fear_greed_enabled=True, spot_trade_depth=0)

    async def run():
        await client.refresh("BTCUSDC")
        await client.refresh("BTCUSDC")

    asyncio.run(run())

    assert _FakeAsyncClient.instances == 1
