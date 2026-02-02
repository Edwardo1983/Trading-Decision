import asyncio

from data.binance.ws import BinanceWSClient


class FakeWebSocket:
    def __init__(self, messages, fail_after: int | None = None):
        self._messages = list(messages)
        self._index = 0
        self._fail_after = fail_after

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._fail_after is not None and self._index >= self._fail_after:
            raise RuntimeError("stream error")
        if self._index >= len(self._messages):
            raise StopAsyncIteration
        message = self._messages[self._index]
        self._index += 1
        return message


class FakeConnection:
    def __init__(self, websocket):
        self._websocket = websocket

    async def __aenter__(self):
        return self._websocket

    async def __aexit__(self, exc_type, exc, tb):
        return False


def test_listen_reconnects_after_failure(monkeypatch):
    calls = {"connect": 0, "sleep": 0}

    def connect(*_args, **_kwargs):
        calls["connect"] += 1
        if calls["connect"] == 1:
            raise RuntimeError("boom")
        return FakeConnection(FakeWebSocket(['{"ok": true}']))

    async def fake_sleep(_seconds):
        calls["sleep"] += 1

    monkeypatch.setattr("data.binance.ws.websockets.connect", connect)
    monkeypatch.setattr("data.binance.ws.asyncio.sleep", fake_sleep)

    client = BinanceWSClient("BTCUSDT", "1m", reconnect_seconds=0)

    async def run():
        async for payload in client.listen():
            client.stop()
            return payload
        return None

    result = asyncio.run(run())

    assert result == {"ok": True}
    assert calls["connect"] >= 2
    assert calls["sleep"] >= 1


def test_listen_skips_bad_json(monkeypatch):
    def connect(*_args, **_kwargs):
        return FakeConnection(FakeWebSocket(["not-json", '{"ok": true}']))

    monkeypatch.setattr("data.binance.ws.websockets.connect", connect)

    client = BinanceWSClient("BTCUSDT", "1m", reconnect_seconds=0)

    async def run():
        async for payload in client.listen():
            client.stop()
            return payload
        return None

    result = asyncio.run(run())

    assert result == {"ok": True}


def test_listen_recovers_after_stream_error(monkeypatch):
    calls = {"connect": 0}

    def connect(*_args, **_kwargs):
        calls["connect"] += 1
        if calls["connect"] == 1:
            return FakeConnection(FakeWebSocket(['{"first": true}'], fail_after=1))
        return FakeConnection(FakeWebSocket(['{"second": true}']))

    async def fake_sleep(_seconds):
        return None

    monkeypatch.setattr("data.binance.ws.websockets.connect", connect)
    monkeypatch.setattr("data.binance.ws.asyncio.sleep", fake_sleep)

    client = BinanceWSClient("BTCUSDT", "1m", reconnect_seconds=0)

    async def run():
        async for payload in client.listen():
            if payload.get("second"):
                client.stop()
                return payload
        return None

    result = asyncio.run(run())

    assert result == {"second": True}
    assert calls["connect"] >= 2
