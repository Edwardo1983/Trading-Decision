from __future__ import annotations

from collections import deque
from datetime import datetime
from threading import Lock
from typing import Callable, Deque, List, Optional

from core.models import Event


class EventBus:
    def __init__(self, max_events: int = 50):
        self._events: Deque[Event] = deque(maxlen=max_events)
        self._lock = Lock()
        self._subscribers: List[Callable[[Event], None]] = []

    def publish(self, level: str, message: str, context: Optional[dict] = None) -> None:
        event = Event(datetime.utcnow(), level, message, context or {})
        with self._lock:
            self._events.appendleft(event)
            for sub in self._subscribers:
                sub(event)

    def subscribe(self, callback: Callable[[Event], None]) -> None:
        with self._lock:
            self._subscribers.append(callback)

    def recent(self) -> List[Event]:
        with self._lock:
            return list(self._events)
