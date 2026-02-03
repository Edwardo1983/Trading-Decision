from __future__ import annotations

from collections import deque
from typing import Deque, Generic, Iterable, Iterator, List, Optional, TypeVar

T = TypeVar("T")


class RingBuffer(Generic[T]):
    def __init__(self, maxlen: int):
        self._data: Deque[T] = deque(maxlen=maxlen)

    def append(self, item: T) -> None:
        self._data.append(item)

    def replace_last(self, item: T) -> None:
        if self._data:
            self._data.pop()
        self._data.append(item)

    def extend(self, items: Iterable[T]) -> None:
        self._data.extend(items)

    def to_list(self) -> List[T]:
        return list(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[T]:
        return iter(self._data)

    def last(self) -> Optional[T]:
        return self._data[-1] if self._data else None
