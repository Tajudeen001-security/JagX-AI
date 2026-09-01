from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Any


@dataclass
class CacheEntry:
    value: Any
    hits: int = 0


class BoundedCache:
    """Thread-safe LRU cache with a hard entry bound for inference/runtime use."""

    def __init__(self, max_entries: int = 1024):
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self._data = OrderedDict()
        self._lock = RLock()

    def get(self, key, default=None):
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return default
            entry.hits += 1
            self._data.move_to_end(key)
            return entry.value

    def put(self, key, value):
        with self._lock:
            self._data[key] = CacheEntry(value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)

    def clear(self):
        with self._lock:
            self._data.clear()

    def __len__(self):
        with self._lock:
            return len(self._data)
