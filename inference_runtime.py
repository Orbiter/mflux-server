"""Small, model-independent building blocks for inference adapters.

Caches belong to a loaded model or a single request, never to global state.
Stage measurements synchronize only when explicitly used by benchmarks.
"""
from collections import OrderedDict
from collections.abc import MutableMapping
from contextlib import contextmanager
import time

import mlx.core as mx


def array_bytes(value):
    if isinstance(value, mx.array):
        return value.nbytes
    if isinstance(value, dict):
        return sum(array_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(array_bytes(v) for v in value)
    return 0


class BoundedArrayCache(MutableMapping):
    """LRU bounded by logical tensor bytes and entry count.

    A count limit also bounds overhead and backing buffers retained by small
    views. The byte budget describes tensor payload, not total process memory.
    """

    def __init__(self, max_bytes, max_entries=16):
        if max_bytes < 0 or max_entries < 1:
            raise ValueError("Cache budget must be nonnegative and entry limit positive")
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.nbytes = 0
        self._entries = OrderedDict()

    def __getitem__(self, key):
        value, _ = self._entries[key]
        self._entries.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        if key in self._entries:
            del self[key]
        size = array_bytes(value)
        if not size or size > self.max_bytes:
            return
        while self.nbytes + size > self.max_bytes or len(self) >= self.max_entries:
            del self[next(iter(self._entries))]
        self._entries[key] = value, size
        self.nbytes += size

    def __delitem__(self, key):
        _, size = self._entries.pop(key)
        self.nbytes -= size

    def __iter__(self):
        return iter(self._entries)

    def __len__(self):
        return len(self._entries)

    def clear(self):
        self._entries.clear()
        self.nbytes = 0


@contextmanager
def instance_override(instance, name, value):
    """Restore instance state, including inherited methods, on every exit path."""
    own = name in vars(instance) or (isinstance(instance, dict) and name in instance)
    original = getattr(instance, name)
    setattr(instance, name, value)
    try:
        yield
    finally:
        if own:
            setattr(instance, name, original)
        else:
            delattr(instance, name)


class StageTimings:
    """Opt-in synchronized timing for native stages in any model adapter."""

    def __init__(self):
        self.seconds = {}

    def wrap(self, name, function):
        def measured(*args, **kwargs):
            mx.synchronize()
            start = time.perf_counter()
            result = function(*args, **kwargs)
            if isinstance(result, (mx.array, tuple, list, dict)):
                mx.eval(result)
            mx.synchronize()
            self.seconds[name] = self.seconds.get(name, 0.0) + time.perf_counter() - start
            return result
        return measured
