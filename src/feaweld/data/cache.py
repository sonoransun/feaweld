"""Lazy-loading data cache with LRU eviction.

Datasets are loaded from bundled YAML/JSON files on first access and
kept in memory. When the total cached data exceeds the memory limit,
the least-recently-used entries are evicted.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from feaweld.data.registry import DataRegistry


@dataclass
class CacheEntry:
    """One cached dataset."""
    data: Any
    loaded_at: float
    last_accessed: float
    size_bytes: int


class DataCache:
    """Thread-safe, lazy-loading data cache with LRU eviction.

    Usage::

        cache = get_cache()
        data = cache.get("cct/A36")        # loads on first access
        data = cache.get("cct/A36")        # returns from memory
        cache.clear()                       # evict everything
        cache.preload("materials")          # eagerly load all materials
    """

    def __init__(
        self,
        registry: DataRegistry | None = None,
        max_memory_bytes: int = 50 * 1024 * 1024,  # 50 MB
    ) -> None:
        self._registry = registry or DataRegistry()
        self._max_memory_bytes = max_memory_bytes
        self._entries: dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self._total_bytes = 0

    def get(self, key: str) -> Any:
        """Get a dataset by key, loading from disk on first access.

        Args:
            key: Dataset key (e.g., "cct/A36", "scf/parametric_coefficients").

        Returns:
            Parsed data (dict, list, or other structure from YAML/JSON).

        Raises:
            KeyError: If the dataset is not in the registry.
        """
        with self._lock:
            if key in self._entries:
                self._entries[key].last_accessed = time.time()
                return self._entries[key].data

        # Load outside the lock to avoid blocking other threads
        path = self._registry.get_dataset_path(key)
        data = self._load_file(path)
        size = sys.getsizeof(data)

        with self._lock:
            # Double-check: another thread may have loaded it
            if key in self._entries:
                return self._entries[key].data

            self._entries[key] = CacheEntry(
                data=data,
                loaded_at=time.time(),
                last_accessed=time.time(),
                size_bytes=size,
            )
            self._total_bytes += size
            self._evict_if_needed()

        return data

    def preload(self, category: str) -> None:
        """Eagerly load all datasets in a category into the cache."""
        for ds in self._registry.list_datasets(category):
            self.get(ds.name)

    def clear(self) -> None:
        """Evict all entries from the cache."""
        with self._lock:
            self._entries.clear()
            self._total_bytes = 0

    @property
    def stats(self) -> dict[str, Any]:
        """Cache statistics."""
        with self._lock:
            return {
                "entries": len(self._entries),
                "total_bytes": self._total_bytes,
                "max_bytes": self._max_memory_bytes,
                "keys": list(self._entries.keys()),
            }

    def _load_file(self, path: Path) -> Any:
        """Load a YAML or JSON file."""
        if path.suffix == ".json":
            with open(path) as f:
                return json.load(f)
        else:
            with open(path) as f:
                return yaml.safe_load(f)

    def _evict_if_needed(self) -> None:
        """Evict least-recently-used entries until under memory limit.

        Must be called while holding self._lock.
        """
        while self._total_bytes > self._max_memory_bytes and self._entries:
            # Find LRU entry
            lru_key = min(self._entries, key=lambda k: self._entries[k].last_accessed)
            evicted = self._entries.pop(lru_key)
            self._total_bytes -= evicted.size_bytes


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_cache_instance: DataCache | None = None
_cache_lock = threading.Lock()


def get_cache() -> DataCache:
    """Get the global DataCache singleton."""
    global _cache_instance
    if _cache_instance is None:
        with _cache_lock:
            if _cache_instance is None:
                _cache_instance = DataCache()
    return _cache_instance
