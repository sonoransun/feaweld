"""Dataset catalog — scans bundled data directories at import time."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_DATA_ROOT = Path(__file__).parent


@dataclass(frozen=True)
class DatasetInfo:
    """Metadata for one dataset file."""
    name: str           # e.g. "materials/A36"
    category: str       # e.g. "materials", "scf", "cct"
    path: Path          # absolute path to file
    format: str         # "yaml" or "json"
    size_bytes: int
    description: str = ""


class DataRegistry:
    """Catalog of all bundled datasets, built from directory scanning."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = root or _DATA_ROOT
        self._datasets: dict[str, DatasetInfo] = {}
        self._scan()

    def _scan(self) -> None:
        """Walk data subdirectories and index all .yaml and .json files."""
        for path in sorted(self._root.rglob("*")):
            if path.suffix not in (".yaml", ".json"):
                continue
            # Skip __pycache__ and __init__
            if "__" in path.name:
                continue

            category = path.parent.name
            stem = path.stem
            key = f"{category}/{stem}"
            fmt = "yaml" if path.suffix == ".yaml" else "json"

            self._datasets[key] = DatasetInfo(
                name=key,
                category=category,
                path=path,
                format=fmt,
                size_bytes=path.stat().st_size,
            )

    def list_datasets(self, category: str | None = None) -> list[DatasetInfo]:
        """List all datasets, optionally filtered by category."""
        if category is None:
            return list(self._datasets.values())
        return [d for d in self._datasets.values() if d.category == category]

    def get_dataset_path(self, key: str) -> Path:
        """Resolve a dataset key to its file path.

        Args:
            key: Dataset key like "cct/A36" or "materials/304SS".

        Raises:
            KeyError: If the dataset is not found.
        """
        if key in self._datasets:
            return self._datasets[key].path
        raise KeyError(f"Dataset not found: {key!r}. Available: {list(self._datasets.keys())[:20]}")

    def search(self, query: str) -> list[DatasetInfo]:
        """Search datasets by substring match on name."""
        q = query.lower()
        return [d for d in self._datasets.values() if q in d.name.lower()]

    @property
    def categories(self) -> list[str]:
        """List all categories."""
        return sorted(set(d.category for d in self._datasets.values()))

    def __len__(self) -> int:
        return len(self._datasets)

    def __contains__(self, key: str) -> bool:
        return key in self._datasets
