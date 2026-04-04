"""Technical reference data repository with lazy-loading cache.

Provides lookup APIs for materials, S-N curves, SCF coefficients,
CCT diagrams, residual stress profiles, filler metals, and weld
efficiency factors. Large datasets are loaded on-demand and cached
in memory with LRU eviction.
"""

from feaweld.data.cache import DataCache, get_cache
from feaweld.data.registry import DataRegistry, DatasetInfo
