# data

Technical reference data repository with lazy-loading cache.

- `registry` — import-time scan of bundled data directories; O(1) lookups via `get_dataset_path("category/name")`.
- `cache` — on-demand dataset loading with LRU eviction.
- `scf` — parametric stress concentration factor coefficients for 10 weld geometries.
- `cct` — CCT diagrams for 20 steel grades.
- `residual_stress` — BS 7910 / API 579 / R6 / FITNET / DNV through-thickness profiles.
- `filler_metals` — AWS A5 filler metal classifications with base metal matching.
- `weld_efficiency` — ASME / AWS / EN weld joint efficiency factors.

::: feaweld.data
