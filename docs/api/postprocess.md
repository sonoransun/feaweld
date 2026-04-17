# postprocess

Standalone post-processing methods, dispatched via `StressMethod` in `pipeline.workflow`. Each module consumes `FEAResults` and returns method-specific output — no inheritance, no shared state.

- `hotspot` — IIW hot-spot stress extrapolation (Types A and B).
- `dong` — Battelle/Dong mesh-insensitive structural stress.
- `notch_stress` — IIW effective notch stress (FAT225).
- `sed` — Lazzarin strain-energy-density control-volume method.
- `linearization` — ASME VIII Div 2 through-thickness linearization.
- `nominal` — ASME VIII nominal stress check.
- `blodgett` — standalone weld-group hand calculation (no FEA needed).

::: feaweld.postprocess
