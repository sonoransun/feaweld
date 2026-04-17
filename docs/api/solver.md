# solver

FEA solver backends and constitutive models.

- `backend` — `SolverBackend` ABC and the `get_backend("auto"|"fenics"|"calculix")` factory.
- `fenics_backend`, `calculix_backend` — concrete backends with identical `FEAResults` outputs.
- `mechanical` — J2 elastoplasticity with radial-return mapping.
- `thermal` — Goldak double-ellipsoid heat source and element birth-death.
- `creep` — Norton-Bailey creep integration for PWHT stress relaxation.
- `constitutive` — reusable stress-strain, yield, and hardening functions.

::: feaweld.solver
