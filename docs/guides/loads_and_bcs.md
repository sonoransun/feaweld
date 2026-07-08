# Loads & boundary conditions

The `load:` block of an analysis case is a compact, engineering-level description:
axial force, bending moment, shear, pressure, and a thermal delta. `run_analysis()`
turns it into an explicit `LoadCase` of per-node boundary conditions
(`_build_load_case` in `pipeline/workflow.py`). This guide documents exactly how
each field is applied so the numbers in your report are traceable.

## The fixed-bottom, top-loaded convention

feaweld's meshers tag two node sets on every joint: `bottom` and `top`. Loads are
applied to `top` and reacted at `bottom`:

- If a `bottom` set exists it is fully fixed — a displacement constraint of
  `[0, 0, 0]`.
- Force-type loads act on the `top` set, distributed across its nodes.

If a set is missing, the relevant load is skipped (forces need a `top` set;
pressure falls back to `all`; temperature always uses `all`).

## `LoadConfig` field reference

```yaml
load:
  axial_force: 50000.0      # N     — tension/compression along +y
  bending_moment: 0.0       # N·mm  — self-equilibrating couple on the top set
  shear_force: 0.0          # N     — transverse load along +x
  pressure: 0.0             # MPa   — surface pressure on the top set
  temperature_delta: 0.0    # C     — uniform thermoelastic temperature rise
```

| Field | Unit | Node set | Applied as |
|-------|------|----------|------------|
| `axial_force` | N | `top` | Per-node force `F / n_top` along `[0, 1, 0]` |
| `shear_force` | N | `top` | Per-node force `V / n_top` along `[1, 0, 0]` |
| `bending_moment` | N·mm | `top` | Self-equilibrating nodal couple (see below) |
| `pressure` | MPa | `top` (or `all`) | Surface pressure |
| `temperature_delta` | C | `all` | Uniform temperature for thermoelastic strain |

Only non-zero fields produce a boundary condition, and they combine additively —
set `axial_force` and `bending_moment` together to get combined membrane +
bending, for example.

## How each load is applied

### Axial force

The total force is divided equally across the `top` nodes so the backend applies
the exact total. With `n` top nodes each node receives `axial_force / n` along the
`+y` unit direction. Using per-node values (rather than one lumped value) keeps the
applied total independent of how finely the top edge is meshed.

### Shear force

Identical treatment, along the `+x` direction: each top node receives
`shear_force / n`.

### Bending moment

A pure moment is converted into a **self-equilibrating force couple** by
`moment_to_nodal_forces()`. Each node gets a force along `+y` proportional to its
lever arm about the set centroid along `x`:

$$f_i = M \cdot \frac{x_i - \bar{x}}{\sum_j (x_j - \bar{x})^2}$$

This produces zero net force and a net moment of exactly `M`. If every node in the
set shares the same `x` coordinate the lever arm vanishes and a `ValueError` is
raised — apply the moment to a set with spatial extent along the bending axis.

### Pressure

`pressure` is applied as a surface pressure on the `top` set (or `all` if there is
no `top` set). The value is in MPa.

### Temperature delta (thermoelastic)

`temperature_delta` applies a **uniform temperature** to `all` nodes for a
thermoelastic (thermal-stress) solve. The value is stored as an absolute
temperature of `20 + temperature_delta`, because the backends use a 20 °C
stress-free reference. The resulting thermal strain is therefore
$\alpha \cdot \Delta T$, with $\alpha$ evaluated from the material at temperature.

!!! note "Thermal delta vs. welding thermal simulation"
    `load.temperature_delta` is a single uniform temperature for a quick
    thermoelastic estimate. It is *not* the welding thermal cycle — for a moving
    Goldak source and a transient history use `solver_type: thermal_transient` or
    `thermomechanical` with a `thermal:` block (see the
    [Solvers guide](solvers.md)).

## Welding thermal boundary conditions

When `thermal.enabled` is true the thermal solve uses a separate boundary set built
by `_build_thermal_load_case`: the `bottom` set is held at
`thermal.ambient_temperature`, and a convection load is applied to `all` nodes with
the film coefficient and ambient temperature
(`[thermal.film_coefficient, thermal.ambient_temperature]`). The moving heat input
comes from a Goldak double-ellipsoid source derived from the welding parameters —
see [PWHT](pwht.md) and [Solvers](solvers.md).

## Combining loads — worked example

Combined tension and in-plane bending on a fillet T-joint:

```yaml
load:
  axial_force: 40000.0     # N tension
  bending_moment: 250000.0 # N·mm in-plane bending
```

Both conditions are emitted on the `top` set: a uniform per-node tension plus the
proportional bending couple. The nominal stress used by the notch-stress and
probabilistic models combines them as

$$\sigma_{nom} = \frac{|F|}{w\,t} + \frac{6\,|M|}{w\,t^2}$$

with `w = geometry.base_width` and `t = geometry.base_thickness`.

## Programmatic loads

For loads beyond the five case fields, build a `LoadCase` directly from
`feaweld.core.types` (`BoundaryCondition`, `LoadType`) and pass it to a backend's
`solve_static`. The helpers in `feaweld.core.loads` — `moment_to_nodal_forces`,
`MechanicalLoad`, `ThermalLoad`, `WeldingHeatInput` — cover the common cases.

## See also

- [Solvers](solvers.md) — solver types and the thermal boundary setup.
- [PWHT](pwht.md) — heat input and post-weld heat treatment.
- [Probabilistic & reliability](probabilistic.md) — the closed-form nominal-stress
  response model built from these loads.
