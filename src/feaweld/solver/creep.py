"""Creep models and PWHT simulation for residual stress relaxation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from feaweld.core.materials import Material
from feaweld.core.loads import PWHTSchedule
from feaweld.core.types import FEAResults, StressField


def norton_bailey_rate(
    stress: NDArray, time: float, A: float, n: float, m: float
) -> NDArray:
    """Norton-Bailey time-hardening creep strain rate.

    The law is:

        eps_dot_cr = A * sigma_vm^n * t^m

    where sigma_vm is the von Mises equivalent stress.

    Parameters
    ----------
    stress : NDArray
        Stress tensor(s) in Voigt notation, shape ``(6,)`` or ``(n_pts, 6)``.
    time : float
        Current time (s).  Must be > 0 for m != 0.
    A : float
        Creep pre-factor.
    n : float
        Stress exponent.
    m : float
        Time exponent.

    Returns
    -------
    NDArray
        Creep strain rate in Voigt notation, same shape as *stress*.
        The direction is the flow rule (proportional to deviatoric stress).
    """
    stress = np.asarray(stress, dtype=np.float64)
    # Handle scalar or 1D von Mises input (not full tensor)
    if stress.ndim == 0 or (stress.ndim == 1 and stress.shape[0] != 6):
        # Treat as von Mises stress values — return scalar creep rate
        sigma_vm = np.atleast_1d(stress)
        t_factor = np.abs(time) ** m if time != 0.0 else (1.0 if m == 0.0 else 0.0)
        return A * np.abs(sigma_vm) ** n * t_factor

    single = stress.ndim == 1
    if single:
        stress = stress[np.newaxis, :]

    # Von Mises equivalent stress
    s = stress.copy()
    hydro = (s[:, 0] + s[:, 1] + s[:, 2]) / 3.0
    s[:, 0] -= hydro
    s[:, 1] -= hydro
    s[:, 2] -= hydro
    s_sq = (
        s[:, 0] ** 2 + s[:, 1] ** 2 + s[:, 2] ** 2
        + 2.0 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2)
    )
    sigma_vm = np.sqrt(1.5 * s_sq)

    # Avoid division by zero for zero stress
    sigma_vm_safe = np.maximum(sigma_vm, 1e-30)

    # Scalar creep rate magnitude
    t_factor = np.abs(time) ** m if time != 0.0 else (1.0 if m == 0.0 else 0.0)
    eps_dot_eq = A * sigma_vm ** n * t_factor

    # Creep strain rate direction: proportional to deviatoric stress
    # eps_dot_cr_ij = (3/2) * eps_dot_eq / sigma_vm * s_ij
    factor = np.where(
        sigma_vm > 1e-30,
        1.5 * eps_dot_eq / sigma_vm_safe,
        0.0,
    )

    eps_dot = np.zeros_like(stress)
    eps_dot[:, 0] = factor * s[:, 0]
    eps_dot[:, 1] = factor * s[:, 1]
    eps_dot[:, 2] = factor * s[:, 2]
    # Engineering shear strain rate = 2 * tensor shear strain rate
    eps_dot[:, 3] = factor * s[:, 3] * 2.0
    eps_dot[:, 4] = factor * s[:, 4] * 2.0
    eps_dot[:, 5] = factor * s[:, 5] * 2.0

    if single:
        return eps_dot[0]
    return eps_dot


def simulate_pwht(
    results: FEAResults,
    material: Material,
    schedule: PWHTSchedule,
    dt: float = 60.0,
) -> FEAResults:
    """Simulate stress relaxation during post-weld heat treatment.

    Time-steps through the PWHT temperature schedule, computing creep
    strain increments via the Norton-Bailey law and updating the stress
    field accordingly.

    Parameters
    ----------
    results : FEAResults
        FEA results containing the initial (as-welded) residual stress field.
        ``results.stress`` must be populated.
    material : Material
        Material with creep parameters (``creep_A``, ``creep_n``, ``creep_m``)
        and temperature-dependent elastic properties.
    schedule : PWHTSchedule
        PWHT temperature-time schedule.
    dt : float
        Time step (s) for the creep integration.  Default 60 s.

    Returns
    -------
    FEAResults
        New results with the relaxed residual stress field and accumulated
        creep strain.  The ``metadata`` dict contains the key
        ``"pwht_creep_strain"`` with the final creep strain array.
    """
    if results.stress is None:
        raise ValueError("Input results must have a stress field for PWHT simulation.")

    times, temperatures = schedule.temperature_profile(dt=dt)
    n_pts = results.stress.values.shape[0]

    # Current stress (mutable copy)
    stress_current = results.stress.values.copy()
    # Accumulated creep strain
    creep_strain = np.zeros_like(stress_current)

    A = material.creep_A
    n_exp = material.creep_n
    m_exp = material.creep_m

    # Time-stepping
    for i in range(1, len(times)):
        t = times[i]
        T = temperatures[i]
        dt_step = times[i] - times[i - 1]

        # Skip if creep parameters are zero or temperature is too low
        if A <= 0.0 or T < 100.0:
            continue

        # Compute elasticity tensor at current temperature
        C = material.elasticity_tensor_3d(T)

        # Creep strain rate at each point
        eps_dot_cr = norton_bailey_rate(stress_current, t, A, n_exp, m_exp)

        # Creep strain increment (clamp to prevent overflow)
        d_eps_cr = eps_dot_cr * dt_step
        max_strain_inc = 0.01  # cap at 1% per step to prevent instability
        d_eps_cr = np.clip(d_eps_cr, -max_strain_inc, max_strain_inc)

        # Update stress: sigma_new = sigma - C : d_eps_cr
        # This is the stress relaxation: creep strain "eats into" the elastic strain
        for pt in range(n_pts):
            d_stress = C @ d_eps_cr[pt]
            # Limit correction so stress relaxes toward zero but doesn't overshoot
            # Scale back if the correction would increase the von Mises stress
            old_vm = np.sqrt(0.5 * (
                (stress_current[pt, 0] - stress_current[pt, 1])**2 +
                (stress_current[pt, 1] - stress_current[pt, 2])**2 +
                (stress_current[pt, 2] - stress_current[pt, 0])**2 +
                6 * (stress_current[pt, 3]**2 + stress_current[pt, 4]**2 + stress_current[pt, 5]**2)
            ))
            if old_vm > 1e-10:
                d_stress_mag = np.linalg.norm(d_stress)
                if d_stress_mag > 0.5 * old_vm:
                    d_stress *= 0.5 * old_vm / d_stress_mag
            stress_current[pt] -= d_stress
            if not np.all(np.isfinite(stress_current[pt])):
                import warnings
                warnings.warn(
                    f"Non-finite stress at point {pt} during PWHT at time "
                    f"{t:.1f}s. Values clamped to 0. Consider smaller time steps.",
                    RuntimeWarning,
                    stacklevel=1,
                )
            stress_current[pt] = np.where(
                np.isfinite(stress_current[pt]), stress_current[pt], 0.0
            )

        # Accumulate creep strain
        creep_strain += d_eps_cr

    # Build output results
    relaxed_stress = StressField(
        values=stress_current,
        location=results.stress.location,
    )

    new_results = FEAResults(
        mesh=results.mesh,
        displacement=results.displacement.copy() if results.displacement is not None else None,
        stress=relaxed_stress,
        strain=results.strain.copy() if results.strain is not None else None,
        temperature=results.temperature.copy() if results.temperature is not None else None,
        nodal_forces=results.nodal_forces.copy() if results.nodal_forces is not None else None,
        time_steps=times,
        metadata={
            **results.metadata,
            "pwht_creep_strain": creep_strain,
            "pwht_schedule": {
                "holding_temperature": schedule.holding_temperature,
                "holding_time_hours": schedule.holding_time,
                "heating_rate_C_per_hour": schedule.heating_rate,
                "cooling_rate_C_per_hour": schedule.cooling_rate,
            },
        },
    )
    return new_results
