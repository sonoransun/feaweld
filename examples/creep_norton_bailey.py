"""Norton-Bailey creep relaxation during a PWHT hold.

Integrates the Norton-Bailey creep strain rate for an initial residual
stress state held at a constant PWHT temperature. Reports the stress
relaxation over the hold period — the core effect PWHT exploits to
drop tensile residual stresses below fatigue-damaging levels.
"""

import numpy as np

from feaweld.solver.creep import norton_bailey_rate


def main():
    print("Norton-Bailey Creep Relaxation (PWHT hold)")
    print("=" * 55)

    # Typical P91 creep constants at ~620 C (SI, stress in MPa, time in s)
    A = 1.0e-18
    n = 5.0
    m = -0.3  # time-hardening (negative = rate decreases with time)

    sigma0 = 250.0  # initial tensile residual stress (MPa)
    E = 180_000.0   # elastic modulus at PWHT temp (MPa)

    print(f"  Initial residual stress: {sigma0:.1f} MPa")
    print(f"  Elastic modulus:         {E:.0f} MPa")
    print(f"  Constants:               A={A:.1e}, n={n}, m={m}")

    # Explicit time integration over a 2-hour hold, 60 s steps
    hold_seconds = 2.0 * 3600.0
    dt = 60.0
    t = 0.0
    sigma = np.array([sigma0])
    history = [(t, float(sigma[0]))]

    while t < hold_seconds:
        # Use physical time for the time-hardening factor; guard t=0 for m<0.
        t_eff = max(t + dt, 1.0)
        rate = norton_bailey_rate(sigma, t_eff, A, n, m)
        deps_cr = rate * dt              # creep strain increment
        # Stress redistribution under total-strain constraint: dsigma = -E * deps_cr.
        sigma = sigma - E * deps_cr
        sigma = np.maximum(sigma, 0.0)   # floor at 0 — not tracking compressive redistribution
        t += dt
        history.append((t, float(sigma[0])))

    print(f"\n  After {hold_seconds/3600:.1f} h hold:")
    print(f"    Relaxed stress:  {sigma[0]:.1f} MPa")
    print(f"    Relaxation:      {(1 - sigma[0]/sigma0)*100:.1f} %")

    print("\n  Time-history (selected rows):")
    for i in (0, len(history)//4, len(history)//2, 3*len(history)//4, -1):
        ti, si = history[i]
        print(f"    t = {ti/60:>6.1f} min   sigma = {si:>6.1f} MPa")

    return history


if __name__ == "__main__":
    main()
