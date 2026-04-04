"""Compare residual stress with and without post-weld heat treatment (PWHT)."""

import numpy as np
from feaweld.core.loads import PWHTSchedule


def main():
    print("PWHT Residual Stress Comparison")
    print("=" * 50)

    # Define PWHT schedule (typical for carbon steel)
    schedule = PWHTSchedule(
        heating_rate=100.0,           # 100 C/hour
        holding_temperature=620.0,    # 620 C holding
        holding_time=2.0,             # 2 hours
        cooling_rate=50.0,            # 50 C/hour
    )

    times, temps = schedule.temperature_profile(dt=300.0)
    total_hours = times[-1] / 3600.0

    print(f"\nSchedule:")
    print(f"  Heating: {schedule.heating_rate} C/hour → {schedule.holding_temperature} C")
    print(f"  Holding: {schedule.holding_time} hours at {schedule.holding_temperature} C")
    print(f"  Cooling: {schedule.cooling_rate} C/hour")
    print(f"  Total cycle: {total_hours:.1f} hours ({len(times)} time steps)")
    print(f"  Peak temperature: {np.max(temps):.1f} C")

    # Simulate residual stress relaxation (simplified)
    # Norton-Bailey: ε̇_cr = A * σ^n * t^m
    A = 1e-20  # creep coefficient
    n_exp = 5.0  # stress exponent
    sigma_0 = 250.0  # initial residual stress (MPa) = yield strength

    print(f"\nResidual stress relaxation:")
    print(f"  Initial: σ_res = {sigma_0:.0f} MPa")

    sigma = sigma_0
    dt = 300.0  # seconds

    for i in range(len(times)):
        T = temps[i]
        if T > 400:  # creep only active above ~400C
            # Temperature-enhanced creep rate (Arrhenius-like)
            T_factor = np.exp(-50000.0 / (8.314 * (T + 273.15)))
            eps_dot = A * sigma**n_exp * T_factor
            d_sigma = eps_dot * 200000.0 * dt  # stress reduction
            sigma = max(sigma - d_sigma, 0)

    print(f"  After PWHT: σ_res = {sigma:.0f} MPa")
    print(f"  Reduction: {(1 - sigma/sigma_0) * 100:.0f}%")

    # Compare fatigue lives
    print(f"\nFatigue impact (FAT90, stress range = 100 MPa):")
    # Without PWHT: effective stress range increased by residual stress
    sr_no_pwht = 100.0 + sigma_0 * 0.3  # residual adds ~30% to effective range
    sr_with_pwht = 100.0 + sigma * 0.3

    C = 90.0**3 * 2e6
    N_no_pwht = C / sr_no_pwht**3
    N_with_pwht = C / sr_with_pwht**3

    print(f"  Without PWHT: N = {N_no_pwht:.0f} cycles (σ_eff = {sr_no_pwht:.0f} MPa)")
    print(f"  With PWHT:    N = {N_with_pwht:.0f} cycles (σ_eff = {sr_with_pwht:.0f} MPa)")
    print(f"  Life improvement: {N_with_pwht/N_no_pwht:.1f}x")


if __name__ == "__main__":
    main()
