"""Fatigue assessment of butt weld using multiple S-N curve standards."""

import numpy as np
from feaweld.fatigue.sn_curves import iiw_fat, dnv_curve, get_sn_curve
from feaweld.fatigue.rainflow import rainflow_count
from feaweld.fatigue.miner import miner_damage, fatigue_life_from_damage
from feaweld.fatigue.knockdown import goodman_correction, surface_finish_factor
from feaweld.core.types import LoadHistory


def main():
    # Example: butt weld under variable amplitude loading
    print("Butt Weld Fatigue Assessment")
    print("=" * 50)

    # Define stress history (simplified)
    time = np.linspace(0, 100, 1000)
    stress_signal = 80 * np.sin(2 * np.pi * time / 10) + 20 * np.sin(2 * np.pi * time / 3)

    # Rainflow counting
    cycles = rainflow_count(stress_signal)
    print(f"\nRainflow counting: {len(cycles)} cycles identified")
    for i, (sr, sm, count) in enumerate(cycles[:5]):
        print(f"  Cycle {i+1}: range={sr:.1f} MPa, mean={sm:.1f} MPa, count={count}")

    # Assess with different S-N curves
    print("\nFatigue Life Assessment:")
    print("-" * 50)

    # IIW FAT90 (typical for ground butt weld)
    fat90 = iiw_fat(90)
    cycles_for_miner = [(sr, count) for sr, sm, count in cycles]
    damage_iiw = miner_damage(cycles_for_miner, fat90)
    print(f"  IIW FAT90:    D = {damage_iiw:.4f}, Life factor = {1/damage_iiw:.0f}x")

    # DNV D curve
    dnv_d = dnv_curve("D")
    damage_dnv = miner_damage(cycles_for_miner, dnv_d)
    print(f"  DNV D:        D = {damage_dnv:.4f}, Life factor = {1/damage_dnv:.0f}x")

    # With mean stress correction (Goodman)
    print("\nWith Goodman mean stress correction (σ_u = 400 MPa):")
    corrected_cycles = []
    for sr, sm, count in cycles:
        sa = sr / 2  # amplitude
        sa_corrected = goodman_correction(sa, sm, sigma_u=400.0)
        corrected_cycles.append((2 * sa_corrected, count))

    damage_corrected = miner_damage(corrected_cycles, fat90)
    print(f"  IIW FAT90 (Goodman): D = {damage_corrected:.4f}")

    # Surface finish knockdown
    k_a = surface_finish_factor(roughness_um=6.3, sigma_u=400.0)
    print(f"\nSurface finish factor (Ra=6.3μm): k_a = {k_a:.3f}")

    return {
        "damage_iiw": damage_iiw,
        "damage_dnv": damage_dnv,
        "damage_corrected": damage_corrected,
    }


if __name__ == "__main__":
    main()
