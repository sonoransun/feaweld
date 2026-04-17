"""Goldak double-ellipsoid heat source along a straight weld path.

Samples the volumetric heat input of a traveling welding torch on a
regular grid and reports the peak power density and its location at a
given snapshot time. This is the building block the transient thermal
solver integrates; here we just visualize the source field itself.
"""

import numpy as np

from feaweld.solver.thermal import GoldakHeatSource


def main():
    # Arc parameters — typical SMAW: 250 A @ 25 V, 80 % efficiency.
    voltage, current, efficiency = 25.0, 250.0, 0.8
    net_power = efficiency * voltage * current  # W

    source = GoldakHeatSource(
        power=net_power,
        a_f=5.0,     # front semi-axis (mm)
        a_r=10.0,    # rear semi-axis (mm)
        b=4.0,       # width (mm)
        c=3.0,       # depth (mm)
        travel_speed=5.0,                  # mm/s
        start_position=np.array([0.0, 0.0, 0.0]),
        direction=np.array([1.0, 0.0, 0.0]),
    )

    print("Goldak Heat Source")
    print("=" * 50)
    print(f"  Net power:      {net_power:.0f} W")
    print(f"  Travel speed:   {source.travel_speed} mm/s")
    print(f"  Ellipsoid a_f / a_r / b / c: "
          f"{source.a_f} / {source.a_r} / {source.b} / {source.c} mm")

    # Evaluate on a regular grid at t = 2.0 s (source centre is 10 mm downstream)
    t = 2.0
    x = np.linspace(-20.0, 40.0, 80)
    y = np.linspace(-10.0, 10.0, 40)
    z = np.linspace(-5.0, 5.0, 20)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    q = source.evaluate(X, Y, Z, t)

    peak = float(q.max())
    peak_idx = np.unravel_index(np.argmax(q), q.shape)
    peak_xyz = (X[peak_idx], Y[peak_idx], Z[peak_idx])

    print(f"\nSnapshot at t = {t:.1f} s:")
    print(f"  Peak power density: {peak:.2f} W/mm^3")
    print(f"  Peak location:      ({peak_xyz[0]:.2f}, "
          f"{peak_xyz[1]:.2f}, {peak_xyz[2]:.2f}) mm")
    print(f"  Expected centre:    ({source.travel_speed * t:.2f}, 0.00, 0.00) mm")

    return source


if __name__ == "__main__":
    main()
