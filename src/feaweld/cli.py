"""Command-line interface for feaweld."""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np

from feaweld import __version__


@click.group()
@click.version_option(version=__version__)
def main() -> None:
    """feaweld - Finite Element Analysis for weld joint stress and fatigue."""
    pass


@main.command()
@click.argument("case_file", type=click.Path(exists=True))
@click.option("--output", "-o", default=None, help="Output directory for results")
@click.option("--report/--no-report", default=True, help="Generate HTML report")
def run(case_file: str, output: str | None, report: bool) -> None:
    """Run a complete analysis from a YAML case file."""
    from feaweld.pipeline.workflow import load_case, run_analysis
    from feaweld.pipeline.report import generate_report

    click.echo(f"Loading case: {case_file}")
    case = load_case(case_file)

    if output:
        case.output_dir = output

    click.echo(f"Running analysis: {case.name}")
    click.echo(f"  Joint: {case.geometry.joint_type.value}")
    click.echo(f"  Material: {case.material.base_metal}")
    click.echo(f"  Solver: {case.solver.solver_type.value}")

    result = run_analysis(case)

    if result.errors:
        click.echo(click.style(f"\nCompleted with {len(result.errors)} error(s):", fg="yellow"))
        for err in result.errors:
            click.echo(f"  - {err}")
    else:
        click.echo(click.style("\nAnalysis completed successfully.", fg="green"))

    if result.fea_results and result.fea_results.stress:
        import numpy as np
        vm = result.fea_results.stress.von_mises
        click.echo(f"\nMax von Mises stress: {np.max(vm):.2f} MPa")

    if result.fatigue_results:
        click.echo("\nFatigue results:")
        for method, data in result.fatigue_results.items():
            if isinstance(data, dict) and "life" in data:
                click.echo(f"  {method}: N = {data['life']:.0f} cycles")

    if report:
        report_path = generate_report(result)
        click.echo(f"\nReport: {report_path}")


@main.command()
@click.option("--geometry", "-g", type=click.Choice(
    ["line", "parallel", "c_shape", "l_shape", "box", "circular", "i_shape", "t_shape"]
), default="line", help="Weld group geometry")
@click.option("--d", type=float, required=True, help="Primary dimension d (mm)")
@click.option("--b", type=float, default=0.0, help="Secondary dimension b (mm)")
@click.option("--throat", "-t", type=float, required=True, help="Weld throat thickness (mm)")
@click.option("--axial", "-P", type=float, default=0.0, help="Axial force (N)")
@click.option("--shear", "-V", type=float, default=0.0, help="Shear force (N)")
@click.option("--moment", "-M", type=float, default=0.0, help="Bending moment (N-mm)")
@click.option("--torsion", "-T", type=float, default=0.0, help="Torsion (N-mm)")
@click.option("--fexx", type=float, default=483.0, help="Electrode strength F_EXX (MPa)")
def blodgett(geometry: str, d: float, b: float, throat: float,
             axial: float, shear: float, moment: float, torsion: float,
             fexx: float) -> None:
    """Blodgett hand calculations for weld groups."""
    from feaweld.core.types import WeldGroupShape
    from feaweld.postprocess.blodgett import (
        weld_group_properties, weld_stress, lrfd_capacity, asd_capacity
    )

    shape_map = {
        "line": WeldGroupShape.LINE,
        "parallel": WeldGroupShape.PARALLEL,
        "c_shape": WeldGroupShape.C_SHAPE,
        "l_shape": WeldGroupShape.L_SHAPE,
        "box": WeldGroupShape.BOX,
        "circular": WeldGroupShape.CIRCULAR,
        "i_shape": WeldGroupShape.I_SHAPE,
        "t_shape": WeldGroupShape.T_SHAPE,
    }

    shape = shape_map[geometry]
    props = weld_group_properties(shape, d, b)

    click.echo(f"\nWeld Group Properties ({geometry.upper()}, d={d}, b={b}):")
    click.echo(f"  A_w  = {props.A_w:.2f} mm")
    click.echo(f"  S_x  = {props.S_x:.4f} mm^2")
    click.echo(f"  S_y  = {props.S_y:.4f} mm^2")
    click.echo(f"  J_w  = {props.J_w:.4f} mm^3")
    click.echo(f"  I_x  = {props.I_x:.4f} mm^2")
    click.echo(f"  I_y  = {props.I_y:.4f} mm^2")

    if axial != 0 or shear != 0 or moment != 0 or torsion != 0:
        stress = weld_stress(props, throat, P=axial, V=shear, M=moment, T=torsion)
        click.echo(f"\nStresses (throat = {throat:.2f} mm):")
        for key, val in stress.items():
            click.echo(f"  {key:>15s} = {val:.2f} MPa")

        lrfd = lrfd_capacity(throat, props.A_w, fexx)
        asd = asd_capacity(throat, props.A_w, fexx)
        click.echo(f"\nCapacity (F_EXX = {fexx:.0f} MPa):")
        click.echo(f"  LRFD: phi*R_n = {lrfd:.0f} N")
        click.echo(f"  ASD:  R_n/Omega = {asd:.0f} N")


@main.command()
@click.argument("results_file", type=click.Path(exists=True))
@click.option("--component", "-c", default="von_mises",
              type=click.Choice(["von_mises", "tresca", "xx", "yy", "zz"]),
              help="Stress component to visualize")
@click.option("--output", "-o", default=None, help="Output image file (PNG)")
@click.option("--clip", default=None, help="Clipping plane normal (x,y,z)")
@click.option("--threshold", type=float, default=None, help="Stress threshold for filtering")
@click.option("--annotate", is_flag=True, help="Add critical point annotations")
def visualize(results_file: str, component: str, output: str | None,
              clip: str | None, threshold: float | None, annotate: bool) -> None:
    """Visualize FEA results from a VTK file."""
    try:
        import pyvista as pv
    except ImportError:
        click.echo("PyVista required: pip install feaweld[viz]")
        raise SystemExit(1)

    mesh = pv.read(results_file)
    plotter = pv.Plotter(off_screen=output is not None)

    display_mesh = mesh

    # Apply clipping if requested
    if clip:
        try:
            normal = [float(x) for x in clip.split(",")]
            display_mesh = mesh.clip(normal=normal)
        except Exception as e:
            click.echo(f"Clipping error: {e}")

    # Apply threshold if requested
    if threshold is not None and component in mesh.array_names:
        try:
            display_mesh = display_mesh.threshold(
                value=threshold, scalars=component
            )
        except Exception as e:
            click.echo(f"Threshold error: {e}")

    scalars = component if component in display_mesh.array_names else None
    plotter.add_mesh(display_mesh, scalars=scalars, cmap="jet", show_edges=True)

    # Add annotations if requested
    if annotate and scalars and scalars in display_mesh.array_names:
        vals = display_mesh[scalars]
        max_idx = int(np.argmax(vals))
        max_val = float(vals[max_idx])
        max_pt = display_mesh.points[max_idx]
        plotter.add_point_labels(
            pv.PolyData(max_pt.reshape(1, 3)),
            [f"Max {component}: {max_val:.1f} MPa"],
            font_size=14, text_color="red", point_size=0,
            shape_opacity=0.7, always_visible=True,
        )

    plotter.add_axes()

    if output:
        plotter.screenshot(output)
        click.echo(f"Saved: {output}")
    else:
        plotter.show()


@main.command()
@click.argument("case_file", type=click.Path(exists=True))
@click.option("--output", "-o", default=None, help="Save dashboard as PNG")
def dashboard(case_file: str, output: str | None) -> None:
    """Generate an engineering visualization dashboard from a YAML case file."""
    try:
        import matplotlib
        matplotlib.use("Agg" if output else "TkAgg")
        import matplotlib.pyplot as plt
    except ImportError:
        click.echo("matplotlib required: pip install feaweld[viz]")
        raise SystemExit(1)

    from feaweld.pipeline.workflow import load_case, run_analysis
    from feaweld.visualization.dashboard import engineering_dashboard

    click.echo(f"Loading case: {case_file}")
    case = load_case(case_file)

    click.echo(f"Running analysis: {case.name}")
    result = run_analysis(case)

    click.echo("Generating dashboard...")
    fig = engineering_dashboard(result, show=output is None)

    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
        click.echo(f"Saved: {output}")


@main.command()
def materials() -> None:
    """List available materials."""
    from feaweld.core.materials import list_available_materials
    mats = list_available_materials()
    click.echo("Available materials:")
    for m in mats:
        click.echo(f"  - {m}")


# ---------------------------------------------------------------------------
# Parametric study commands
# ---------------------------------------------------------------------------

@main.group()
def study() -> None:
    """Parametric study management — run and compare multiple cases."""
    pass


@study.command("run")
@click.argument("study_file", type=click.Path(exists=True))
@click.option("--max-workers", "-j", default=4, type=int, help="Parallel worker count")
@click.option("--output", "-o", default=None, help="Output directory for comparison report")
@click.option("--report/--no-report", default=True, help="Generate HTML comparison report")
def study_run(study_file: str, max_workers: int, output: str | None, report: bool) -> None:
    """Run a parametric study from a YAML definition file."""
    from feaweld.pipeline.study import load_study, Study

    click.echo(f"Loading study: {study_file}")
    config = load_study(study_file)

    click.echo(f"Study: {config.name} ({config.mode} mode)")
    s = Study(config.name, config.base_case)
    for p in config.parameters:
        s.vary(p.name, p.values)

    cases = s._generate_cases(config.mode)
    click.echo(f"Generated {len(cases)} cases, running with {max_workers} workers...")

    results = s.run(max_workers=max_workers, mode=config.mode)

    click.echo(f"\nCompleted in {results.elapsed_seconds:.1f}s")
    click.echo(f"  Succeeded: {results.n_succeeded}/{results.n_cases}")
    if results.n_failed > 0:
        click.echo(click.style(f"  Failed: {results.n_failed}", fg="yellow"))

    if report:
        out_dir = output or "results"
        from feaweld.pipeline.comparison import generate_comparison_report
        try:
            path = generate_comparison_report(results, out_dir)
            click.echo(f"\nComparison report: {path}")
        except Exception as e:
            click.echo(f"Report generation failed: {e}")


@study.command("compare")
@click.argument("case_files", nargs=-1, type=click.Path(exists=True))
@click.option("--baseline", "-b", default=None, help="Baseline case name for delta computation")
@click.option("--output", "-o", default="results", help="Output directory")
@click.option("--max-workers", "-j", default=4, type=int, help="Parallel workers for analysis")
def study_compare(case_files: tuple[str, ...], baseline: str | None,
                  output: str, max_workers: int) -> None:
    """Compare multiple analysis case YAML files by running each and generating a comparison."""
    from feaweld.pipeline.workflow import load_case
    from feaweld.pipeline.study import Study

    if len(case_files) < 2:
        click.echo("Provide at least 2 case files to compare.")
        raise SystemExit(1)

    cases = {}
    for path in case_files:
        case = load_case(path)
        name = case.name if case.name != "default" else Path(path).stem
        cases[name] = case

    click.echo(f"Comparing {len(cases)} cases: {', '.join(cases.keys())}")

    s = Study("comparison", list(cases.values())[0])
    for name, case in cases.items():
        s.add_case(name, case)
    # Clear sweeps — only use explicitly added cases
    s._sweeps = []

    results = s.run(max_workers=max_workers, mode="grid")

    click.echo(f"Completed in {results.elapsed_seconds:.1f}s")

    from feaweld.pipeline.comparison import generate_comparison_report
    try:
        path = generate_comparison_report(results, output, baseline=baseline)
        click.echo(f"Comparison report: {path}")
    except Exception as e:
        click.echo(f"Report generation failed: {e}")


# ---------------------------------------------------------------------------
# Defect acceptance criteria
# ---------------------------------------------------------------------------

@main.group()
def defects() -> None:
    """Defect acceptance criteria and downgrade helpers."""
    pass


@defects.command("list")
@click.option(
    "--standard", default="ISO 5817",
    help="Standard to display (ISO 5817, ASME BPVC IX, AWS D1.1, BS 7910)",
)
@click.option("--level", default=None, help="Quality level key (B/C/D or normal/severe)")
def defects_list(standard: str, level: str | None) -> None:
    """List bundled defect acceptance criteria."""
    from feaweld.defects.loader import load_acceptance_criteria

    data = load_acceptance_criteria(standard)
    click.echo(f"Standard: {data.get('standard', standard)}")
    levels = data.get("levels", {})
    if level:
        entry = levels.get(level)
        if not entry:
            click.echo(f"Level {level!r} not found. Available: {list(levels.keys())}")
            return
        for k, v in entry.items():
            click.echo(f"  {k}: {v}")
    else:
        for lvl, entry in levels.items():
            click.echo(f"Level {lvl}:")
            for k, v in entry.items():
                click.echo(f"  {k}: {v}")


@main.command()
def groove_types() -> None:
    """List available groove preparations with a schematic."""
    click.echo("Available groove preparations:")
    click.echo("  V-groove  - classic single-bevel V (feaweld.geometry.groove.VGroove)")
    click.echo("  U-groove  - curved root U (feaweld.geometry.groove.UGroove)")
    click.echo("  J-groove  - asymmetric one-sided (feaweld.geometry.groove.JGroove)")
    click.echo("  X-groove  - double-V symmetric (feaweld.geometry.groove.XGroove)")
    click.echo("  K-groove  - single-bevel one-sided (feaweld.geometry.groove.KGroove)")


@main.command("j-integral")
@click.argument("results_file", type=click.Path(exists=True))
@click.option("--crack-tip", required=True, help="Crack tip coordinates as x,y[,z]")
@click.option("--radius", type=float, default=2.0, help="q-function radius (mm)")
def j_integral_cmd(results_file: str, crack_tip: str, radius: float) -> None:
    """Run the 2D J-integral on a saved VTK FE result."""
    try:
        import pyvista as pv
    except ImportError:
        click.echo("PyVista required: pip install feaweld[viz]")
        raise SystemExit(1)

    mesh = pv.read(results_file)
    tip = np.array([float(x) for x in crack_tip.split(",")])
    if tip.size == 2:
        tip = np.append(tip, 0.0)
    click.echo(f"Loaded mesh with {mesh.n_points} nodes, {mesh.n_cells} cells.")
    click.echo(f"Crack tip: {tip}")
    click.echo(f"Radius: {radius} mm")
    click.echo("j-integral CLI stub: full VTK-to-FEAResults conversion not in MVP.")
    click.echo("Use the library API directly: feaweld.fracture.j_integral_2d(...)")


if __name__ == "__main__":
    main()
