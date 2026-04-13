"""Command-line interface for feaweld."""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np

from feaweld import __version__


@click.group()
@click.version_option(version=__version__)
@click.option("--verbose", "-v", count=True, help="Increase log verbosity (-v info, -vv debug)")
@click.option(
    "--log-format",
    type=click.Choice(["text", "json", "journal"]),
    default="text",
    help="Log output format",
)
@click.pass_context
def main(ctx: click.Context, verbose: int, log_format: str) -> None:
    """feaweld - Finite Element Analysis for weld joint stress and fatigue."""
    import logging
    from feaweld.core.logging import setup_logging

    level = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}.get(
        verbose, logging.DEBUG
    )
    setup_logging(
        level,
        use_journal=(log_format == "journal"),
        json_format=(log_format == "json"),
    )
    ctx.ensure_object(dict)


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


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@main.command()
@click.argument("case_file", type=click.Path(exists=True))
def validate(case_file: str) -> None:
    """Validate a YAML case file without running the analysis."""
    from feaweld.pipeline.workflow import load_case

    click.echo(f"Validating: {case_file}")
    try:
        case = load_case(case_file)
    except Exception as e:
        click.echo(click.style(f"INVALID: {e}", fg="red"))
        raise SystemExit(1)

    click.echo(click.style("YAML schema: OK", fg="green"))
    click.echo(f"  Name: {case.name}")
    click.echo(f"  Joint: {case.geometry.joint_type.value}")
    click.echo(f"  Material: {case.material.base_metal}")
    click.echo(f"  Solver: {case.solver.solver_type.value}")

    # Check material availability
    warnings: list[str] = []
    try:
        from feaweld.core.materials import load_material
        for name in (case.material.base_metal, case.material.weld_metal, case.material.haz):
            load_material(name)
    except Exception as e:
        warnings.append(f"Material lookup: {e}")

    # Check solver backend availability
    try:
        from feaweld.solver.backend import get_backend
        get_backend(case.solver.backend)
    except Exception as e:
        warnings.append(f"Solver backend ({case.solver.backend}): {e}")

    # Check numeric ranges
    if case.geometry.base_thickness <= 0:
        warnings.append("base_thickness must be positive")
    if case.mesh.global_size <= 0:
        warnings.append("global_size must be positive")
    if case.mesh.weld_toe_size <= 0:
        warnings.append("weld_toe_size must be positive")

    if warnings:
        click.echo(click.style(f"\n{len(warnings)} warning(s):", fg="yellow"))
        for w in warnings:
            click.echo(f"  - {w}")
    else:
        click.echo(click.style("All checks passed.", fg="green"))


# ---------------------------------------------------------------------------
# Mesh commands
# ---------------------------------------------------------------------------

@main.group()
def mesh() -> None:
    """Mesh generation and inspection utilities."""
    pass


@mesh.command("generate")
@click.argument("case_file", type=click.Path(exists=True))
@click.option("--output", "-o", default="mesh.vtk", help="Output mesh file (VTK)")
def mesh_generate(case_file: str, output: str) -> None:
    """Generate a mesh from a case file without running analysis."""
    from feaweld.pipeline.workflow import load_case, _build_geometry
    from feaweld.mesh.generator import generate_mesh, WeldMeshConfig

    case = load_case(case_file)
    click.echo(f"Building geometry: {case.geometry.joint_type.value}")
    joint = _build_geometry(case.geometry)

    click.echo(f"Generating mesh (global={case.mesh.global_size}, toe={case.mesh.weld_toe_size})")
    mesh_config = WeldMeshConfig(
        global_size=case.mesh.global_size,
        weld_toe_size=case.mesh.weld_toe_size,
        element_order=case.mesh.element_order,
        element_type_2d=case.mesh.element_type,
    )
    fe_mesh = generate_mesh(joint, mesh_config)

    click.echo(f"  Nodes:    {fe_mesh.n_nodes}")
    click.echo(f"  Elements: {fe_mesh.n_elements}")
    click.echo(f"  Type:     {fe_mesh.element_type.value}")

    import meshio
    cells = [(fe_mesh.element_type.value.lower(), fe_mesh.connectivity)]
    m = meshio.Mesh(points=fe_mesh.nodes, cells=cells)
    m.write(output)
    click.echo(f"Saved: {output}")


@mesh.command("inspect")
@click.argument("mesh_file", type=click.Path(exists=True))
def mesh_inspect(mesh_file: str) -> None:
    """Inspect a mesh file and report quality metrics."""
    import meshio

    click.echo(f"Reading: {mesh_file}")
    m = meshio.read(mesh_file)

    click.echo(f"  Nodes:    {m.points.shape[0]}")
    total_cells = sum(len(c.data) for c in m.cells)
    click.echo(f"  Elements: {total_cells}")
    for cell_block in m.cells:
        click.echo(f"  Type:     {cell_block.type} ({len(cell_block.data)} elements)")

    # Bounding box
    mins = m.points.min(axis=0)
    maxs = m.points.max(axis=0)
    click.echo(f"  Bounds:   x=[{mins[0]:.2f}, {maxs[0]:.2f}] "
               f"y=[{mins[1]:.2f}, {maxs[1]:.2f}]")
    if m.points.shape[1] > 2:
        click.echo(f"            z=[{mins[2]:.2f}, {maxs[2]:.2f}]")

    # Try to compute quality metrics
    try:
        from feaweld.mesh.quality import aspect_ratio
        from feaweld.mesh.convert import meshio_to_femesh
        fe_mesh = meshio_to_femesh(m)
        ratios = aspect_ratio(fe_mesh)
        click.echo(f"\n  Aspect ratio: min={ratios.min():.3f} max={ratios.max():.3f} "
                   f"mean={ratios.mean():.3f}")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Convergence study
# ---------------------------------------------------------------------------

@main.command()
@click.argument("case_file", type=click.Path(exists=True))
@click.option("--refinements", "-n", default=3, help="Number of refinement levels")
def convergence(case_file: str, refinements: int) -> None:
    """Run a mesh convergence study on a case file."""
    from feaweld.pipeline.workflow import load_case, run_analysis

    case = load_case(case_file)
    base_size = case.mesh.global_size

    click.echo(f"Convergence study: {case.name}")
    click.echo(f"  Refinement levels: {refinements}")
    click.echo(f"  Base mesh size: {base_size}")

    mesh_sizes: list[float] = []
    max_stresses: list[float] = []

    for i in range(refinements):
        factor = 2 ** i
        size = base_size / factor
        case_copy = case.model_copy(deep=True)
        case_copy.mesh.global_size = size
        case_copy.mesh.weld_toe_size = case.mesh.weld_toe_size / factor

        click.echo(f"\n  Level {i + 1}: mesh_size={size:.4f}")
        result = run_analysis(case_copy)

        if result.fea_results and result.fea_results.stress:
            max_vm = float(np.max(result.fea_results.stress.von_mises))
            n_nodes = result.mesh.n_nodes if result.mesh else 0
            click.echo(f"    Nodes: {n_nodes}, Max von Mises: {max_vm:.2f} MPa")
            mesh_sizes.append(size)
            max_stresses.append(max_vm)
        else:
            click.echo(f"    Failed: {result.errors}")

    if len(max_stresses) >= 3:
        try:
            from feaweld.singularity.convergence import richardson_extrapolation
            conv = richardson_extrapolation(mesh_sizes, max_stresses)
            click.echo(f"\n  Richardson extrapolation:")
            click.echo(f"    Extrapolated value: {conv.extrapolated_value:.4f} MPa")
            click.echo(f"    Convergence order:  {conv.order:.2f}")
            click.echo(f"    GCI (finest):       {conv.gci:.4f}")
        except Exception as e:
            click.echo(f"\n  Richardson extrapolation failed: {e}")
    elif len(max_stresses) >= 2:
        change = abs(max_stresses[-1] - max_stresses[-2]) / max(abs(max_stresses[-2]), 1e-12) * 100
        click.echo(f"\n  Change between last two levels: {change:.2f}%")


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

@main.command()
@click.argument("case_file", type=click.Path(exists=True))
@click.option("--param", required=True, help="Parameter dot-path (e.g. load.axial_force)")
@click.option("--range", "param_range", required=True,
              help="Values as start:stop:n_points (e.g. 10000:50000:5)")
def sensitivity(case_file: str, param: str, param_range: str) -> None:
    """Run a single-parameter sensitivity sweep."""
    from feaweld.pipeline.workflow import load_case
    from feaweld.pipeline.study import Study

    case = load_case(case_file)

    parts = param_range.split(":")
    if len(parts) != 3:
        click.echo("--range must be start:stop:n_points (e.g. 10000:50000:5)")
        raise SystemExit(1)

    start, stop, n = float(parts[0]), float(parts[1]), int(parts[2])
    values = list(np.linspace(start, stop, n))

    click.echo(f"Sensitivity: {param} = {values}")

    s = Study(f"sensitivity_{param}", case)
    s.vary(param, values)
    results = s.run(max_workers=min(n, 4), mode="grid")

    click.echo(f"\nCompleted in {results.elapsed_seconds:.1f}s")
    click.echo(f"\n{'Value':>15s}  {'Max vM (MPa)':>15s}  {'Status':>10s}")
    click.echo("-" * 45)
    for name, wr in sorted(results.results.items()):
        val_str = name.split("=")[-1] if "=" in name else name
        if wr.fea_results and wr.fea_results.stress:
            max_vm = float(np.max(wr.fea_results.stress.von_mises))
            click.echo(f"{val_str:>15s}  {max_vm:>15.2f}  {'OK':>10s}")
        else:
            click.echo(f"{val_str:>15s}  {'N/A':>15s}  {'FAIL':>10s}")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

@main.command()
@click.argument("results_file", type=click.Path(exists=True))
@click.option("--format", "fmt", type=click.Choice(["csv", "json"]),
              default="csv", help="Export format")
@click.option("--output", "-o", default=None, help="Output file path")
def export(results_file: str, fmt: str, output: str | None) -> None:
    """Export FEA results to CSV or JSON."""
    import meshio
    import json as json_mod

    m = meshio.read(results_file)
    out_path = output or f"{Path(results_file).stem}.{fmt}"

    if fmt == "csv":
        import csv
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["x", "y", "z"]
            arrays = {}
            for name in m.point_data:
                data = m.point_data[name]
                if data.ndim == 1:
                    header.append(name)
                    arrays[name] = data
                elif data.ndim == 2:
                    for j in range(data.shape[1]):
                        col = f"{name}_{j}"
                        header.append(col)
                        arrays[col] = data[:, j]
            writer.writerow(header)
            for i in range(m.points.shape[0]):
                row = list(m.points[i])
                for col in header[3:]:
                    row.append(float(arrays[col][i]))
                writer.writerow(row)
    elif fmt == "json":
        data = {
            "n_points": m.points.shape[0],
            "points": m.points.tolist(),
            "point_data": {
                name: arr.tolist() for name, arr in m.point_data.items()
            },
        }
        with open(out_path, "w") as f:
            json_mod.dump(data, f, indent=2)

    click.echo(f"Exported to: {out_path}")


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------

@main.command()
@click.argument("case_file", type=click.Path(exists=True))
@click.option("--output", "-o", default=None, help="Output JSON file for timing data")
def profile(case_file: str, output: str | None) -> None:
    """Profile an analysis case and report per-stage timing."""
    import time as time_mod
    import json as json_mod
    from feaweld.pipeline.workflow import load_case, run_analysis

    case = load_case(case_file)
    click.echo(f"Profiling: {case.name}")

    start = time_mod.perf_counter()
    result = run_analysis(case)
    total = time_mod.perf_counter() - start

    click.echo(f"\nTotal time: {total:.3f}s")
    click.echo(f"Errors: {len(result.errors)}")

    timing = {"total_seconds": total, "errors": len(result.errors)}

    if result.mesh:
        click.echo(f"Mesh: {result.mesh.n_nodes} nodes, {result.mesh.n_elements} elements")
        timing["mesh_nodes"] = result.mesh.n_nodes
        timing["mesh_elements"] = result.mesh.n_elements

    if result.fea_results and result.fea_results.stress:
        max_vm = float(np.max(result.fea_results.stress.von_mises))
        click.echo(f"Max von Mises: {max_vm:.2f} MPa")
        timing["max_von_mises_mpa"] = max_vm

    if output:
        with open(output, "w") as f:
            json_mod.dump(timing, f, indent=2)
        click.echo(f"Timing data: {output}")


# ---------------------------------------------------------------------------
# Digital twin commands
# ---------------------------------------------------------------------------

@main.group()
def twin() -> None:
    """Digital twin daemon and dashboard management."""
    pass


@twin.command("start")
@click.option("--host", default="localhost", help="MQTT broker host")
@click.option("--port", default=1883, type=int, help="MQTT broker port")
def twin_start(host: str, port: int) -> None:
    """Start the digital twin daemon (foreground)."""
    import os
    os.environ.setdefault("FEAWELD_MQTT_HOST", host)
    os.environ.setdefault("FEAWELD_MQTT_PORT", str(port))
    from feaweld.digital_twin.daemon import _run_daemon
    _run_daemon()


@twin.command("status")
def twin_status() -> None:
    """Check digital twin daemon health."""
    import subprocess
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "feaweld-twin"],
            capture_output=True, text=True,
        )
        status = result.stdout.strip()
        if status == "active":
            click.echo(click.style("feaweld-twin: active (running)", fg="green"))
        else:
            click.echo(f"feaweld-twin: {status}")
    except FileNotFoundError:
        click.echo("systemctl not available (not running under systemd?)")


# ---------------------------------------------------------------------------
# Job queue commands
# ---------------------------------------------------------------------------

@main.group()
def queue() -> None:
    """Analysis job queue management."""
    pass


@queue.command("submit")
@click.argument("case_file", type=click.Path(exists=True))
@click.option("--priority", "-p", default=0, type=int, help="Job priority (lower = higher priority)")
def queue_submit(case_file: str, priority: int) -> None:
    """Submit an analysis job to the queue."""
    from feaweld.pipeline.workflow import load_case
    from feaweld.pipeline.queue import AnalysisJobQueue

    case = load_case(case_file)
    q = AnalysisJobQueue()
    job_id = q.submit(case, priority=priority)
    click.echo(f"Job submitted: {job_id}")
    click.echo(f"  Case: {case.name}")
    click.echo(f"  Priority: {priority}")


@queue.command("status")
@click.option("--job-id", default=None, help="Specific job ID to check")
def queue_status(job_id: str | None) -> None:
    """Show job queue status."""
    from feaweld.pipeline.queue import AnalysisJobQueue

    q = AnalysisJobQueue()
    if job_id:
        status = q.get_status(job_id)
        click.echo(f"Job {job_id}: {status.value}")
    else:
        jobs = q.list_jobs()
        if not jobs:
            click.echo("No jobs in queue.")
            return
        click.echo(f"{'ID':>36s}  {'Status':>10s}  {'Priority':>8s}  {'Name':>20s}")
        click.echo("-" * 80)
        for j in jobs:
            click.echo(
                f"{j['id']:>36s}  {j['status']:>10s}  "
                f"{j['priority']:>8d}  {j.get('name', ''):>20s}"
            )


if __name__ == "__main__":
    main()
