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
@click.option("--interactive/--static", default=False,
              help="Embed interactive Plotly figures (requires feaweld[viz])")
def run(case_file: str, output: str | None, report: bool, interactive: bool) -> None:
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
        report_path = generate_report(result, interactive=interactive)
        click.echo(f"\nReport: {report_path}")


@main.command()
@click.option("--geometry", "-g", type=click.Choice(
    ["line", "parallel", "c_shape", "l_shape", "box", "circular", "i_shape",
     "t_shape", "u_shape"]
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
        "u_shape": WeldGroupShape.U_SHAPE,
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


def _parse_vec3(value: str, option: str) -> tuple[float, float, float]:
    try:
        parts = tuple(float(x) for x in value.split(","))
    except ValueError:
        parts = ()
    if len(parts) != 3:
        raise click.ClickException(f"Bad {option} '{value}' (expected 'x,y,z')")
    return parts


@main.command()
@click.argument("results_file", type=click.Path(exists=True))
@click.option("--component", "-c", default="von_mises",
              type=click.Choice(["von_mises", "tresca", "xx", "yy", "zz",
                                 "xy", "yz", "xz", "principal_1",
                                 "principal_2", "principal_3"]),
              help="Stress component to visualize")
@click.option("--output", "-o", default=None, help="Output image file (PNG)")
@click.option("--clip", default=None, help="Clipping plane normal (x,y,z)")
@click.option("--clip-origin", default=None, help="Clipping plane origin (x,y,z)")
@click.option("--threshold", type=float, default=None,
              help="Stress threshold for filtering")
@click.option("--below", is_flag=True,
              help="With --threshold, keep values below instead of above")
@click.option("--iso", default=None,
              help="Comma-separated iso-surface levels, e.g. '150,200,250'")
@click.option("--deformed", type=float, default=None,
              help="Warp by the 'displacement' array with this scale factor")
@click.option("--cmap", default=None,
              help="Colormap (semantic name like 'stress' or any matplotlib name)")
@click.option("--annotate", is_flag=True, help="Add critical point annotations")
def visualize(results_file: str, component: str, output: str | None,
              clip: str | None, clip_origin: str | None,
              threshold: float | None, below: bool, iso: str | None,
              deformed: float | None, cmap: str | None, annotate: bool) -> None:
    """Visualize FEA results from a VTK/VTU file.

    Renders through the feaweld visualization library (semantic theme
    colormaps, consistent annotations). Exactly one view mode applies,
    in precedence order: --iso, --threshold, --clip, then plain contour;
    --deformed and --annotate combine with the plain contour view.
    """
    try:
        import pyvista as pv
    except ImportError:
        click.echo("PyVista required: pip install feaweld[viz]")
        raise SystemExit(1)

    from feaweld.visualization import enhanced_3d, export, stress_plots
    from feaweld.visualization.theme import get_cmap

    grid = pv.read(results_file)
    scalar_key = stress_plots.resolve_component(component)
    if scalar_key not in grid.array_names:
        raise click.ClickException(
            f"Array '{scalar_key}' not in file. Available: "
            f"{', '.join(grid.array_names) or '(none)'}"
        )

    if cmap:
        try:
            colormap = get_cmap(cmap)
        except (KeyError, ValueError):
            colormap = cmap
    else:
        colormap = get_cmap("stress")

    show = output is None
    common = {"component": component, "show": show, "cmap": colormap}

    if iso is not None:
        try:
            iso_values = [float(x) for x in iso.split(",")]
        except ValueError:
            raise click.ClickException(f"Bad --iso '{iso}' (expected 'v1,v2,...')")
        plotter = enhanced_3d.plot_iso_surface(grid, iso_values=iso_values, **common)
    elif threshold is not None:
        plotter = enhanced_3d.plot_stress_threshold(
            grid, threshold=threshold, above=not below, **common,
        )
    elif clip is not None:
        origin = _parse_vec3(clip_origin, "--clip-origin") if clip_origin else None
        plotter = enhanced_3d.plot_stress_with_clipping(
            grid,
            clip_normal=_parse_vec3(clip, "--clip"),
            clip_origin=origin,
            **common,
        )
    else:
        if deformed is not None:
            if "displacement" not in grid.array_names:
                raise click.ClickException(
                    "No 'displacement' array in file for --deformed."
                )
            grid = grid.warp_by_vector("displacement", factor=deformed)
        if annotate:
            plotter = enhanced_3d.plot_annotated_stress(grid, **common)
        else:
            plotter = stress_plots.plot_stress_field(grid, **common)

    if output:
        export.export_png(plotter, output)
        click.echo(f"Saved: {output}")


@main.command()
@click.option("--power", "-P", type=float, default=4000.0, help="Net power (W)")
@click.option("--a-f", type=float, default=5.0, help="Front semi-axis (mm)")
@click.option("--a-r", type=float, default=10.0, help="Rear semi-axis (mm)")
@click.option("--b", type=float, default=4.0, help="Width semi-axis (mm)")
@click.option("--c", type=float, default=3.0, help="Depth semi-axis (mm)")
@click.option("--speed", type=float, default=5.0, help="Travel speed (mm/s)")
@click.option("--time", "-t", "t_snapshot", type=float, default=2.0,
              help="Snapshot time in seconds for the Goldak render")
@click.option("--iso", type=float, default=0.1,
              help="Iso-surface level as fraction of peak q")
@click.option("--output", "-o", default=None, help="Save screenshot PNG")
def goldak(power: float, a_f: float, a_r: float, b: float, c: float,
           speed: float, t_snapshot: float, iso: float,
           output: str | None) -> None:
    """Render a Goldak double-ellipsoid heat source as a 3-D iso-surface."""
    try:
        import pyvista as pv  # noqa: F401
    except ImportError:
        click.echo("PyVista required: pip install feaweld[viz]")
        raise SystemExit(1)

    import numpy as np
    from feaweld.solver.thermal import GoldakHeatSource
    from feaweld.visualization.thermal_plots import render_goldak_source

    source = GoldakHeatSource(
        power=power, a_f=a_f, a_r=a_r, b=b, c=c,
        travel_speed=speed,
        start_position=np.zeros(3),
        direction=np.array([1.0, 0.0, 0.0]),
    )
    render_goldak_source(
        source, t=t_snapshot, iso_fraction=iso,
        show=output is None, screenshot=output,
    )
    if output:
        click.echo(f"Saved: {output}")


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
@click.argument("case_file", type=click.Path(exists=True))
@click.option("--output", "-o", default="damage.gif",
              help="Output animation file (.gif or .mp4)")
@click.option("--fps", default=10, type=int, help="Frames per second")
@click.option("--blocks", default=10, type=int,
              help="Number of synthetic load blocks if the case has none")
def animate(case_file: str, output: str, fps: int, blocks: int) -> None:
    """Animate Palmgren-Miner damage accumulation across load blocks.

    If the case exposes a sequence of rainflow-counted load blocks via
    ``postprocess_results["load_blocks"]`` those are animated directly.
    Otherwise a synthetic sequence of *blocks* uniform blocks is
    generated from the case's stress-range distribution for illustration.
    """
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        click.echo("matplotlib required: pip install feaweld[viz]")
        raise SystemExit(1)

    from feaweld.pipeline.workflow import load_case, run_analysis
    from feaweld.fatigue.sn_curves import parse_sn_spec
    from feaweld.visualization.fatigue_plots import animate_damage_evolution

    click.echo(f"Loading case: {case_file}")
    case = load_case(case_file)
    click.echo(f"Running analysis: {case.name}")
    result = run_analysis(case)

    sn = parse_sn_spec(case.postprocess.sn_curve)

    # Try to pull a real rainflow block sequence; otherwise synthesize.
    block_seq = None
    if result.postprocess_results:
        raw = result.postprocess_results.get("load_blocks")
        if isinstance(raw, list) and raw and isinstance(raw[0], list):
            block_seq = raw

    if block_seq is None:
        click.echo(f"No rainflow blocks on result — generating {blocks} synthetic blocks.")
        # Very simple synthetic: increasing block severity over time.
        max_stress = 100.0
        if result.fea_results is not None and result.fea_results.stress is not None:
            import numpy as np
            max_stress = float(np.max(result.fea_results.stress.von_mises) * 0.6)
        block_seq = [
            [(max_stress * (0.5 + 0.5 * (i / max(1, blocks - 1))), 0.0, 10.0)]
            for i in range(blocks)
        ]

    path = animate_damage_evolution(block_seq, sn, output, fps=fps)
    click.echo(f"Saved: {path}")


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
@click.option("--max-workers", "-j", default=None, type=int,
              help="Parallel worker count (overrides the study file's max_workers)")
@click.option("--output", "-o", default=None, help="Output directory for comparison report")
@click.option("--report/--no-report", default=True, help="Generate HTML comparison report")
def study_run(study_file: str, max_workers: int | None, output: str | None,
              report: bool) -> None:
    """Run a parametric study from a YAML definition file."""
    from feaweld.pipeline.study import load_study, Study

    click.echo(f"Loading study: {study_file}")
    config = load_study(study_file)

    workers = max_workers if max_workers is not None else config.max_workers

    click.echo(f"Study: {config.name} ({config.mode} mode)")
    s = Study(config.name, config.base_case)
    for p in config.parameters:
        s.vary(p.name, p.values)

    cases = s._generate_cases(config.mode)
    click.echo(f"Generated {len(cases)} cases, running with {workers} workers...")

    results = s.run(max_workers=workers, mode=config.mode)

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

    # Register each provided case explicitly; with no swept parameters the
    # study runs exactly these cases and nothing else. The base case is only a
    # required constructor argument here and is not itself run.
    s = Study("comparison", next(iter(cases.values())))
    for name, case in cases.items():
        s.add_case(name, case)

    results = s.run(max_workers=max_workers, mode="grid")

    click.echo(f"Completed in {results.elapsed_seconds:.1f}s")

    from feaweld.pipeline.comparison import generate_comparison_report
    try:
        path = generate_comparison_report(results, output, baseline=baseline)
        click.echo(f"Comparison report: {path}")
    except Exception as e:
        click.echo(f"Report generation failed: {e}")


# ---------------------------------------------------------------------------
# Probabilistic / reliability commands
# ---------------------------------------------------------------------------

@main.group()
def reliability() -> None:
    """Probabilistic and reliability analysis (Monte Carlo, FORM, Sobol)."""
    pass


@reliability.command("mc")
@click.argument("case_file", type=click.Path(exists=True))
@click.option("--samples", "-n", default=None, type=int, help="Number of MC samples")
@click.option("--seed", default=None, type=int, help="Random seed")
@click.option("--sobol", is_flag=True, help="Also compute Sobol sensitivity indices")
@click.option("--save-samples", default=None, type=click.Path(),
              help="Save samples/results arrays to a .npz file")
def reliability_mc(case_file: str, samples: int | None, seed: int | None,
                   sobol: bool, save_samples: str | None) -> None:
    """Monte Carlo analysis of the safety factor for a YAML case."""
    from feaweld.pipeline.workflow import load_case, run_probabilistic_case

    case = load_case(case_file)
    if samples is not None:
        case.probabilistic.n_samples = samples
    if seed is not None:
        case.probabilistic.seed = seed
    if sobol:
        case.probabilistic.sobol = True

    click.echo(f"Monte Carlo ({case.probabilistic.method}, "
               f"n={case.probabilistic.n_samples}) on: {case.name}")
    result = run_probabilistic_case(case)

    click.echo(f"\nResponse: {result['response']}")
    click.echo(f"  mean = {result['mean']:.4g}")
    click.echo(f"  std  = {result['std']:.4g}")
    click.echo(f"  cov  = {result['cov']:.4g}")
    click.echo("  percentiles:")
    for p, val in sorted(result["percentiles"].items()):
        click.echo(f"    P{p:<3d} = {val:.4g}")
    click.echo(f"  converged: {result['converged']} (n_effective={result['n_effective']})")

    if "sobol" in result:
        click.echo("\nSobol indices (first-order / total):")
        first = result["sobol"]["first_order"]
        total = result["sobol"]["total"]
        for name in sorted(total, key=lambda n: -total[n]):
            click.echo(f"  {name:>22s}: {first[name]:6.3f} / {total[name]:6.3f}")

    if save_samples:
        np.savez(
            save_samples,
            samples=result["samples"],
            results=result["results"],
            variable_names=np.array(result["variable_names"]),
        )
        click.echo(f"\nSamples saved: {save_samples}")


@reliability.command("form")
@click.argument("case_file", type=click.Path(exists=True))
def reliability_form(case_file: str) -> None:
    """FORM reliability index (beta) against yield for a YAML case."""
    from feaweld.pipeline.workflow import build_probabilistic_model, load_case
    from feaweld.probabilistic.sensitivity import reliability_index_form

    case = load_case(case_file)
    variables, analysis_func = build_probabilistic_model(case)

    # Limit state: failure when the safety factor drops below 1
    def limit_state(params: dict[str, float]) -> float:
        return analysis_func(params) - 1.0

    click.echo(f"FORM (HL-RF) reliability analysis on: {case.name}")
    result = reliability_index_form(variables, limit_state)

    click.echo(f"\n  beta = {result['beta']:.4f}")
    click.echo(f"  P_f  = {result['probability_of_failure']:.4e}")
    click.echo("  design point:")
    for name, val in result["design_point"].items():
        click.echo(f"    {name:>22s} = {val:.4g}")


@reliability.command("sobol")
@click.argument("case_file", type=click.Path(exists=True))
@click.option("--n-base", default=256, type=int,
              help="Base sample size (evaluations = n_base * (2k + 2))")
@click.option("--seed", default=None, type=int, help="Random seed")
def reliability_sobol(case_file: str, n_base: int, seed: int | None) -> None:
    """Sobol sensitivity indices of the safety factor for a YAML case."""
    from feaweld.pipeline.workflow import build_probabilistic_model, load_case
    from feaweld.probabilistic.sensitivity import sobol_indices

    case = load_case(case_file)
    variables, analysis_func = build_probabilistic_model(case)

    k = len(variables)
    click.echo(f"Sobol (Saltelli) analysis on: {case.name} "
               f"({n_base * (2 * k + 2)} evaluations, {k} variables)")
    indices = sobol_indices(variables, analysis_func, n_base=n_base, seed=seed)

    click.echo("\n  variable                first-order    total")
    total = indices["total"]
    for name in sorted(total, key=lambda n: -total[n]):
        click.echo(f"  {name:>22s}  {indices['first_order'][name]:11.4f}  {total[name]:8.4f}")


# ---------------------------------------------------------------------------
# ML fatigue prediction commands
# ---------------------------------------------------------------------------

@main.group()
def ml() -> None:
    """Machine-learning fatigue life prediction."""
    pass


def _load_fatigue_csv(path: str, target_column: str):
    """Load a CSV of named feature columns into FatigueFeatures."""
    from feaweld.ml.features import FatigueFeatures

    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    names = list(data.dtype.names)
    if target_column not in names:
        raise click.ClickException(
            f"Target column '{target_column}' not in CSV columns: {names}"
        )
    feature_names = [n for n in names if n != target_column]
    values = np.column_stack([data[n] for n in feature_names])
    target = np.asarray(data[target_column], dtype=np.float64)
    return FatigueFeatures(
        feature_names=feature_names, values=values, target=target,
    )


def _validate_feature_names(provided: list[str], model_names: list[str]) -> None:
    """Validate user-provided feature names against a model's features.

    Parameters
    ----------
    provided : list[str]
        Feature names supplied on the command line or as CSV headers.
    model_names : list[str]
        Feature names the model was trained on, in training order.

    Raises
    ------
    click.ClickException
        If any provided name is not a model feature. Model features that were
        not provided are reported via a yellow warning (they fall back to the
        model's default handling) but are not treated as an error.
    """
    known = set(model_names)
    unknown = [n for n in provided if n not in known]
    if unknown:
        raise click.ClickException(
            f"Unknown feature name(s): {', '.join(unknown)}. "
            f"Model features are: {', '.join(model_names)}"
        )
    missing = [n for n in model_names if n not in set(provided)]
    if missing:
        click.echo(click.style(
            f"Warning: {len(missing)} model feature(s) not provided, "
            f"using model defaults: {', '.join(missing)}",
            fg="yellow",
        ))


@ml.command("train")
@click.argument("data_csv", type=click.Path(exists=True))
@click.option("--model", "-m", "model_type", default="random_forest",
              type=click.Choice(["random_forest", "xgboost", "ensemble"]),
              help="Model type")
@click.option("--target", default="log_life",
              help="Target column name (log10 of fatigue life)")
@click.option("--output", "-o", default="fatigue_model.joblib",
              help="Path for the saved model")
def ml_train(data_csv: str, model_type: str, target: str, output: str) -> None:
    """Train a fatigue life predictor on a CSV of features + log10(N) target."""
    try:
        import sklearn  # noqa: F401
    except ImportError:
        click.echo("scikit-learn required: pip install feaweld[ml]")
        raise SystemExit(1)

    from feaweld.ml.models import FatiguePredictor, MLModelConfig

    features = _load_fatigue_csv(data_csv, target)
    click.echo(f"Training {model_type} on {features.values.shape[0]} samples, "
               f"{len(features.feature_names)} features")

    predictor = FatiguePredictor(MLModelConfig(model_type=model_type))
    metrics = predictor.train(features)

    click.echo(f"\n  RMSE (log10 N): {metrics['rmse']:.4f}")
    click.echo(f"  R^2:            {metrics['r2']:.4f}")
    cv = metrics["cv_scores"]
    click.echo(f"  CV RMSE:        {np.mean(cv):.4f} +/- {np.std(cv):.4f}")

    predictor.save(output)
    click.echo(f"\nModel saved: {output}")


@ml.command("predict")
@click.argument("model_file", type=click.Path(exists=True))
@click.option("--set", "-s", "assignments", multiple=True,
              help="Feature assignment, e.g. -s stress_range=120 (repeatable)")
@click.option("--input", "-i", "input_csv", default=None, type=click.Path(exists=True),
              help="CSV of feature rows to predict (headers must match features)")
def ml_predict(model_file: str, assignments: tuple[str, ...],
               input_csv: str | None) -> None:
    """Predict fatigue life with a trained model.

    Accepts either a base model saved by ``ml train`` or a fine-tuned model
    saved by ``ml transfer``. Provided feature names are validated against the
    model's features: unknown names are rejected and unspecified model features
    are reported (they fall back to the model's default handling).
    """
    try:
        import sklearn  # noqa: F401
    except ImportError:
        click.echo("scikit-learn required: pip install feaweld[ml]")
        raise SystemExit(1)

    from feaweld.ml.models import FatiguePredictor
    from feaweld.ml.transfer import TransferLearner

    # A base model round-trips through FatiguePredictor.load; a fine-tuned
    # model is a pickled TransferLearner. Try the former, fall back to the
    # latter so both artifacts are usable here.
    predictor: FatiguePredictor | TransferLearner = FatiguePredictor()
    try:
        predictor.load(model_file)
    except Exception:
        import joblib
        obj = joblib.load(model_file)
        if isinstance(obj, TransferLearner):
            predictor = obj
        else:
            raise click.ClickException(
                f"Could not load '{model_file}' as a fatigue model or a "
                f"fine-tuned TransferLearner."
            )

    model_names = predictor.feature_names

    def _echo_prediction(pred) -> None:
        lo, hi = pred.confidence_interval
        click.echo(f"  N = {pred.predicted_life:.3e} cycles "
                   f"(95% CI: {lo:.3e} .. {hi:.3e})")

    if input_csv:
        data = np.genfromtxt(input_csv, delimiter=",", names=True, dtype=np.float64)
        names = list(data.dtype.names)
        _validate_feature_names(names, model_names)
        rows = np.atleast_1d(data)
        click.echo(f"Predicting {len(rows)} rows from {input_csv}:")
        for i, row in enumerate(rows):
            pred = predictor.predict({n: float(row[n]) for n in names})
            click.echo(f"row {i}:")
            _echo_prediction(pred)
        return

    if not assignments:
        raise click.ClickException("Provide -s feature=value assignments or --input CSV.")

    features = {}
    for item in assignments:
        if "=" not in item:
            raise click.ClickException(f"Bad assignment '{item}' (expected name=value)")
        name, val = item.split("=", 1)
        features[name.strip()] = float(val)

    _validate_feature_names(list(features.keys()), model_names)

    pred = predictor.predict(features)
    _echo_prediction(pred)

    importances = sorted(pred.feature_importances.items(), key=lambda kv: -kv[1])
    click.echo("\nTop feature importances:")
    for name, imp in importances[:5]:
        click.echo(f"  {name:>28s}: {imp:.3f}")


@ml.command("transfer")
@click.argument("model_file", type=click.Path(exists=True))
@click.argument("data_csv", type=click.Path(exists=True))
@click.option("--target", default="log_life", help="Target column name")
@click.option("--output", "-o", default="fatigue_model_tuned.joblib",
              help="Path for the fine-tuned model")
def ml_transfer(model_file: str, data_csv: str, target: str, output: str) -> None:
    """Fine-tune a trained model on plant-specific data (transfer learning)."""
    try:
        import sklearn  # noqa: F401
    except ImportError:
        click.echo("scikit-learn required: pip install feaweld[ml]")
        raise SystemExit(1)

    import joblib
    from feaweld.ml.models import FatiguePredictor
    from feaweld.ml.transfer import TransferLearner

    base = FatiguePredictor()
    base.load(model_file)

    features = _load_fatigue_csv(data_csv, target)
    learner = TransferLearner(base)
    metrics = learner.fine_tune(features)

    click.echo(f"  RMSE base:      {metrics['rmse_base']:.4f}")
    click.echo(f"  RMSE corrected: {metrics['rmse_corrected']:.4f}")
    click.echo(f"  R^2 corrected:  {metrics['r2_corrected']:.4f}")

    joblib.dump(learner, output)
    click.echo(f"\nFine-tuned model saved: {output}")


# ---------------------------------------------------------------------------
# Multiscale command
# ---------------------------------------------------------------------------

@main.command()
@click.option("--grade", "-g", default="A36", help="Steel grade (CCT database key)")
@click.option("--cooling-rate", "-r", default=30.0, type=float,
              help="Cooling rate at 700 C (C/s)")
@click.option("--base-yield", default=250.0, type=float,
              help="Base metal yield strength (MPa)")
@click.option("--base-uts", default=400.0, type=float,
              help="Base metal ultimate strength (MPa)")
def multiscale(grade: str, cooling_rate: float, base_yield: float,
               base_uts: float) -> None:
    """Multiscale weld-zone property estimation (CCT -> phases -> properties)."""
    from feaweld.data.cct import get_cct_diagram
    from feaweld.multiscale.meso import WeldZone, estimate_zone_properties
    from feaweld.multiscale.micro import (
        HALL_PETCH_LOW_CARBON_STEEL, estimate_grain_size_from_cooling,
    )

    try:
        diagram = get_cct_diagram(grade)
    except (KeyError, FileNotFoundError) as e:
        from feaweld.data.cct import list_cct_grades
        raise click.ClickException(
            f"Unknown CCT grade '{grade}' ({e}). Available: {', '.join(list_cct_grades())}"
        )

    phases = diagram.predict_phases(cooling_rate)
    click.echo(f"CCT phase prediction for {grade} at {cooling_rate:.1f} C/s:")
    click.echo(f"  ferrite    = {phases.ferrite:.3f}")
    click.echo(f"  pearlite   = {phases.pearlite:.3f}")
    click.echo(f"  bainite    = {phases.bainite:.3f}")
    click.echo(f"  martensite = {phases.martensite:.3f}")

    grain = estimate_grain_size_from_cooling(cooling_rate)
    hp_yield = HALL_PETCH_LOW_CARBON_STEEL.yield_strength(grain)
    click.echo(f"\nGrain size (Hall-Petch): {grain:.1f} um -> "
               f"sigma_y ~ {hp_yield:.0f} MPa")

    click.echo(f"\nZone properties (base: sigma_y={base_yield:.0f}, "
               f"sigma_u={base_uts:.0f} MPa):")
    click.echo(f"  {'zone':>20s}  {'yield':>7s}  {'UTS':>7s}  {'HV':>5s}  {'grain':>7s}")
    for zone in WeldZone:
        props = estimate_zone_properties(zone, phases, base_yield, base_uts)
        click.echo(
            f"  {zone.value:>20s}  {props.yield_strength:7.0f}  "
            f"{props.ultimate_strength:7.0f}  {props.hardness_hv:5.0f}  "
            f"{props.grain_size_um:5.0f} um"
        )


# ---------------------------------------------------------------------------
# Convergence / submodel commands
# ---------------------------------------------------------------------------

@main.command()
@click.argument("case_file", type=click.Path(exists=True))
@click.option("--levels", default=3, type=int, help="Number of refinement levels (>= 3)")
@click.option("--ratio", default=2.0, type=float, help="Mesh size ratio between levels")
def convergence(case_file: str, levels: int, ratio: float) -> None:
    """Mesh convergence study (Richardson extrapolation + GCI).

    Runs the case at LEVELS successively coarser meshes and reports the
    grid convergence index of the peak von Mises stress. Requires gmsh
    and an FEA solver backend.
    """
    from feaweld.pipeline.workflow import load_case, run_analysis
    from feaweld.singularity.convergence import convergence_study

    if levels < 3:
        raise click.ClickException("Convergence study requires at least 3 levels.")

    case = load_case(case_file)
    case.postprocess.singularity_check = False  # avoid redundant coarse solves

    values: list[float] = []
    sizes: list[float] = []
    base_global = case.mesh.global_size
    base_toe = case.mesh.weld_toe_size

    for level in range(levels):
        factor = ratio ** level
        case.mesh.global_size = base_global * factor
        case.mesh.weld_toe_size = base_toe * factor
        click.echo(f"Level {level}: global={case.mesh.global_size:.3g} mm, "
                   f"toe={case.mesh.weld_toe_size:.3g} mm ...")
        result = run_analysis(case)
        if not result.success or result.fea_results is None or result.fea_results.stress is None:
            raise click.ClickException(
                f"Level {level} failed: {'; '.join(result.errors) or 'no stress result'}"
            )
        vm_max = float(np.max(result.fea_results.stress.von_mises))
        values.append(vm_max)
        sizes.append(case.mesh.weld_toe_size)
        click.echo(f"  max von Mises = {vm_max:.2f} MPa")

    conv = convergence_study(values, sizes)
    click.echo("\nConvergence study (peak von Mises):")
    click.echo(f"  extrapolated value = {conv.extrapolated_value:.2f} MPa")
    click.echo(f"  observed order     = {conv.convergence_order:.2f}")
    click.echo(f"  GCI                = {conv.gci * 100:.1f}%")
    status = "converged" if conv.is_converged else "NOT converged (possible singularity)"
    click.echo(f"  status             = {status}")


@main.command()
@click.argument("case_file", type=click.Path(exists=True))
@click.option("--center", required=True,
              help="Submodel centre coordinates 'x,y,z' (mm)")
@click.option("--radius", "-r", required=True, type=float,
              help="Submodel extraction radius (mm)")
@click.option("--refine", default=4, type=int, help="Mesh refinement factor")
@click.option("--backend", default="auto",
              type=click.Choice(["auto", "fenics", "calculix"]),
              help="FEA backend for the local re-solve")
def submodel(case_file: str, center: str, radius: float, refine: int,
             backend: str) -> None:
    """Global-local submodel analysis around a point of interest.

    Runs the global case, then re-solves a refined local region with
    cut-boundary displacements interpolated from the global solution.
    """
    from feaweld.core.materials import load_material
    from feaweld.pipeline.workflow import load_case, run_analysis
    from feaweld.singularity.submodeling import solve_submodel

    try:
        center_vals = [float(x) for x in center.split(",")]
    except ValueError:
        raise click.ClickException(
            f"Bad --center '{center}' (expected 'x,y' or 'x,y,z')"
        )
    if len(center_vals) not in (2, 3):
        raise click.ClickException(
            f"--center must have exactly 2 or 3 comma-separated values, "
            f"got {len(center_vals)}: '{center}'"
        )
    center_pt = np.array(center_vals, dtype=float)
    if center_pt.shape[0] == 2:
        center_pt = np.append(center_pt, 0.0)

    case = load_case(case_file)
    click.echo(f"Running global analysis: {case.name}")
    result = run_analysis(case)
    if result.fea_results is None or result.fea_results.displacement is None:
        raise click.ClickException(
            f"Global analysis failed: {'; '.join(result.errors) or 'no displacements'}"
        )

    global_vm = float(np.max(result.fea_results.stress.von_mises)) \
        if result.fea_results.stress is not None else float("nan")

    click.echo(f"Solving submodel (centre={center_pt.tolist()}, radius={radius}, "
               f"refine={refine})")
    sub = solve_submodel(
        result.fea_results, center_pt, radius,
        material=load_material(case.material.base_metal),
        refinement_factor=refine,
        backend=backend,
    )

    click.echo(f"\n  solve mode:     {sub.metadata.get('submodel_solve')}")
    click.echo(f"  submodel nodes: {sub.mesh.n_nodes}")
    if sub.stress is not None:
        sub_vm = float(np.max(sub.stress.von_mises))
        click.echo(f"  max von Mises:  global {global_vm:.2f} MPa -> "
                   f"submodel {sub_vm:.2f} MPa")
    residual = sub.metadata.get("boundary_bc_residual")
    if residual is not None:
        click.echo(f"  boundary BC residual: {residual:.3e} mm")


# ---------------------------------------------------------------------------
# Digital-twin commands
# ---------------------------------------------------------------------------

@main.group()
def twin() -> None:
    """Digital-twin monitoring: live dashboard and Bayesian model updating."""
    pass


@twin.command("dashboard")
@click.option("--host", default="localhost", help="WebSocket host")
@click.option("--port", default=8765, type=int, help="WebSocket port")
@click.option("--http-port", default=8766, type=int, help="HTTP port for the web UI")
@click.option("--demo", is_flag=True, help="Feed synthetic sensor data")
@click.option("--open/--no-open", "open_browser", default=True,
              help="Open the dashboard in a browser")
def twin_dashboard(host: str, port: int, http_port: int, demo: bool,
                   open_browser: bool) -> None:
    """Serve the live digital-twin web dashboard."""
    try:
        import websockets  # noqa: F401
    except ImportError:
        click.echo("websockets required: pip install feaweld[digital-twin]")
        raise SystemExit(1)

    from feaweld.digital_twin.serve import run_dashboard

    click.echo(f"Dashboard: http://localhost:{http_port}/dashboard.html "
               f"(WebSocket ws://{host}:{port})")
    if demo:
        click.echo("Demo mode: streaming synthetic weld sensor data. Ctrl-C to stop.")
    run_dashboard(
        ws_host=host, ws_port=port, http_port=http_port,
        demo=demo, open_browser=open_browser,
    )


@twin.command("update")
@click.option("--priors", "priors_file", required=True, type=click.Path(exists=True),
              help="YAML of priors: [{name, distribution, params: {...}}, ...]")
@click.option("--data", "data_csv", required=True, type=click.Path(exists=True),
              help="CSV of observations: columns 'value' and optional 'noise_std'")
@click.option("--walkers", default=16, type=int, help="MCMC walkers")
@click.option("--steps", default=200, type=int, help="MCMC steps")
@click.option("--burnin", default=50, type=int, help="Burn-in steps")
def twin_update(priors_file: str, data_csv: str, walkers: int, steps: int,
                burnin: int) -> None:
    """Bayesian update of model parameters from observed data (emcee MCMC).

    The forward model predicts each observation as the mean of the
    sampled parameters (suitable for direct parameter observation);
    for custom physics use the Python API
    (feaweld.digital_twin.bayesian.BayesianUpdater).
    """
    try:
        import emcee  # noqa: F401
    except ImportError:
        click.echo("emcee required: pip install feaweld[digital-twin]")
        raise SystemExit(1)

    import yaml
    from feaweld.digital_twin.bayesian import (
        BayesianUpdater, ObservedData, PriorSpec,
    )

    with open(priors_file) as f:
        prior_data = yaml.safe_load(f)
    if not isinstance(prior_data, list):
        raise click.ClickException("Priors YAML must be a list of prior specs.")
    priors = [
        PriorSpec(
            name=p["name"],
            distribution=p.get("distribution", "normal"),
            params=p.get("params", {}),
        )
        for p in prior_data
    ]

    raw = np.genfromtxt(data_csv, delimiter=",", names=True, dtype=np.float64)
    rows = np.atleast_1d(raw)
    values = np.asarray(rows["value"], dtype=np.float64).ravel()
    noise = float(np.mean(rows["noise_std"])) if "noise_std" in (raw.dtype.names or ()) else 0.05

    param_names = [p.name for p in priors]

    def forward_model(params: dict[str, float]):
        mean_val = float(np.mean([params[n] for n in param_names]))
        return np.full(values.shape, mean_val)

    updater = BayesianUpdater(
        priors, forward_model,
        n_walkers=walkers, n_steps=steps, n_burnin=burnin,
    )

    click.echo(f"Running MCMC: {walkers} walkers x {steps} steps "
               f"({len(values)} observations)")
    summary = updater.update(ObservedData(
        measurement_type="parameter",
        positions=np.zeros((len(values), 3)),
        values=values,
        uncertainty=noise,
        timestamp=0.0,
    ))

    click.echo("\nPosterior summary:")
    for name in param_names:
        mean = summary.means[name]
        std = summary.stds[name]
        click.echo(f"  {name:>22s} = {mean:.4g} +/- {std:.4g}")
    if summary.r_hat:
        worst = max(summary.r_hat.values())
        click.echo(f"  max R-hat = {worst:.3f} "
                   f"({'converged' if worst < 1.1 else 'NOT converged'})")


if __name__ == "__main__":
    main()
