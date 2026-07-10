"""Tests for the Click command-line interface."""

import re
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from feaweld.cli import main


@pytest.fixture
def runner():
    return CliRunner()


def test_help_lists_commands(runner):
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    for command in ("run", "blodgett", "reliability", "ml", "multiscale",
                    "convergence", "submodel", "twin", "study"):
        assert command in result.output


def test_blodgett_u_shape(runner):
    result = runner.invoke(main, [
        "blodgett", "-g", "u_shape", "--d", "100", "--b", "50",
        "-t", "5", "-P", "10000",
    ])
    assert result.exit_code == 0
    assert "U_SHAPE" in result.output
    assert "A_w" in result.output


def test_blodgett_all_shapes(runner):
    for shape in ("line", "parallel", "c_shape", "l_shape", "box",
                  "circular", "i_shape", "t_shape", "u_shape"):
        result = runner.invoke(main, [
            "blodgett", "-g", shape, "--d", "100", "--b", "50", "-t", "5",
        ])
        assert result.exit_code == 0, f"{shape}: {result.output}"


def test_fatigue_constant_amplitude_damage(runner):
    from feaweld.fatigue.sn_curves import parse_sn_spec

    result = runner.invoke(main, [
        "fatigue", "--stress-range", "100", "-n", "1e5", "-c", "IIW_FAT90",
    ])
    assert result.exit_code == 0, result.output
    expected = 1e5 / parse_sn_spec("IIW_FAT90").life(100.0)
    assert f"{expected:.4e}" in result.output


def test_fatigue_ec3_reference_life(runner):
    result = runner.invoke(main, [
        "fatigue", "--stress-range", "90", "-c", "EC3_90",
    ])
    assert result.exit_code == 0, result.output
    # EC3 category 90 is anchored at 2e6 cycles for a 90 MPa range.
    assert "2.000e+06" in result.output


def test_fatigue_history_csv(runner, tmp_path):
    hist = tmp_path / "history.csv"
    hist.write_text("0\n100\n20\n120\n0\n80\n10\n90\n5\n")

    result = runner.invoke(main, [
        "fatigue", "--history", str(hist), "-c", "BS7608_D",
    ])
    assert result.exit_code == 0, result.output
    match = re.search(r"cycles counted = ([\d.]+)", result.output)
    assert match is not None, result.output
    assert float(match.group(1)) > 0
    assert "spectrum repeats" in result.output


def test_fatigue_list_curves(runner):
    result = runner.invoke(main, ["fatigue", "--list-curves"])
    assert result.exit_code == 0, result.output
    for standard in ("IIW", "DNV", "ASME", "EC3", "BS 7608", "AWS"):
        assert standard in result.output


def test_fatigue_requires_one_loading_style(runner):
    result = runner.invoke(main, ["fatigue"])
    assert result.exit_code != 0
    assert "exactly one" in result.output


def test_fatigue_goodman_requires_sigma_u(runner):
    result = runner.invoke(main, [
        "fatigue", "--stress-range", "100", "--mean-correction", "goodman",
    ])
    assert result.exit_code != 0
    assert "--sigma-u" in result.output
    assert "goodman" in result.output


def test_materials_lists(runner):
    result = runner.invoke(main, ["materials"])
    assert result.exit_code == 0
    assert "A36" in result.output


@pytest.fixture
def case_yaml(tmp_path):
    from feaweld.pipeline.workflow import AnalysisCase, save_case
    case = AnalysisCase(
        name="cli_test",
        load={"axial_force": 50000.0},
        probabilistic={"enabled": True, "n_samples": 50, "seed": 11},
    )
    path = tmp_path / "case.yaml"
    save_case(case, path)
    return str(path)


def test_reliability_mc(runner, case_yaml, tmp_path):
    npz = tmp_path / "samples.npz"
    result = runner.invoke(main, [
        "reliability", "mc", case_yaml, "-n", "60", "--seed", "3",
        "--save-samples", str(npz),
    ])
    assert result.exit_code == 0, result.output
    assert "mean" in result.output
    assert "P50" in result.output
    data = np.load(npz, allow_pickle=False)
    assert data["samples"].shape[0] == 60
    assert data["results"].shape == (60,)


def test_reliability_form(runner, case_yaml):
    result = runner.invoke(main, ["reliability", "form", case_yaml])
    assert result.exit_code == 0, result.output
    assert "beta" in result.output
    assert "design point" in result.output


def test_reliability_sobol(runner, case_yaml):
    result = runner.invoke(main, [
        "reliability", "sobol", case_yaml, "--n-base", "8", "--seed", "1",
    ])
    assert result.exit_code == 0, result.output
    assert "first-order" in result.output


def test_multiscale(runner):
    result = runner.invoke(main, ["multiscale", "--grade", "A36",
                                  "--cooling-rate", "30"])
    assert result.exit_code == 0, result.output
    assert "martensite" in result.output
    assert "base_metal" in result.output


def test_multiscale_unknown_grade(runner):
    result = runner.invoke(main, ["multiscale", "--grade", "unobtainium"])
    assert result.exit_code != 0
    assert "Available" in result.output


@pytest.fixture
def fatigue_csv(tmp_path):
    rng = np.random.default_rng(0)
    n = 60
    sr = rng.uniform(50, 300, n)
    t = rng.uniform(5, 40, n)
    log_life = 12.0 - 3.0 * np.log10(sr) + 0.02 * t + rng.normal(0, 0.05, n)
    path = tmp_path / "fatigue.csv"
    lines = ["stress_range,plate_thickness,log_life"]
    lines += [f"{a:.4f},{b:.4f},{c:.4f}" for a, b, c in zip(sr, t, log_life)]
    path.write_text("\n".join(lines))
    return str(path)


def test_ml_train_predict_roundtrip(runner, fatigue_csv, tmp_path):
    pytest.importorskip("sklearn")
    model = tmp_path / "model.joblib"
    result = runner.invoke(main, [
        "ml", "train", fatigue_csv, "-m", "random_forest",
        "-o", str(model),
    ])
    assert result.exit_code == 0, result.output
    assert "R^2" in result.output
    assert model.exists()

    result = runner.invoke(main, [
        "ml", "predict", str(model),
        "-s", "stress_range=150", "-s", "plate_thickness=20",
    ])
    assert result.exit_code == 0, result.output
    assert "cycles" in result.output
    assert "importances" in result.output


def test_ml_predict_requires_input(runner, fatigue_csv, tmp_path):
    pytest.importorskip("sklearn")
    model = tmp_path / "model.joblib"
    runner.invoke(main, ["ml", "train", fatigue_csv, "-o", str(model)])
    result = runner.invoke(main, ["ml", "predict", str(model)])
    assert result.exit_code != 0


def test_ml_predict_rejects_unknown_feature(runner, fatigue_csv, tmp_path):
    pytest.importorskip("sklearn")
    model = tmp_path / "model.joblib"
    runner.invoke(main, ["ml", "train", fatigue_csv, "-o", str(model)])

    result = runner.invoke(main, [
        "ml", "predict", str(model),
        "-s", "stress_range=150", "-s", "plate_thickness=20",
        "-s", "bogus_feature=1.0",
    ])
    assert result.exit_code != 0
    assert "bogus_feature" in result.output
    assert "Model features" in result.output


def test_ml_predict_warns_missing_feature(runner, fatigue_csv, tmp_path):
    pytest.importorskip("sklearn")
    model = tmp_path / "model.joblib"
    runner.invoke(main, ["ml", "train", fatigue_csv, "-o", str(model)])

    # Provide only one of the two model features -> warning, still succeeds.
    result = runner.invoke(main, [
        "ml", "predict", str(model), "-s", "stress_range=150",
    ])
    assert result.exit_code == 0, result.output
    assert "Warning" in result.output
    assert "plate_thickness" in result.output  # named in the warning
    assert "cycles" in result.output


def test_ml_transfer_predict_roundtrip(runner, fatigue_csv, tmp_path):
    """A model fine-tuned by `ml transfer` is usable by `ml predict`."""
    pytest.importorskip("sklearn")
    model = tmp_path / "model.joblib"
    result = runner.invoke(main, ["ml", "train", fatigue_csv, "-o", str(model)])
    assert result.exit_code == 0, result.output

    tuned = tmp_path / "tuned.joblib"
    result = runner.invoke(main, [
        "ml", "transfer", str(model), fatigue_csv, "-o", str(tuned),
    ])
    assert result.exit_code == 0, result.output
    assert tuned.exists()

    result = runner.invoke(main, [
        "ml", "predict", str(tuned),
        "-s", "stress_range=150", "-s", "plate_thickness=20",
    ])
    assert result.exit_code == 0, result.output
    assert "cycles" in result.output


def test_twin_update_recovers_parameter(runner, tmp_path):
    pytest.importorskip("emcee")
    priors = tmp_path / "priors.yaml"
    priors.write_text(
        "- name: yield_strength\n"
        "  distribution: normal\n"
        "  params: {mean: 240.0, std: 20.0}\n"
    )
    obs = tmp_path / "obs.csv"
    obs.write_text("value,noise_std\n255,5\n248,5\n252,5\n249,5\n")

    result = runner.invoke(main, [
        "twin", "update", "--priors", str(priors), "--data", str(obs),
        "--walkers", "8", "--steps", "60", "--burnin", "20",
    ])
    assert result.exit_code == 0, result.output
    assert "yield_strength" in result.output
    assert "Posterior" in result.output


# ---------------------------------------------------------------------------
# animate: block-sequence sourcing
# ---------------------------------------------------------------------------

def _animate_setup(tmp_path, monkeypatch, postprocess_results):
    """Save a minimal case YAML and stub the heavy animate dependencies.

    Patches ``run_analysis`` to return a WorkflowResult carrying
    *postprocess_results* and ``animate_damage_evolution`` to capture the
    block sequence it receives.  Returns (case_path, captured_dict).
    """
    from feaweld.pipeline import workflow as workflow_mod
    from feaweld.pipeline.workflow import AnalysisCase, WorkflowResult, save_case
    from feaweld.visualization import fatigue_plots as fatigue_plots_mod

    case = AnalysisCase(name="animate_test")
    path = tmp_path / "case.yaml"
    save_case(case, path)

    monkeypatch.setattr(
        workflow_mod, "run_analysis",
        lambda c: WorkflowResult(case=c, postprocess_results=postprocess_results),
    )

    captured: dict = {}

    def fake_animate(load_blocks, sn_curve, output, fps=10, **kwargs):
        captured["blocks"] = [list(block) for block in load_blocks]
        return Path(output)

    monkeypatch.setattr(
        fatigue_plots_mod, "animate_damage_evolution", fake_animate,
    )
    return str(path), captured


def test_animate_uses_pipeline_rainflow_stash(runner, tmp_path, monkeypatch):
    """With only the workflow's postprocess_results["rainflow"] stash (a flat
    list of (range, mean, count) triples), animate converts it to one block
    per cycle family instead of synthesizing blocks."""
    pytest.importorskip("matplotlib")
    stash = [(160.0, 20.0, 4.0), (80.0, 0.0, 12.0)]
    case_path, captured = _animate_setup(
        tmp_path, monkeypatch, {"rainflow": stash},
    )

    result = runner.invoke(main, [
        "animate", case_path, "-o", str(tmp_path / "damage.gif"),
    ])
    assert result.exit_code == 0, result.output
    assert "synthetic" not in result.output
    assert captured["blocks"] == [[(160.0, 20.0, 4.0)], [(80.0, 0.0, 12.0)]]


def test_animate_load_blocks_takes_precedence(runner, tmp_path, monkeypatch):
    """An explicit load_blocks sequence is animated as-is, even when the
    rainflow stash is also present."""
    pytest.importorskip("matplotlib")
    blocks = [[(100.0, 0.0, 5.0)], [(120.0, 0.0, 5.0), (60.0, 0.0, 8.0)]]
    case_path, captured = _animate_setup(
        tmp_path, monkeypatch,
        {"load_blocks": blocks, "rainflow": [(999.0, 0.0, 1.0)]},
    )

    result = runner.invoke(main, [
        "animate", case_path, "-o", str(tmp_path / "damage.gif"),
    ])
    assert result.exit_code == 0, result.output
    assert "synthetic" not in result.output
    assert captured["blocks"] == blocks


def test_animate_synthetic_fallback(runner, tmp_path, monkeypatch):
    """Without load_blocks or a rainflow stash, animate synthesizes blocks."""
    pytest.importorskip("matplotlib")
    case_path, captured = _animate_setup(tmp_path, monkeypatch, {"other": 1})

    result = runner.invoke(main, [
        "animate", case_path, "-o", str(tmp_path / "damage.gif"),
        "--blocks", "4",
    ])
    assert result.exit_code == 0, result.output
    assert "synthetic" in result.output
    assert len(captured["blocks"]) == 4


def test_study_compare_runs_each_case_once(runner, tmp_path, monkeypatch):
    """`study compare` runs exactly the provided cases -- no phantom baseline."""
    from feaweld.pipeline.workflow import AnalysisCase, save_case
    from feaweld.pipeline import study as study_mod

    case_a = AnalysisCase(name="alpha", load={"axial_force": 10000.0})
    case_b = AnalysisCase(name="beta", load={"axial_force": 20000.0})
    path_a = tmp_path / "a.yaml"
    path_b = tmp_path / "b.yaml"
    save_case(case_a, path_a)
    save_case(case_b, path_b)

    captured: dict[str, list[str]] = {}

    def fake_run(self, max_workers=4, mode="grid", progress_callback=None):
        case_dict = self._generate_cases(mode)
        captured["names"] = list(case_dict.keys())
        return study_mod.StudyResults(
            study_name=self._name, cases=case_dict,
            results={}, errors={}, elapsed_seconds=0.0,
        )

    monkeypatch.setattr(study_mod.Study, "run", fake_run)

    result = runner.invoke(main, [
        "study", "compare", str(path_a), str(path_b),
        "-o", str(tmp_path / "out"),
    ])
    assert result.exit_code == 0, result.output
    assert captured["names"] == ["alpha", "beta"]
    assert "baseline" not in captured["names"]


def test_study_run_uses_config_max_workers(runner, tmp_path, monkeypatch):
    """`study run` honours the study file's max_workers when -j is absent,
    and lets -j override it when supplied."""
    from feaweld.pipeline.workflow import AnalysisCase
    from feaweld.pipeline import study as study_mod
    from feaweld.pipeline.study import StudyConfig, ParameterSweep, save_study

    config = StudyConfig(
        name="wtest",
        base_case=AnalysisCase(name="base"),
        parameters=[ParameterSweep(name="load.axial_force", values=[10000.0, 20000.0])],
        mode="grid",
        max_workers=2,
    )
    study_path = tmp_path / "study.yaml"
    save_study(config, study_path)

    captured: dict[str, int] = {}

    def fake_run(self, max_workers=4, mode="grid", progress_callback=None):
        captured["max_workers"] = max_workers
        return study_mod.StudyResults(
            study_name=self._name, cases={}, results={}, errors={}, elapsed_seconds=0.0,
        )

    monkeypatch.setattr(study_mod.Study, "run", fake_run)

    result = runner.invoke(main, ["study", "run", str(study_path), "--no-report"])
    assert result.exit_code == 0, result.output
    assert captured["max_workers"] == 2  # from the study file

    result = runner.invoke(main, [
        "study", "run", str(study_path), "-j", "5", "--no-report",
    ])
    assert result.exit_code == 0, result.output
    assert captured["max_workers"] == 5  # -j overrides the file


@pytest.mark.parametrize("bad_center", ["5", "1,2,3,4"])
def test_submodel_rejects_bad_center(runner, tmp_path, bad_center):
    """`submodel` rejects a --center that is not 2 or 3 values, before solving."""
    from feaweld.pipeline.workflow import AnalysisCase, save_case

    case = AnalysisCase(name="sub_test")
    path = tmp_path / "case.yaml"
    save_case(case, path)

    result = runner.invoke(main, [
        "submodel", str(path), "--center", bad_center, "-r", "2.0",
    ])
    assert result.exit_code != 0, result.output
    assert "center" in result.output.lower()


@pytest.mark.requires_gmsh
def test_run_command_without_solver(runner, tmp_path):
    """`feaweld run` degrades cleanly when no FEA backend is installed."""
    gmsh = pytest.importorskip("gmsh")  # noqa: F841
    from feaweld.pipeline.workflow import AnalysisCase, save_case
    case = AnalysisCase(name="cli_run_test", output_dir=str(tmp_path / "out"))
    path = tmp_path / "case.yaml"
    save_case(case, path)

    result = runner.invoke(main, ["run", str(path), "--no-report"])
    # Either a full solve (backend present) or a graceful error report
    assert result.exit_code == 0, result.output
    assert "Running analysis" in result.output
