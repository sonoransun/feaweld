"""CLI tests using click.testing.CliRunner."""

from __future__ import annotations

import textwrap

import pytest
import yaml
from click.testing import CliRunner

from feaweld.cli import main


@pytest.fixture
def runner():
    return CliRunner()


# ---------------------------------------------------------------------------
# Basic smoke tests
# ---------------------------------------------------------------------------


class TestBasicCLI:

    def test_version(self, runner):
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "feaweld" in result.output.lower() or "0." in result.output

    def test_help(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "feaweld" in result.output.lower()


# ---------------------------------------------------------------------------
# Validate command
# ---------------------------------------------------------------------------


class TestValidateCommand:

    def test_validate_valid_case(self, runner, tmp_path):
        """A minimal valid YAML case file should pass validation."""
        case_data = {
            "name": "test_case",
            "geometry": {
                "joint_type": "fillet_t",
                "base_thickness": 20.0,
            },
            "mesh": {
                "global_size": 2.0,
                "weld_toe_size": 0.2,
            },
            "solver": {
                "solver_type": "linear_elastic",
            },
        }
        case_file = tmp_path / "valid_case.yaml"
        case_file.write_text(yaml.dump(case_data))

        result = runner.invoke(main, ["validate", str(case_file)])
        assert result.exit_code == 0
        assert "OK" in result.output or "ok" in result.output.lower() or "test_case" in result.output

    def test_validate_invalid_yaml(self, runner, tmp_path):
        """A YAML file with invalid field values should fail validation."""
        case_file = tmp_path / "bad_case.yaml"
        # Write invalid YAML: joint_type that does not exist in the enum
        case_file.write_text(
            yaml.dump({"geometry": {"joint_type": "nonexistent_joint_type_xyz"}})
        )
        result = runner.invoke(main, ["validate", str(case_file)])
        # Should exit with error code or contain error output
        assert result.exit_code != 0 or "INVALID" in result.output or "error" in result.output.lower()


# ---------------------------------------------------------------------------
# Materials command
# ---------------------------------------------------------------------------


class TestMaterialsCommand:

    def test_materials_list(self, runner):
        result = runner.invoke(main, ["materials"])
        assert result.exit_code == 0
        assert "material" in result.output.lower() or "Available" in result.output


# ---------------------------------------------------------------------------
# Blodgett command
# ---------------------------------------------------------------------------


class TestBlodgettCommand:

    def test_blodgett_line(self, runner):
        result = runner.invoke(
            main,
            ["blodgett", "--geometry", "line", "--d", "100", "--throat", "5", "-P", "1000"],
        )
        assert result.exit_code == 0
        assert "A_w" in result.output
        assert "f_a" in result.output or "Stress" in result.output

    @pytest.mark.parametrize(
        "shape",
        ["line", "parallel", "c_shape", "l_shape", "box", "circular", "i_shape", "t_shape"],
    )
    def test_blodgett_all_shapes(self, runner, shape):
        args = ["blodgett", "--geometry", shape, "--d", "100", "--throat", "5"]
        if shape not in ("line", "circular"):
            args += ["--b", "50"]
        result = runner.invoke(main, args)
        assert result.exit_code == 0
        assert "A_w" in result.output


# ---------------------------------------------------------------------------
# Groove types command
# ---------------------------------------------------------------------------


class TestGrooveTypes:

    def test_groove_types(self, runner):
        result = runner.invoke(main, ["groove-types"])
        assert result.exit_code == 0
        assert "V-groove" in result.output
        assert "U-groove" in result.output


# ---------------------------------------------------------------------------
# Command group existence tests
# ---------------------------------------------------------------------------


class TestCommandGroups:

    def test_mesh_group_exists(self, runner):
        result = runner.invoke(main, ["mesh", "--help"])
        assert result.exit_code == 0
        assert "mesh" in result.output.lower()

    def test_queue_group_exists(self, runner):
        result = runner.invoke(main, ["queue", "--help"])
        assert result.exit_code == 0
        assert "queue" in result.output.lower()

    def test_twin_group_exists(self, runner):
        result = runner.invoke(main, ["twin", "--help"])
        assert result.exit_code == 0
        assert "twin" in result.output.lower()

    def test_study_group_exists(self, runner):
        result = runner.invoke(main, ["study", "--help"])
        assert result.exit_code == 0
        assert "study" in result.output.lower()

    def test_defects_group_exists(self, runner):
        result = runner.invoke(main, ["defects", "--help"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Export command error handling
# ---------------------------------------------------------------------------


class TestExportCommand:

    def test_export_invalid_format(self, runner, tmp_path):
        """Passing an invalid format should produce an error."""
        dummy_file = tmp_path / "dummy.vtk"
        dummy_file.write_text("")  # create a dummy file so click.Path(exists=True) passes
        result = runner.invoke(main, ["export", str(dummy_file), "--format", "parquet"])
        # "parquet" is not in the Choice list, so Click should reject it
        assert result.exit_code != 0
        assert "Invalid value" in result.output or "invalid" in result.output.lower() or "parquet" in result.output
