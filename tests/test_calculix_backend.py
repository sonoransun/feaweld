"""Tests for the CalculiX solver backend using mocked subprocess calls."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from feaweld.core.materials import Material
from feaweld.core.types import (
    BoundaryCondition,
    ElementType,
    FEAResults,
    FEMesh,
    LoadCase,
    LoadType,
    StressField,
)
from feaweld.solver.calculix_backend import (
    CalculiXBackend,
    generate_inp,
    parse_frd,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_frd(
    n_nodes: int,
    include_displacement: bool = True,
    include_stress: bool = False,
) -> str:
    """Generate a minimal ASCII .frd file for testing.

    The .frd format used by CalculiX:
    - ``  100C`` marks the start of a result block header.
    - `` -4`` lines carry the block name (e.g. DISPLACEMENT, STRESS).
    - `` -5`` lines list each component name.
    - `` -1`` lines carry per-node data: node ID in cols 3-13, then
      12-char-wide fields for each component value.
    """
    lines: list[str] = []
    lines.append("    1C")  # header (ignored by parser)
    lines.append("    2C")

    if include_displacement:
        lines.append("  100CL  101       1    1    1")
        lines.append(" -4  DISPLACEMENT       1    1")
        lines.append(" -5  D1                  1    1    0")
        lines.append(" -5  D2                  1    1    0")
        lines.append(" -5  D3                  1    1    0")
        for i in range(1, n_nodes + 1):
            dx = 0.001 * i
            dy = 0.002 * i
            dz = 0.0
            lines.append(f" -1{i:10d}{dx:12.5E}{dy:12.5E}{dz:12.5E}")

    if include_stress:
        lines.append("  100CL  102       1    1    1")
        lines.append(" -4  STRESS              1    1")
        lines.append(" -5  SXX                 1    1    0")
        lines.append(" -5  SYY                 1    1    0")
        lines.append(" -5  SZZ                 1    1    0")
        lines.append(" -5  SXY                 1    1    0")
        lines.append(" -5  SYZ                 1    1    0")
        lines.append(" -5  SXZ                 1    1    0")
        for i in range(1, n_nodes + 1):
            sxx = 100.0 + i
            syy = 50.0
            szz = 0.0
            sxy = 10.0
            syz = 0.0
            sxz = 0.0
            lines.append(
                f" -1{i:10d}"
                f"{sxx:12.5E}{syy:12.5E}{szz:12.5E}"
                f"{sxy:12.5E}{syz:12.5E}{sxz:12.5E}"
            )

    lines.append("    3C")  # end marker
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# .inp generation tests
# ---------------------------------------------------------------------------


class TestGenerateInpStatic:
    """Verify that .inp files for static analysis contain the expected sections."""

    def test_generate_inp_static(self, simple_plate_mesh, steel_material, simple_load_case, tmp_path):
        inp_path = tmp_path / "test_static.inp"
        generate_inp(
            mesh=simple_plate_mesh,
            material=steel_material,
            load_case=simple_load_case,
            path=inp_path,
            analysis="static",
        )
        content = inp_path.read_text()

        assert "*NODE" in content
        assert "*ELEMENT" in content
        assert "*MATERIAL" in content
        assert "*STEP" in content
        assert "*STATIC" in content
        assert "*END STEP" in content
        assert "*ELASTIC" in content
        assert "*DENSITY" in content
        # Should contain the correct number of node lines (4 nodes)
        node_lines = [l for l in content.splitlines() if l.strip() and not l.startswith("*") and not l.startswith("**")]
        # At least 4 node lines exist
        assert len(node_lines) >= simple_plate_mesh.n_nodes

    def test_generate_inp_thermal(self, simple_plate_mesh, steel_material, simple_load_case, tmp_path):
        inp_path = tmp_path / "test_thermal.inp"
        generate_inp(
            mesh=simple_plate_mesh,
            material=steel_material,
            load_case=simple_load_case,
            path=inp_path,
            analysis="thermal_steady",
        )
        content = inp_path.read_text()

        assert "*HEAT TRANSFER, STEADY STATE" in content
        assert "*INITIAL CONDITIONS, TYPE=TEMPERATURE" in content
        assert "NT" in content  # node file output for temperature
        assert "*NODE" in content
        assert "*ELEMENT" in content
        assert "*END STEP" in content

    def test_inp_element_type(self, simple_plate_mesh, steel_material, simple_load_case, tmp_path):
        """Element type mapping is applied correctly in the .inp file."""
        inp_path = tmp_path / "test_elem.inp"
        generate_inp(
            mesh=simple_plate_mesh,
            material=steel_material,
            load_case=simple_load_case,
            path=inp_path,
        )
        content = inp_path.read_text()
        # TRI3 should map to S3
        assert "TYPE=S3" in content

    def test_inp_node_sets_written(self, simple_plate_mesh, steel_material, simple_load_case, tmp_path):
        """All mesh node sets appear as *NSET blocks."""
        inp_path = tmp_path / "test_nset.inp"
        generate_inp(
            mesh=simple_plate_mesh,
            material=steel_material,
            load_case=simple_load_case,
            path=inp_path,
        )
        content = inp_path.read_text()
        for name in simple_plate_mesh.node_sets:
            assert f"*NSET, NSET={name}" in content


# ---------------------------------------------------------------------------
# .frd parser tests
# ---------------------------------------------------------------------------


class TestParseFrd:

    def test_parse_frd_displacement(self, tmp_path):
        frd_content = _make_synthetic_frd(n_nodes=4, include_displacement=True)
        frd_path = tmp_path / "test.frd"
        frd_path.write_text(frd_content)

        result = parse_frd(frd_path)

        assert "displacement" in result
        disp = result["displacement"]
        assert disp.shape == (4, 3)
        # Check first node displacement values
        np.testing.assert_allclose(disp[0, 0], 0.001 * 1, rtol=1e-3)
        np.testing.assert_allclose(disp[0, 1], 0.002 * 1, rtol=1e-3)

    def test_parse_frd_stress(self, tmp_path):
        frd_content = _make_synthetic_frd(n_nodes=4, include_stress=True, include_displacement=False)
        frd_path = tmp_path / "test_stress.frd"
        frd_path.write_text(frd_content)

        result = parse_frd(frd_path)

        assert "stress" in result
        stress = result["stress"]
        assert stress.shape == (4, 6)
        # Node 1: sxx = 101, syy = 50
        np.testing.assert_allclose(stress[0, 0], 101.0, rtol=1e-3)
        np.testing.assert_allclose(stress[0, 1], 50.0, rtol=1e-3)

    def test_parse_frd_both_fields(self, tmp_path):
        frd_content = _make_synthetic_frd(
            n_nodes=3, include_displacement=True, include_stress=True,
        )
        frd_path = tmp_path / "both.frd"
        frd_path.write_text(frd_content)

        result = parse_frd(frd_path)

        assert "displacement" in result
        assert "stress" in result
        assert result["displacement"].shape[0] == 3
        assert result["stress"].shape[0] == 3

    def test_parse_frd_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_frd(tmp_path / "nonexistent.frd")


# ---------------------------------------------------------------------------
# Full solve with mocked subprocess
# ---------------------------------------------------------------------------


class TestSolveStaticMocked:

    def test_solve_static_mocked(self, simple_plate_mesh, steel_material, simple_load_case, tmp_path):
        """Patch subprocess.run and write a synthetic .frd to verify FEAResults."""
        n_nodes = simple_plate_mesh.n_nodes
        frd_content = _make_synthetic_frd(
            n_nodes=n_nodes, include_displacement=True, include_stress=True,
        )

        def fake_run(cmd, **kwargs):
            # Write the synthetic .frd file where the backend expects it
            cwd = Path(kwargs.get("cwd", "."))
            job_name = cmd[2]  # ccx -i <job_name>
            frd_path = cwd / f"{job_name}.frd"
            frd_path.write_text(frd_content)
            return MagicMock(returncode=0, stdout="", stderr="")

        backend = CalculiXBackend(ccx_path="/usr/bin/ccx", work_dir=str(tmp_path))

        with patch("feaweld.solver.calculix_backend.subprocess.run", side_effect=fake_run):
            result = backend.solve_static(
                mesh=simple_plate_mesh,
                material=steel_material,
                load_case=simple_load_case,
                temperature=20.0,
            )

        assert isinstance(result, FEAResults)
        assert result.displacement is not None
        assert result.displacement.shape == (n_nodes, 3)
        assert result.stress is not None
        assert result.stress.values.shape == (n_nodes, 6)
        assert result.metadata["solver"] == "calculix"


class TestCcxNotFound:

    def test_ccx_not_found(self):
        """When _find_ccx raises, CalculiXBackend.solve_static should propagate the error."""
        backend = CalculiXBackend()

        with patch(
            "feaweld.solver.calculix_backend._find_ccx",
            side_effect=FileNotFoundError("ccx not found"),
        ):
            with pytest.raises(FileNotFoundError, match="ccx not found"):
                backend.solve_static(
                    mesh=FEMesh(
                        nodes=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
                        elements=np.array([[0, 1, 2]]),
                        element_type=ElementType.TRI3,
                    ),
                    material=Material(
                        name="Dummy",
                        density=7850,
                        elastic_modulus={20: 200000},
                        poisson_ratio={20: 0.3},
                    ),
                    load_case=LoadCase(name="dummy"),
                )


class TestScratchDir:

    def test_scratch_dir_uses_env(self, tmp_path, monkeypatch):
        """FEAWELD_TMPDIR environment variable should control scratch directory location."""
        scratch_base = tmp_path / "custom_scratch"
        scratch_base.mkdir()
        monkeypatch.setenv("FEAWELD_TMPDIR", str(scratch_base))

        backend = CalculiXBackend()
        scratch = backend._scratch_dir()

        assert str(scratch).startswith(str(scratch_base))

    def test_scratch_dir_with_work_dir(self, tmp_path):
        """If work_dir is provided it should be returned directly."""
        work = tmp_path / "workdir"
        work.mkdir()
        backend = CalculiXBackend(work_dir=str(work))
        assert backend._scratch_dir() == work
