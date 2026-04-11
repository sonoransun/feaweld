"""CLI smoke tests for Track H: defects group, groove-types, version."""

from __future__ import annotations

from click.testing import CliRunner

from feaweld.cli import main


def test_cli_defects_list_prints_iso5817():
    runner = CliRunner()
    result = runner.invoke(main, ["defects", "list"])
    assert result.exit_code == 0, result.output
    assert "Level B" in result.output


def test_cli_defects_list_with_level():
    runner = CliRunner()
    result = runner.invoke(main, ["defects", "list", "--level", "B"])
    assert result.exit_code == 0, result.output


def test_cli_groove_types():
    runner = CliRunner()
    result = runner.invoke(main, ["groove-types"])
    assert result.exit_code == 0, result.output
    assert "V-groove" in result.output


def test_cli_version_unchanged():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
