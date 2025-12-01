"""Tests for CLI commands."""

import pytest
from click.testing import CliRunner
from sqlalchemy import create_engine

from gradualbench.cli import cli
from gradualbench.models import Base, Dataset, Prompt, get_session


@pytest.fixture
def runner():
    """Create a CLI runner."""
    return CliRunner()


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database."""
    db_path = tmp_path / "test.db"
    return f"sqlite:///{db_path}"


def test_cli_init(runner, temp_db):
    """Test database initialization command."""
    result = runner.invoke(cli, ["--db", temp_db, "init"])
    assert result.exit_code == 0
    assert "Database initialized" in result.output


def test_dataset_list_empty(runner, temp_db):
    """Test listing datasets when none exist."""
    runner.invoke(cli, ["--db", temp_db, "init"])
    result = runner.invoke(cli, ["--db", temp_db, "dataset", "list"])
    assert result.exit_code == 0
    assert "No datasets found" in result.output


def test_cli_help(runner):
    """Test CLI help command."""
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "GradualBench" in result.output


def test_dataset_help(runner):
    """Test dataset subcommand help."""
    result = runner.invoke(cli, ["dataset", "--help"])
    assert result.exit_code == 0
    assert "Manage datasets" in result.output


def test_inference_help(runner):
    """Test inference subcommand help."""
    result = runner.invoke(cli, ["inference", "--help"])
    assert result.exit_code == 0
    assert "Run model inference" in result.output


def test_evaluate_help(runner):
    """Test evaluate subcommand help."""
    result = runner.invoke(cli, ["evaluate", "--help"])
    assert result.exit_code == 0
    assert "Run evaluations" in result.output


def test_compare_help(runner):
    """Test compare subcommand help."""
    result = runner.invoke(cli, ["compare", "--help"])
    assert result.exit_code == 0
    assert "Compare and track" in result.output
