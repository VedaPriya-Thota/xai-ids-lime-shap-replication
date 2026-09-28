from pathlib import Path

from src.config import PROJECT_ROOT, resolve_dir_or_default


def test_resolve_dir_or_default_falls_back_when_none():
    # No explicit override -> the historical/default relative path, resolved against the project root.
    assert resolve_dir_or_default(None, "results/experiments/phase3a_strict_reconstructed_mlp") == (
        PROJECT_ROOT / "results/experiments/phase3a_strict_reconstructed_mlp"
    )


def test_resolve_dir_or_default_uses_explicit_absolute_override(tmp_path):
    override = tmp_path / "some_rerun_dir"
    assert resolve_dir_or_default(str(override), "results/experiments/phase3a_strict_reconstructed_mlp") == override


def test_resolve_dir_or_default_resolves_explicit_relative_override_against_project_root():
    assert resolve_dir_or_default("results/experiments/phase3a_rerun", "results/experiments/phase3a_strict_reconstructed_mlp") == (
        PROJECT_ROOT / "results/experiments/phase3a_rerun"
    )


def test_resolve_dir_or_default_accepts_path_objects(tmp_path):
    default = Path("results/experiments/phase3a_strict_reconstructed_mlp")
    assert resolve_dir_or_default(None, default) == PROJECT_ROOT / default
    override = tmp_path / "override_dir"
    assert resolve_dir_or_default(override, default) == override
