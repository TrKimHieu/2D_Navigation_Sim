"""The data home (<repo>/data for a checkout) is both the download folder and a workspace."""

from pathlib import Path

import pytest

from hm3denv import paths


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for k in ("HM3D_HOME", "HM3D_WORKSPACE", "HM3D_DATASETS"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.chdir(tmp_path)                       # no workspace above the cwd
    return monkeypatch


def test_checkout_uses_repo_data_folder(clean_env):
    repo = paths.repo_root()
    assert repo == Path(__file__).resolve().parents[1]
    assert paths.default_home() == repo / "data"
    assert paths.download_dir() == repo / "data" / "datasets"


def test_hm3d_home_overrides(clean_env, tmp_path):
    clean_env.setenv("HM3D_HOME", str(tmp_path / "h"))
    assert paths.download_dir() == (tmp_path / "h" / "datasets").resolve()


def test_ensure_workspace_creates_data_home_once(clean_env, tmp_path):
    clean_env.setenv("HM3D_HOME", str(tmp_path / "h"))
    assert paths.find_workspace(required=False) is None
    ws = paths.ensure_workspace()
    assert ws == (tmp_path / "h").resolve()
    assert (ws / "workspace.yaml").exists() and (ws / "raw" / "glb").is_dir()
    assert any((ws / "configs").glob("*.yaml"))
    (ws / "configs" / "svg_9robots.yaml").write_text("name: mine\n", encoding="utf-8")
    assert paths.ensure_workspace() == ws                     # idempotent, user edits kept
    assert (ws / "configs" / "svg_9robots.yaml").read_text(encoding="utf-8") == "name: mine\n"
    assert paths.find_workspace() == ws                       # found without HM3D_WORKSPACE


def test_dataset_dirs_have_no_duplicates(clean_env, tmp_path):
    clean_env.setenv("HM3D_HOME", str(tmp_path / "h"))
    paths.ensure_workspace()
    dirs = paths.dataset_dirs()
    assert len(dirs) == len(set(dirs))
    assert dirs[0] == (tmp_path / "h" / "datasets").resolve()
