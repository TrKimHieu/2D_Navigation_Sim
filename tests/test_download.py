"""`hm3d download` against a fake Hugging Face Hub (no network)."""

import shutil
from types import SimpleNamespace

import pytest

import hm3denv
from hm3denv import download as D
from hm3denv.cli import main


class _Gated(Exception):
    pass


def fake_hub(repo_dir, gated=False):
    """snapshot_download copies the requested top-level folders of a local 'repository'."""
    def snapshot_download(repo_id, repo_type, local_dir, token, allow_patterns):
        if gated:
            raise _Gated("gated")
        for pat in allow_patterns:
            src = repo_dir / pat.split("/")[0]
            if src.exists():
                shutil.copytree(src, local_dir / src.name)
    return SimpleNamespace(snapshot_download=snapshot_download,
                           errors=SimpleNamespace(GatedRepoError=_Gated, RepositoryNotFoundError=_Gated))


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HM3D_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HM3D_DATASETS", raising=False)
    return tmp_path / "home"


def test_download_then_load_by_name(svg_dataset, home, monkeypatch):
    monkeypatch.setattr(D, "_hub", lambda: fake_hub(svg_dataset.parent))
    assert main(["download", "svg-tiny"]) == 0
    ds = hm3denv.load_dataset("svg-tiny")
    assert ds.root == (home / "datasets" / "svg-tiny").resolve() and ds.validate() == []
    assert D.download("svg-tiny")[0] == ds.root          # already present: kept


def test_download_rejects_modified_files(svg_dataset, tmp_path, home, monkeypatch):
    repo = tmp_path / "repo"
    shutil.copytree(svg_dataset, repo / "svg-tiny")
    next((repo / "svg-tiny" / "tasks").rglob("*.json")).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(D, "_hub", lambda: fake_hub(repo))
    with pytest.raises(ValueError, match="do not match the manifest"):
        D.download("svg-tiny")


def test_gated_repo_explains_how_to_get_access(home, monkeypatch, tmp_path):
    monkeypatch.setattr(D, "_hub", lambda: fake_hub(tmp_path, gated=True))
    with pytest.raises(PermissionError, match="Request access"):
        D.download("svg-v1")
