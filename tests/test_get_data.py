"""get-data (scripts/get_data.py): exit codes and the default folder, with a fake Hub."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

hub = pytest.importorskip("huggingface_hub")

from hm3denv import download as D  # noqa: E402
from test_download import fake_hub  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "get_data.py"


@pytest.fixture
def get_data(monkeypatch, tmp_path):
    monkeypatch.setenv("HM3D_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HM3D_DATASETS", raising=False)
    spec = importlib.util.spec_from_file_location("get_data", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def run(*argv):
        monkeypatch.setattr(sys, "argv", ["get_data.py", *argv])
        with pytest.raises(SystemExit) as e:
            mod.main()
            raise SystemExit(0)
        return e.value.code
    return mod, run


class _Api:
    def __init__(self, status=None):
        self.status = status

    def __call__(self, token=None):
        return self

    def whoami(self):
        if self.status == 401:
            raise RuntimeError("401 Unauthorized") from None
        if self.status:
            raise ConnectionError("offline")
        return {"name": "tester"}


def _err401():
    e = RuntimeError("401 Unauthorized")
    e.response = SimpleNamespace(status_code=401)
    return e


def test_downloads_into_data_home(get_data, svg_dataset, monkeypatch, tmp_path):
    mod, run = get_data
    monkeypatch.setattr(hub, "get_token", lambda: "tok")
    monkeypatch.setattr(hub, "HfApi", _Api())
    monkeypatch.setattr(D, "_hub", lambda: fake_hub(svg_dataset.parent))
    assert run("svg-tiny") == 0
    assert (tmp_path / "home" / "datasets" / "svg-tiny" / "manifest.json").exists()


def test_invalid_token_exit_4(get_data, monkeypatch, capsys):
    mod, run = get_data
    monkeypatch.setattr(hub, "get_token", lambda: "bad")

    class Api(_Api):
        def whoami(self):
            raise _err401()
    monkeypatch.setattr(hub, "HfApi", Api())
    assert run("svg-tiny") == 4
    assert "invalid or expired" in capsys.readouterr().err


def test_not_logged_in_without_terminal_exit_4(get_data, monkeypatch, capsys):
    mod, run = get_data
    monkeypatch.setattr(hub, "get_token", lambda: None)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    assert run("svg-tiny") == 4
    assert "hf auth login" in capsys.readouterr().err


def test_no_access_exit_5(get_data, monkeypatch, tmp_path, capsys):
    mod, run = get_data
    monkeypatch.setattr(hub, "get_token", lambda: "tok")
    monkeypatch.setattr(hub, "HfApi", _Api())
    monkeypatch.setattr(D, "_hub", lambda: fake_hub(tmp_path, gated=True))
    assert run("svg-v1") == 5
    assert "access request form" in capsys.readouterr().err


def test_offline_exit_6(get_data, monkeypatch, capsys):
    mod, run = get_data
    monkeypatch.setattr(hub, "get_token", lambda: "tok")
    monkeypatch.setattr(hub, "HfApi", _Api(status=503))
    assert run("svg-tiny") == 6
    assert "internet connection" in capsys.readouterr().err
