"""Where things live on disk.

Resolution order everywhere: explicit argument > environment variable > the
workspace found from the current directory. Datasets fall back last to the small
demo datasets shipped inside the package (``demo-svg``, ``demo-grid``).

    HM3D_WORKSPACE   workspace root (contains workspace.yaml)
    HM3D_DATASETS    extra dataset directories, separated by os.pathsep
    HM3D_HOME        per-user data (default ~/.hm3denv): datasets/ holds `hm3d download` output

Workspace layout::

    workspace.yaml
    raw/glb/                 HM3D scenes (*.glb)
    cache/                   mesh .npz cache, downloaded robot models
    stages/slice/<scene>/    storey maps + review.json (manual review)
    robots/                  user robot presets (override the packaged ones)
    datasets/<name>/         built datasets
"""

from __future__ import annotations

import os
from pathlib import Path

MARKER = "workspace.yaml"
DEMO_DATASETS = Path(__file__).resolve().parent / "demo"   # packaged demo-svg, demo-grid


class WorkspaceNotFound(RuntimeError):
    pass


def find_workspace(path: str | os.PathLike | None = None, required: bool = True) -> Path | None:
    """Workspace root: `path` > $HM3D_WORKSPACE > first parent of cwd holding workspace.yaml."""
    if path:
        root = Path(path).expanduser().resolve()
    elif os.environ.get("HM3D_WORKSPACE"):
        root = Path(os.environ["HM3D_WORKSPACE"]).expanduser().resolve()
    else:
        root = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / MARKER).exists()), None)
    if root is None or not (root / MARKER).exists():
        if required:
            raise WorkspaceNotFound(
                "No HM3D workspace found. Create one with `hm3d init <dir>` and set "
                "HM3D_WORKSPACE, or run from inside it.")
        return None
    return root


def glb_dir(ws: Path) -> Path:
    return ws / "raw" / "glb"


def cache_dir(ws: Path) -> Path:
    return ws / "cache"


def slice_dir(ws: Path) -> Path:
    return ws / "stages" / "slice"


def user_robots_dir(ws: Path | None) -> Path | None:
    return ws / "robots" if ws else None


def download_dir() -> Path:
    """Where `hm3d download` puts datasets: $HM3D_HOME/datasets (default ~/.hm3denv/datasets)."""
    home = os.environ.get("HM3D_HOME") or Path.home() / ".hm3denv"
    return Path(home).expanduser() / "datasets"


def dataset_dirs(extra: str | os.PathLike | None = None) -> list[Path]:
    """Directories searched for datasets, in priority order."""
    dirs: list[Path] = []
    if extra:
        dirs.append(Path(extra).expanduser())
    for p in os.environ.get("HM3D_DATASETS", "").split(os.pathsep):
        if p:
            dirs.append(Path(p).expanduser())
    ws = find_workspace(required=False)
    if ws:
        dirs.append(ws / "datasets")
    dirs.append(download_dir())
    dirs.append(DEMO_DATASETS)
    return [d.resolve() for d in dirs]


def resolve_dataset(name_or_path: str | os.PathLike, search: str | os.PathLike | None = None) -> Path:
    """A dataset directory from a name (searched in dataset_dirs) or a path."""
    p = Path(name_or_path).expanduser()
    if (p / "manifest.json").exists():
        return p.resolve()
    for d in dataset_dirs(search):
        if (d / str(name_or_path) / "manifest.json").exists():
            return (d / str(name_or_path)).resolve()
    found = sorted({q.parent.name for d in dataset_dirs(search) if d.exists()
                    for q in d.glob("*/manifest.json")})
    raise FileNotFoundError(
        f"Dataset '{name_or_path}' not found. Searched: "
        f"{[str(d) for d in dataset_dirs(search)] or 'nothing (set HM3D_DATASETS)'}. "
        f"Available: {found or 'none'}.")
