"""Where things live on disk.

Resolution order everywhere: explicit argument > environment variable > the
workspace found from the current directory > the data home. Datasets fall back last to
the small demo datasets shipped inside the package (``demo-svg``, ``demo-grid``).

    HM3D_WORKSPACE   workspace root (contains workspace.yaml)
    HM3D_DATASETS    extra dataset directories, separated by os.pathsep
    HM3D_HOME        the data home: <repo>/data for a source checkout, else ~/.hm3denv.
                     It is a workspace too (created on demand by `ensure_workspace`), so
                     downloaded and built datasets share <home>/datasets/.

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
WORKSPACE_DIRS = ("raw/glb", "cache", "stages/slice", "robots", "datasets", "configs")
WORKSPACE_YAML = """# HM3D workspace (hm3denv {version})
# raw/glb/        HM3D scenes (*.glb)
# cache/          mesh cache, downloaded robot models
# stages/slice/   storey maps + review.json
# robots/         your robot presets (override the packaged ones)
# datasets/       built and downloaded datasets
version: 1
"""


class WorkspaceNotFound(RuntimeError):
    pass


def repo_root() -> Path | None:
    """The source checkout hm3denv runs from (editable install), or None for a wheel."""
    root = Path(__file__).resolve().parents[2]
    py = root / "pyproject.toml"
    try:
        return root if py.exists() and 'name = "hm3denv"' in py.read_text(encoding="utf-8") else None
    except OSError:
        return None


def default_home() -> Path:
    """The data home: $HM3D_HOME > <repo>/data (source checkout) > ~/.hm3denv."""
    if os.environ.get("HM3D_HOME"):
        return Path(os.environ["HM3D_HOME"]).expanduser().resolve()
    repo = repo_root()
    return repo / "data" if repo else Path.home() / ".hm3denv"


def find_workspace(path: str | os.PathLike | None = None, required: bool = True) -> Path | None:
    """Workspace root: `path` > $HM3D_WORKSPACE > first parent of cwd holding workspace.yaml
    > the data home if it is a workspace."""
    if path:
        root = Path(path).expanduser().resolve()
    elif os.environ.get("HM3D_WORKSPACE"):
        root = Path(os.environ["HM3D_WORKSPACE"]).expanduser().resolve()
    else:
        root = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / MARKER).exists()), None)
        if root is None and (default_home() / MARKER).exists():
            root = default_home()
    if root is None or not (root / MARKER).exists():
        if required:
            raise WorkspaceNotFound(
                "No HM3D workspace found. Create one with `hm3d init <dir>` and set "
                "HM3D_WORKSPACE, or run from inside it.")
        return None
    return root


def init_workspace(root: str | os.PathLike) -> Path:
    """Create (or complete) a workspace at `root`, with the example build configs."""
    from . import __version__
    root = Path(root).expanduser().resolve()
    for d in WORKSPACE_DIRS:
        (root / d).mkdir(parents=True, exist_ok=True)
    y = root / MARKER
    if not y.exists():
        y.write_text(WORKSPACE_YAML.format(version=__version__), encoding="utf-8")
    for cfg in (Path(__file__).resolve().parent / "configs").glob("*.yaml"):   # build configs
        if not (root / "configs" / cfg.name).exists():
            (root / "configs" / cfg.name).write_text(cfg.read_text(encoding="utf-8"), encoding="utf-8")
    return root


def ensure_workspace() -> Path:
    """The workspace to build in: the one found by `find_workspace`, else the data home
    (created on first use)."""
    return find_workspace(required=False) or init_workspace(default_home())


def glb_dir(ws: Path) -> Path:
    return ws / "raw" / "glb"


def cache_dir(ws: Path) -> Path:
    return ws / "cache"


def slice_dir(ws: Path) -> Path:
    return ws / "stages" / "slice"


def user_robots_dir(ws: Path | None) -> Path | None:
    return ws / "robots" if ws else None


def download_dir() -> Path:
    """Where `hm3d download` puts datasets: <data home>/datasets (see `default_home`)."""
    return default_home() / "datasets"


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
    out: list[Path] = []
    for d in dirs:                       # the workspace may be the data home: no duplicates
        d = d.resolve()
        if d not in out:
            out.append(d)
    return out


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
