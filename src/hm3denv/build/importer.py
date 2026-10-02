"""``hm3d build-map``: GLB scenes -> a ready dataset in one command.

    hm3d build-map house.glb                          one file
    hm3d build-map D:\\scenes                          every .glb in a folder (and below)
    hm3d build-map D:\\hm3d\\val --name hm3d-val        an HM3D split: <id>-<hash>/<hash>.glb folders
    hm3d build-map house.glb --env grid               grid maps (JetAuto Pro cells) instead of SVG

The GLBs are copied into the workspace (``data/raw/glb`` of the repository, created on first
use) under a scene name: the HM3D folder name (``00800-TEEsavR23oF``) for HM3D scenes,
else the file name cleaned up (letters, digits and '-'). Then the dataset ``--name``
(default ``my-maps``) is built with the same pipeline and defaults as the published
datasets: storeys sliced from the mesh, SVG maps or grids, tasks checked on the 3D mesh,
previews. If the dataset already exists the new scenes are added to it (hm3d extend):
its maps and tasks are kept and its scenes keep their split.

Meshes are read in metres; the up axis is detected (``--up z`` to force it).
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
from pathlib import Path

from .. import paths

log = logging.getLogger(__name__)
HM3D_DIR = re.compile(r"^\d{5}-[A-Za-z0-9]{11}$")
UP = {"x": 0, "y": 1, "z": 2}


def scene_name(glb: Path) -> str:
    """Scene name of a GLB: its HM3D folder name, else the cleaned file name. Never
    contains '_' (map ids are '<scene>_s<storey>')."""
    if HM3D_DIR.match(glb.parent.name):
        return glb.parent.name
    stem = glb.name[:-len(".glb")] if glb.name.lower().endswith(".glb") else glb.stem
    stem = re.sub(r"\.basis$", "", stem, flags=re.I)
    name = re.sub(r"-{2,}", "-", re.sub(r"[^A-Za-z0-9-]+", "-", stem)).strip("-")
    return name or "scene"


def find_glbs(inputs) -> list[Path]:
    """GLB files of the inputs (files or folders, searched recursively); semantic meshes
    of HM3D are skipped, and one mesh per HM3D scene folder is kept."""
    found = []
    for x in inputs:
        p = Path(x).expanduser()
        if p.is_file():
            if p.suffix.lower() != ".glb":
                raise ValueError(f"{p} is not a .glb file")
            found.append(p)
        elif p.is_dir():
            found += sorted(q for q in p.rglob("*") if q.suffix.lower() == ".glb")
        else:
            raise FileNotFoundError(f"{p} does not exist")
    out, per_scene = [], {}
    for g in found:
        if ".semantic" in g.name.lower():
            continue
        key = g.parent if HM3D_DIR.match(g.parent.name) else g
        if key in per_scene:                       # e.g. <hash>.glb next to <hash>.basis.glb
            continue
        per_scene[key] = g
        out.append(g)
    if not out:
        raise FileNotFoundError(f"no .glb file in {[str(x) for x in inputs]}")
    return out


def _sha1(p: Path) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def import_glbs(ws: Path, inputs, replace: bool = False) -> list[str]:
    """Copy the GLBs into <ws>/raw/glb/<scene>.glb; returns the scene names."""
    dst_dir = paths.glb_dir(ws)
    dst_dir.mkdir(parents=True, exist_ok=True)
    names, seen = [], {}
    for g in find_glbs(inputs):
        name = scene_name(g)
        if name in seen:
            raise ValueError(f"{g} and {seen[name]} both give the scene name {name!r}: rename one")
        seen[name] = g
        dst = dst_dir / f"{name}.glb"
        if dst.exists() and dst.resolve() != g.resolve():
            same = dst.stat().st_size == g.stat().st_size and _sha1(dst) == _sha1(g)
            if not same and not replace:
                raise FileExistsError(f"another {dst.name} is already in {dst_dir}; rename your file "
                                      "or use --replace")
            if not same:
                shutil.copy2(g, dst)
                for d in (paths.slice_dir(ws) / name,):         # stale slices of the old mesh
                    shutil.rmtree(d, ignore_errors=True)
        elif not dst.exists():
            shutil.copy2(g, dst)
        log.info("  %s <- %s", name, g)
        names.append(name)
    return names


def build_map(inputs, name: str = "my-maps", env: str = "svg", robots=None, up: str | None = None,
              ws=None, replace: bool = False, preview: bool = True, verify: bool = True) -> tuple[Path, list[str]]:
    """Import GLBs and build (or extend) dataset `name`. Returns (dataset dir, new scenes)."""
    from .pipeline import build
    ws = paths.find_workspace(ws) if ws else paths.ensure_workspace()
    scenes = import_glbs(ws, inputs, replace)
    root = ws / "datasets" / name
    if (root / "manifest.json").exists():
        from .extend import dataset_scenes, extend
        from ..dataset import Dataset
        new = sorted(set(scenes) - dataset_scenes(Dataset(root)))
        if robots:
            log.warning("--robots is ignored when adding to an existing dataset (use hm3d extend --robots)")
        if not new:
            log.info("every scene is already in %s", name)
            return root, []
        extend(str(root), ws=ws, in_place=True, scenes=new, verify=verify)
        return root, new
    cfg = {"name": name, "env": env, "scenes": scenes,
           "split": {"train": 0.7, "val": 0.15, "test": 0.15, "seed": 0}}
    if up:
        cfg["slice"] = {"up": UP[up]}
    if env == "svg":
        cfg["robots"] = list(robots) if robots else "all"
    else:
        if robots and len(robots) != 1:
            raise ValueError("a grid dataset is built for one robot: give one --robots")
        cfg["grid"] = {"robot": robots[0] if robots else "jetauto_pro"}
    if not verify:
        cfg["tasks"] = {"verify": False}
    return build(cfg, ws=ws, out=root, preview=preview), scenes
