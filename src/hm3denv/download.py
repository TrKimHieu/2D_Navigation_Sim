"""Download pre-built datasets from the Hugging Face Hub.

The datasets live in one Hugging Face dataset repository, one top-level folder per dataset
(``isb-svg-v1/``, ``svg-v1/``, ...). Access may be gated: request it on the repository page,
then log in with ``hf auth login`` (or pass a token). Needs ``pip install "hm3denv[hub]"``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from . import paths
from .dataset.schema import validate

REPO_ID = "TranKimHieu/2D_Navigation_Sim"
REPO_URL = f"https://huggingface.co/datasets/{REPO_ID}"


def _hub():
    try:
        import huggingface_hub
        import huggingface_hub.errors  # noqa: F401
    except ImportError:
        raise ImportError('downloading datasets needs huggingface_hub: pip install "hm3denv[hub]"') from None
    return huggingface_hub


def _access_error(e: Exception) -> PermissionError:
    return PermissionError(
        f"no access to {REPO_URL} ({type(e).__name__}). Request access on that page, wait for "
        "the approval, then log in with `hf auth login` (or pass --token).")


def available(token: str | None = None, repo_id: str = REPO_ID) -> list[str]:
    """Names of the datasets in the repository (its top-level folders)."""
    hub = _hub()
    try:
        tree = hub.HfApi(token=token).list_repo_tree(repo_id, repo_type="dataset")
        return sorted(e.path for e in tree if isinstance(e, hub.hf_api.RepoFolder)
                      and not e.path.startswith("."))
    except (hub.errors.GatedRepoError, hub.errors.RepositoryNotFoundError) as e:
        raise _access_error(e) from None


def download(names, dest: str | os.PathLike | None = None, token: str | None = None,
             force: bool = False, repo_id: str = REPO_ID) -> list[Path]:
    """Download datasets by name into ``dest`` (default: ``paths.download_dir()``, which
    ``load_dataset`` searches) and validate them. Existing datasets are kept unless ``force``."""
    hub = _hub()
    names = [names] if isinstance(names, str) else list(names)
    dest = Path(dest).expanduser() if dest else paths.download_dir()
    dest.mkdir(parents=True, exist_ok=True)
    todo = []
    for n in names:
        if (dest / n / "manifest.json").exists() and not force:
            continue
        if force and (dest / n).exists():
            shutil.rmtree(dest / n)
        todo.append(n)
    if todo:
        try:
            hub.snapshot_download(repo_id, repo_type="dataset", local_dir=dest, token=token,
                                  allow_patterns=[f"{n}/*" for n in todo])
        except (hub.errors.GatedRepoError, hub.errors.RepositoryNotFoundError) as e:
            raise _access_error(e) from None
    out = []
    for n in names:
        root = dest / n
        if not (root / "manifest.json").exists():
            raise FileNotFoundError(f"dataset '{n}' is not in {REPO_URL}; available: "
                                    f"{available(token, repo_id)}")
        problems = validate(root)
        if problems:
            raise ValueError(f"{n}: downloaded files do not match the manifest: {problems[:5]}")
        out.append(root.resolve())
    return out
