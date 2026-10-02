"""Download full hm3denv datasets from Hugging Face into the folder hm3denv searches by name.

    .\\get-data.bat                        # choose from a list       (Linux/macOS: ./get-data.sh)
    .\\get-data.bat isb-svg-v1             # by name (several names allowed)
    .\\get-data.bat --all                  # everything you have access to

Datasets go to data/datasets/<name> in this repository ($HM3D_HOME/datasets if set, or --dir);
after that `dataset="<name>"` works everywhere (gym.make, sim, train, hm3d eval).

Exit codes: 0 ok, 2 unknown dataset or corrupt download, 3 huggingface_hub missing,
4 not logged in or invalid token, 5 no access to the datasets yet, 6 Hugging Face unreachable.
"""

import argparse
import sys


def fail(code: int, msg: str):
    sys.stderr.write(msg.rstrip() + "\n")
    sys.exit(code)


try:
    import huggingface_hub
except ImportError:
    fail(3, "error: huggingface_hub is not installed. Run .\\get-data.bat / ./get-data.sh "
            "(it installs it), or: pip install \"hm3denv[hub]\"")

from hm3denv import download as D
from hm3denv import paths
from hm3denv.dataset import load_dataset

ABOUT = {   # name -> what it is (shown in the menu)
    "isb-svg-v1": "continuous env, 204 maps from Isaac-Scene-Builder scenes, all 9 robots",
    "isb-grid-v1": "grid env, 202 maps from Isaac-Scene-Builder scenes, jetauto_pro",
    "svg-v1": "continuous env, 177 maps from real HM3D houses, all 9 robots (HM3D licence)",
    "grid-jetauto-v1": "grid env, 157 maps from real HM3D houses, jetauto_pro (HM3D licence)",
    "grid-s15-v1": "grid env with 15 cm cells, 166 HM3D maps (HM3D licence)",
}
ACCESS = f"""
error: you do not have access to the datasets yet. Once per account:
  1. Create a free account on https://huggingface.co (and log in).
  2. Open {D.REPO_URL}
     and fill in the access request form. HM3D-based datasets also need Matterport's HM3D
     licence (https://aihabitat.org/datasets/hm3d/).
  3. Wait for the approval e-mail, then run this command again.
"""
OFFLINE = "Check your internet connection (or proxy) and run this command again."


def ensure_login() -> str:
    """A valid Hugging Face token: the saved one (or $HF_TOKEN), else ask for one."""
    token = huggingface_hub.get_token()
    if not token:
        print("You are not logged in to Hugging Face.\n"
              "Create a token (type: Read) at https://huggingface.co/settings/tokens and paste it below.\n")
        if not sys.stdin.isatty():
            fail(4, "error: not logged in to Hugging Face and no terminal to ask for a token.\n"
                    "Run `hf auth login` first, or set the HF_TOKEN environment variable.")
        huggingface_hub.login()
        token = huggingface_hub.get_token()
        if not token:
            fail(4, "error: no Hugging Face token was saved.")
    try:
        user = huggingface_hub.HfApi(token=token).whoami()
    except Exception as e:                               # noqa: BLE001  any hub / network error
        if getattr(getattr(e, "response", None), "status_code", None) == 401:
            fail(4, "error: your Hugging Face token is invalid or expired.\n"
                    "Create a new one (type: Read) at https://huggingface.co/settings/tokens, then run\n"
                    "`hf auth login` (or set HF_TOKEN) and run this command again.")
        fail(6, f"error: cannot reach Hugging Face ({type(e).__name__}: {e}).\n{OFFLINE}")
    print(f"Logged in to Hugging Face as {user.get('name', '?')}.")
    return token


def choose(names: list[str]) -> list[str]:
    print("Datasets you can download:\n")
    for i, n in enumerate(names, 1):
        print(f"  {i}. {n:<16} {ABOUT.get(n, '')}")
    default = "isb-svg-v1" if "isb-svg-v1" in names else names[0]
    ans = input(f"\nNumber(s) or name(s), separated by spaces [{default}]: ").split() or [default]
    out = []
    for x in ans:
        n = names[int(x) - 1] if x.isdigit() and 0 < int(x) <= len(names) else x
        if n not in names:
            fail(2, f"error: unknown dataset: {x} (choose from {', '.join(names)})")
        out.append(n)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="*", help="dataset names (none: choose from a list)")
    p.add_argument("--all", action="store_true", help="download every dataset you have access to")
    p.add_argument("--dir", help=f"target folder (default: {paths.download_dir()})")
    p.add_argument("--force", action="store_true", help="download again even if present")
    a = p.parse_args()

    token = ensure_login()
    try:
        names = a.names
        if a.all or not names:
            avail = D.available(token)
            names = avail if a.all else choose(avail)
        dest = a.dir or paths.download_dir()
        print(f"\nDownloading {', '.join(names)} -> {dest} (checked against the sha256 manifest) ...")
        roots = D.download(names, a.dir, token, a.force)
    except PermissionError:
        fail(5, ACCESS)
    except (FileNotFoundError, ValueError) as e:      # unknown name, files not matching the manifest
        fail(2, f"error: {e}")
    except Exception as e:                             # noqa: BLE001  offline, proxy, timeout, server error
        fail(6, f"error: cannot reach Hugging Face ({type(e).__name__}: {e}).\n{OFFLINE}")
    print()
    for r in roots:
        print(f"  OK  {r.name:<16} {r}")
    n = roots[0].name
    try:
        ds = load_dataset(str(roots[0]))
        env, robot = ("HM3D/Grid-v0" if ds.env_type == "grid" else "HM3D/Svg-v0"), ds.robots[0]
    except (FileNotFoundError, ValueError, KeyError):
        env, robot = "HM3D/Svg-v0", "turtlebot4"
    print(f"""
Ready. Use it by name (Linux/macOS: ./sim.sh, ./train.sh):
  .\\sim.bat --dataset {n} --robot {robot} --view
  .\\train.bat --dataset {n} --robot {robot}
  gym.make("{env}", dataset="{n}", robot="{robot}")""")
    if a.dir:
        print(f"\nCustom folder: set HM3D_DATASETS={a.dir} so hm3denv finds it by name.")


if __name__ == "__main__":
    main()
