"""Menu of every hm3denv feature (app.bat / ./app.sh). Each choice asks a few questions
(Enter keeps the default) and runs the same one-line command you could type yourself,
which is printed first so you can reuse it."""

import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WIN = os.name == "nt"


def ask(question, default=""):
    try:
        ans = input(f"  {question}" + (f" [{default}]" if default else "") + ": ").strip()
    except EOFError:
        ans = ""
    return ans or default


def run(feature, args):
    """Run a launcher of the repository root (get-data, sim, ...) with `args`."""
    if WIN:
        cmd = ["cmd", "/c", str(ROOT / f"{feature}.bat"), *args]
        shown = subprocess.list2cmdline([f"{feature}.bat", *args])
    else:
        cmd = [str(ROOT / f"{feature}.sh"), *args]
        shown = shlex.join([f"./{feature}.sh", *args])
    print(f"\n> {shown}\n")
    try:
        return subprocess.call(cmd, cwd=ROOT)
    except KeyboardInterrupt:
        return 130


def opt(flag, value):
    return [flag, value] if value else []


def get_data():
    names = ask("dataset name(s), empty = choose from the list", "")
    return run("get-data", names.split())


def sim_view():
    ds = ask("dataset (demo-svg, demo-grid, or one in data/datasets)", "demo-svg")
    args = ["--dataset", ds] + opt("--robot", ask("robot (empty = default)"))
    args += opt("--map", ask("map id (empty = random maps)"))
    if "--map" in args:
        start = ask("your own start x,y[,theta] (empty = tasks of the dataset)")
        if start:
            args += ["--start", start, "--goal", ask("goal x,y")]
    args += ["--num-envs", ask("number of environments", "1"),
             "--agent", ask("agent: oracle, random or module:function", "oracle"),
             "--episodes", ask("episodes (0 = until Ctrl+C)", "5"), "--view"]
    return run("sim", args)


def sim_serve():
    cfg = ask("session config (YAML file or packaged name: demo, custom_pairs)", "demo")
    args = [cfg, "--serve", ask("address", "tcp://127.0.0.1:5555")]
    if ask("also watch it in the browser? (y/n)", "y").lower().startswith("y"):
        args.append("--view")
    print("  Then drive it from your program, e.g. python examples/remote_client.py")
    return run("sim", args)


def train():
    cfg = ask("session config (empty = use dataset/robot below)")
    args = opt("--config", cfg)
    if not cfg:
        args += ["--dataset", ask("dataset", "demo-svg")] + opt("--robot", ask("robot (empty = default)"))
    args += ["--steps", ask("training steps", "200000")]
    return run("train", args)


def build_map():
    src = ask("GLB file or folder")
    if not src:
        print("  nothing to do")
        return 0
    args = [src, "--name", ask("dataset name", "my-maps"), "--env", ask("svg or grid", "svg")]
    robots = ask("robots, separated by spaces (empty = default)").split()
    if robots:
        args += ["--robots", *robots]
    return run("build-map", args)


def extend_data():
    name = ask("dataset to continue (name or path)")
    if not name:
        return 0
    run("extend-data", [name, "--status"])
    what = ask("add: [s]cenes from data/raw/glb, [t]asks per map, [r]obots, nothing", "nothing")
    args = [name]
    if what.startswith("s"):
        args.append("--new-scenes")
    elif what.startswith("t"):
        args += ["--add-tasks", ask("tasks to add per map", "10")]
    elif what.startswith("r"):
        args += ["--robots", *ask("robot ids").split()]
    else:
        return 0
    return run("extend-data", args)


def datasets():
    return subprocess.call([sys.executable, "-m", "hm3denv", "datasets"], cwd=ROOT)


MENU = [
    ("Download a dataset from Hugging Face            (get-data)", get_data),
    ("Open a simulation and watch it in the browser   (sim --view)", sim_view),
    ("Let your own program drive a simulation         (sim --serve)", sim_serve),
    ("Train a PPO agent                               (train)", train),
    ("Build maps from GLB files                       (build-map)", build_map),
    ("Continue a dataset: scenes, tasks, robots       (extend-data)", extend_data),
    ("List the datasets found                         (hm3d datasets)", datasets),
]


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return
    while True:
        print("\nhm3denv - indoor robot navigation simulator\n")
        for i, (label, _) in enumerate(MENU, 1):
            print(f"  {i}. {label}")
        print("  0. Quit")
        choice = ask("\nchoose", "0")
        if choice in ("0", "q", ""):
            return
        if not choice.isdigit() or not 1 <= int(choice) <= len(MENU):
            print("  unknown choice")
            continue
        code = MENU[int(choice) - 1][1]()
        if code:
            print(f"\n(exit code {code})")
        ask("press Enter to go back to the menu")


if __name__ == "__main__":
    main()
