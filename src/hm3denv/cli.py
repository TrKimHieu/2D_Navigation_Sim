"""``hm3d`` command line.

Using datasets (no workspace needed):
    hm3d datasets                         list datasets found
    hm3d datasets validate NAME           check files against the manifest
    hm3d info NAME                        robots, splits, counts, flags
    hm3d download [NAME ...] [--dir D]    pre-built datasets from Hugging Face (no name: list)
    hm3d robots                           list robot presets
    hm3d robots check [ID|FILE ...]       validate presets
    hm3d eval NAME [--robot R] [--agent oracle|random|mod:fn] [--split S]
    hm3d render NAME --robot R --map M [--task I] [--out F]
    hm3d bench NAME [--robot R] [--num-envs N ...] [--envs-per-worker K] [--seconds S]
    hm3d cache build NAME [--robot R ...] [--split S] [--fields] [--jobs N]
    hm3d cache info | clear [NAME]        map cache on disk ($HM3D_CACHE)

Building datasets (workspace, ``pip install hm3denv[build]``):
    hm3d init DIR                         create a workspace
    hm3d slice [SCENE ...] [--height H | --robot R]
    hm3d review [--port P]
    hm3d build CONFIG [--force STAGE ...] [--out DIR] [--no-preview]
    hm3d verify NAME [--robot R]          re-check every task on the 3D mesh
    hm3d robots import-urdf ID ... [--dry-run]
    hm3d robots render [--out F]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from . import __version__, paths


def _ws(args, required=True):
    return paths.find_workspace(args.workspace, required=required)


def _print_table(rows, cols):
    if not rows:
        return
    w = [max(len(str(c)), *(len(str(r[i])) for r in rows)) for i, c in enumerate(cols)]
    print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(cols)))
    for r in rows:
        print("  ".join(str(v).ljust(w[i]) for i, v in enumerate(r)))


# ------------------------------------------------------------------ using

def cmd_datasets(args):
    from .dataset import list_datasets, load_dataset
    if args.action == "validate":
        if not args.name:
            sys.exit("usage: hm3d datasets validate NAME")
        ds = load_dataset(args.name, args.search)
        problems = ds.validate()
        for p in problems:
            print(p)
        print(f"{ds.name}: {'OK' if not problems else f'{len(problems)} problem(s)'}")
        return 1 if problems else 0
    rows = []
    for ds in list_datasets(args.search):
        r = ds.manifest["robots"]
        rows.append((ds.name, ds.env_type, len(ds.robots), sum(v["n_tasks"] for v in r.values()),
                     ds.manifest.get("created", ""), ds.root))
    if not rows:
        print("no datasets found; searched:", [str(d) for d in paths.dataset_dirs(args.search)])
        return 1
    _print_table(rows, ("name", "env", "robots", "tasks", "created", "path"))


def cmd_download(args):
    from . import download as D
    if not args.names:
        for n in D.available(args.token):
            print(n)
        return
    for root in D.download(args.names, args.dir, args.token, args.force):
        print(f"{root.name}: OK -> {root}")


def cmd_info(args):
    from .dataset import load_dataset
    ds = load_dataset(args.name, args.search)
    m = ds.manifest
    print(f"{ds.name}  env={ds.env_type}  schema={m['schema_version']}  tool={m.get('tool_version')}  "
          f"created={m.get('created')}\n  {ds.root}")
    print("splits: " + ", ".join(f"{k} {len(v)} scenes" for k, v in ds.splits.items()))
    rows = []
    for rid, v in sorted(m["robots"].items()):
        per = {s: len(ds.maps(rid, s)) for s in ("train", "val", "test")}
        rows.append((rid, v.get("height_class", ""), v["n_maps"], v["n_tasks"],
                     f"{per['train']}/{per['val']}/{per['test']}", len(v.get("flagged", []))))
    _print_table(rows, ("robot", "class", "maps", "tasks", "maps train/val/test", "flagged"))
    if args.flags:
        for rid, v in sorted(m["robots"].items()):
            for mid in v.get("flagged", []):
                print(f"  {rid} {mid}: {', '.join(v['maps'][mid]['flags'])}")


def cmd_robots(args):
    from . import robots as R
    if args.action in (None, "list"):
        rows = []
        for rid in R.list_robots():
            r = R.load(rid)
            fp = r.raw["footprint"]["type"]
            rows.append((rid, r.name, r.drive, f"{r.height:.3f}", r.height_class, fp,
                         f"{r.v_max:g}", f"{r.w_max:g}", R.preset_path(rid).parent.name))
        _print_table(rows, ("id", "name", "drive", "height", "class", "footprint", "v_max", "w_max", "source"))
    elif args.action == "check":
        bad = 0
        for item in args.items or R.list_robots():
            try:
                r = R.load(item)
                print(f"{r.id}: OK ({len(r.footprint)} vertices, r_circ {r.r_circ:.3f} m)")
            except (ValueError, FileNotFoundError, KeyError) as e:
                bad += 1
                print(f"{item}: {e}")
        return 1 if bad else 0
    elif args.action == "render":
        from .build.preview import robots_svg
        out = Path(args.out or "robots.svg")
        out.write_text(robots_svg([R.load(i) for i in (args.items or R.list_robots())]), encoding="utf-8")
        print(f"-> {out}")
    elif args.action == "import-urdf":
        from .build.urdf import MODELS, import_urdf
        ws = _ws(args)
        ids = args.items or sorted(MODELS)
        dst = paths.user_robots_dir(ws)
        dst.mkdir(parents=True, exist_ok=True)
        for rid in ids:
            raw = json.loads(R.preset_path(rid).read_text(encoding="utf-8"))
            try:
                new, msg = import_urdf(raw, paths.cache_dir(ws) / "urdf", args.max_dev)
            except Exception as e:                         # one broken model must not stop the rest
                print(f"{rid}: ERROR {type(e).__name__}: {e}")
                continue
            print(msg)
            if not args.dry_run:
                R.validate(new)
                (dst / f"{rid}.json").write_text(json.dumps(new, indent=1, ensure_ascii=False), encoding="utf-8")
                print(f"  -> {dst / (rid + '.json')}")


def cmd_eval(args):
    from .dataset import load_dataset
    from .evaluate import evaluate, write_summary
    kw = {}
    if args.budget:
        kw["budget"] = args.budget
    s = evaluate(load_dataset(args.name, args.search), robot=args.robot, agent=args.agent, split=args.split, per_map=args.per_map,
                 seed=args.seed, out=args.out, trajectory=args.trajectory, **kw)
    print(json.dumps(s, indent=1))
    if args.summary:
        write_summary(s, args.summary)


def cmd_bench(args):
    from .bench import bench
    rows, results = [], []
    for n in args.num_envs:
        r = bench(args.name if not args.search else _dataset(args), robot=args.robot, num_envs=n,
                  seconds=args.seconds, map_repeat=args.map_repeat, split=args.split,
                  envs_per_worker=args.envs_per_worker)
        results.append(r)
        rows.append([n, f"{r['steps_per_s']:.0f}", f"{r['steps_per_s'] / n:.0f}",
                     f"{r['step_ms']:.3f}" if "step_ms" in r else "-",
                     f"{r['reset_ms']:.0f} / {r['reset_p95_ms']:.0f}" if r.get("reset_ms") else "-"])
    r = results[0]
    print(f"{r['dataset']}  robot={r['robot']}  split={r['split']}  map_repeat={r['map_repeat']}  "
          f"envs_per_worker={r['envs_per_worker']}  "
          f"backend={r['backend']}  cpus={r['cpu_count']}  python={r['python']}")
    _print_table(rows, ["envs", "step/s", "step/s/env", "step ms", "reset ms mean / p95"])
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1), encoding="utf-8")


def cmd_cache(args):
    from . import cache as C
    if args.action == "info":
        root = C.cache_root()
        print(f"cache root: {root or 'disabled (HM3D_CACHE=off)'}")
        _print_table([[e["dataset"], e["files"], f"{e['mb']:.0f}"] for e in C.entries()],
                     ["dataset", "files", "MB"])
        return
    if args.action == "clear":
        for d in C.clear(args.name):
            print(f"removed {d}")
        return
    if not args.name:
        raise ValueError("cache build needs a dataset name")
    if C.cache_root() is None:
        raise ValueError("the cache is disabled (HM3D_CACHE=off)")
    ds = _dataset(args)
    if ds.env_type != "svg":
        raise ValueError("only SVG datasets are cached (grid maps load in ~2 ms)")
    robots = args.robot or ds.robots
    jobs = [(str(ds.root), r, m, args.fields) for r in robots for m in ds.maps(r, args.split)]
    print(f"{len(jobs)} (robot, map) entries -> {C.cache_root()}")
    from concurrent.futures import ProcessPoolExecutor, as_completed
    failed = []
    with ProcessPoolExecutor(max_workers=args.jobs or min(8, os.cpu_count() or 1)) as ex:
        futs = {ex.submit(C.build_entry, *j): j for j in jobs}
        for done, fut in enumerate(as_completed(futs), 1):
            try:
                msg = fut.result()
            except Exception as e:                   # keep going; report at the end
                failed.append((futs[fut], e))
                msg = f"{futs[fut][2]}: FAILED ({e})"
            if done % 50 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)}  last: {msg}")
    for (_, r, m, _), e in failed:
        print(f"failed: {r} {m}: {e}", file=sys.stderr)
    return 1 if failed else 0


def _dataset(args):
    from .dataset import load_dataset
    return load_dataset(args.name, args.search)


def cmd_render(args):
    from .build import preview as PV
    from .dataset import load_dataset
    ds = load_dataset(args.name, args.search)
    robot = args.robot or (ds.robots[0] if len(ds.robots) == 1 else None)
    if robot is None:
        sys.exit(f"choose --robot: {ds.robots}")
    if ds.env_type == "svg" and not (args.out or "").endswith(".png"):
        svg, i, info = PV.episode_svg(ds, robot, args.map, args.task, args.every)
        out = Path(args.out or f"{robot}__{args.map}__task{i}.svg")
        out.write_text(svg, encoding="utf-8")
    else:
        from PIL import Image
        img, i, info = PV.episode_png(ds, robot, args.map, args.task)
        out = Path(args.out or f"{robot}__{args.map}__task{i}.png")
        Image.fromarray(img).save(out)
    print(f"task {i}: {info['termination']}, path {info['path_length']:.2f} m, spl {info['spl']:.3f} -> {out}")


# ------------------------------------------------------------------ building

WORKSPACE_YAML = """# HM3D workspace (hm3denv {version})
# raw/glb/        HM3D scenes (*.glb)
# cache/          mesh cache, downloaded robot models
# stages/slice/   storey maps + review.json
# robots/         your robot presets (override the packaged ones)
# datasets/       built datasets
version: 1
"""


def cmd_init(args):
    root = Path(args.dir).resolve()
    for d in ("raw/glb", "cache", "stages/slice", "robots", "datasets", "configs"):
        (root / d).mkdir(parents=True, exist_ok=True)
    y = root / paths.MARKER
    if not y.exists():
        y.write_text(WORKSPACE_YAML.format(version=__version__), encoding="utf-8")
    for cfg in (Path(__file__).resolve().parent / "configs").glob("*.yaml"):   # example configs
        if not (root / "configs" / cfg.name).exists():
            (root / "configs" / cfg.name).write_text(cfg.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"workspace ready: {root}\n  copy GLB files into {root / 'raw' / 'glb'}\n"
          f"  then: set HM3D_WORKSPACE={root}  (or run hm3d from inside it)")


def cmd_slice(args):
    from .build import slice as S
    from .robots import load
    ws = _ws(args)
    h = args.height or (load(args.robot).height if args.robot else 0.626)
    rows = S.run(ws, args.scenes or None, S.SliceParams(robot_height=h, slices=args.slices),
                 force=args.force)
    print(f"{len(rows)} storey maps ({S.height_tag(h)}) in {paths.slice_dir(ws)}")


def cmd_review(args):
    from .build.review import serve
    serve(_ws(args), args.port, not args.no_open)


def cmd_build(args):
    from .build.pipeline import build
    root = build(args.config, ws=args.workspace, out=args.out, force=args.force or (),
                 preview=not args.no_preview)
    from .dataset import Dataset
    ds = Dataset(root)
    print(f"{ds.name}: " + ", ".join(f"{r} {v['n_maps']} maps / {v['n_tasks']} tasks"
                                     for r, v in ds.manifest["robots"].items()) + f"\n  {root}")


def cmd_verify(args):
    from .build import verify as V
    from .dataset import load_dataset
    ds = load_dataset(args.name, args.search)
    src = ds.manifest.get("source") or {}
    if src.get("verified_3d") is False:
        raise ValueError(f"{ds.name} was built from a {src.get('type')} source: no 3D mesh to verify")
    ws = _ws(args)
    for rid in [args.robot] if args.robot else ds.robots:
        if ds.env_type == "grid":
            rows, cols = V.recheck_grid(ws, ds, rid), V.GRID_COLS
        else:
            rows, cols = V.recheck_svg(ws, ds, rid), V.SVG_COLS
        out = ds.root / "verify" / f"{rid}_recheck.csv"
        V.write_csv(rows, out, cols)
        print(rid, json.dumps(V.summarize(rows)), "->", out)


# ------------------------------------------------------------------ parser

def parser():
    p = argparse.ArgumentParser(prog="hm3d", description="HM3D navigation environments and dataset tools.",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--version", action="version", version=f"hm3denv {__version__}")
    p.add_argument("--workspace", "-w", help="workspace directory (default: $HM3D_WORKSPACE or cwd)")
    p.add_argument("--search", help="extra directory to search for datasets")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("datasets", help="list or validate datasets")
    s.add_argument("action", nargs="?", choices=["list", "validate"], default="list")
    s.add_argument("name", nargs="?")
    s.set_defaults(fn=cmd_datasets)

    s = sub.add_parser("download", help="download pre-built datasets from Hugging Face")
    s.add_argument("names", nargs="*", help="dataset names (none: list the available ones)")
    s.add_argument("--dir", help="target directory (default: $HM3D_HOME/datasets, searched by name)")
    s.add_argument("--token", help="Hugging Face token (default: the one from `hf auth login`)")
    s.add_argument("--force", action="store_true", help="download again even if present")
    s.set_defaults(fn=cmd_download)

    s = sub.add_parser("info", help="describe a dataset")
    s.add_argument("name")
    s.add_argument("--flags", action="store_true", help="list flagged maps")
    s.set_defaults(fn=cmd_info)

    s = sub.add_parser("robots", help="robot presets")
    s.add_argument("action", nargs="?", choices=["list", "check", "render", "import-urdf"])
    s.add_argument("items", nargs="*", help="robot ids or preset files")
    s.add_argument("--out")
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--max-dev", type=float, default=0.25,
                   help="import-urdf: max bounding-box deviation from the official dimensions")
    s.set_defaults(fn=cmd_robots)

    s = sub.add_parser("eval", help="evaluate an agent")
    s.add_argument("name")
    s.add_argument("--robot")
    s.add_argument("--agent", default="oracle")
    s.add_argument("--split", choices=["train", "val", "test"])
    s.add_argument("--per-map", type=int)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--budget", type=int, help="grid: step budget")
    s.add_argument("--out", help="episodes JSONL")
    s.add_argument("--trajectory", action="store_true", help="store trajectories in the JSONL")
    s.add_argument("--summary", help="write the summary JSON here")
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("render", help="render one oracle episode (SVG, or PNG)")
    s.add_argument("name")
    s.add_argument("--robot")
    s.add_argument("--map", required=True)
    s.add_argument("--task", type=int)
    s.add_argument("--every", type=int, default=15, help="SVG: draw the footprint every N steps")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_render)

    s = sub.add_parser("bench", help="measure environment throughput on this machine")
    s.add_argument("name")
    s.add_argument("--robot")
    s.add_argument("--split", choices=["train", "val", "test"], default="train")
    s.add_argument("--num-envs", type=int, nargs="+", default=[1],
                   help="one or more vector sizes; >1 uses one process per env")
    s.add_argument("--seconds", type=float, default=10.0, help="measuring time per size")
    s.add_argument("--map-repeat", type=int, default=1)
    s.add_argument("--envs-per-worker", type=int, default=1,
                   help="environments per worker process (fewer inter-process messages)")
    s.add_argument("--json", help="write the results here")
    s.set_defaults(fn=cmd_bench)

    s = sub.add_parser("cache", help="disk cache of loaded maps (build, info, clear)")
    s.add_argument("action", choices=["build", "info", "clear"])
    s.add_argument("name", nargs="?", help="dataset (build: required; clear: default all)")
    s.add_argument("--robot", nargs="+")
    s.add_argument("--split", choices=["train", "val", "test"])
    s.add_argument("--fields", action="store_true",
                   help="also store the goal distance field of every task (~3.5 MB per map)")
    s.add_argument("--jobs", type=int, help="worker processes (default: min(8, CPUs); each needs ~200 MB)")
    s.set_defaults(fn=cmd_cache)

    s = sub.add_parser("init", help="create a workspace")
    s.add_argument("dir")
    s.set_defaults(fn=cmd_init)

    s = sub.add_parser("slice", help="slice scenes into storey maps")
    s.add_argument("scenes", nargs="*")
    g = s.add_mutually_exclusive_group()
    g.add_argument("--height", type=float)
    g.add_argument("--robot")
    s.add_argument("--slices", action="store_true", help="also write layer images (for review)")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_slice)

    s = sub.add_parser("review", help="manual review web tool")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--no-open", action="store_true")
    s.set_defaults(fn=cmd_review)

    s = sub.add_parser("build", help="build a dataset from a config file")
    s.add_argument("config")
    s.add_argument("--force", nargs="*", metavar="STAGE",
                   help="re-run stages: slice gridify vectorize splits tasks (and what follows)")
    s.add_argument("--out", help="dataset directory (default: <workspace>/datasets/<name>)")
    s.add_argument("--no-preview", action="store_true")
    s.set_defaults(fn=cmd_build)

    s = sub.add_parser("verify", help="re-check every task on the 3D mesh")
    s.add_argument("name")
    s.add_argument("--robot")
    s.set_defaults(fn=cmd_verify)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose or args.cmd in ("build", "slice") else logging.WARNING,
                        format="%(message)s")
    try:
        return args.fn(args) or 0
    except (paths.WorkspaceNotFound, FileNotFoundError, FileExistsError, KeyError, ValueError,
            PermissionError, ImportError) as e:
        if args.verbose:
            raise
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
