"""Images for looking at a dataset by eye.

    grid_map_png    one grid: free white, wall black, nofloor grey, noroof light purple,
                    free cells outside the main region light grey; start green, goal red,
                    BFS paths light blue
    grid_sheets     contact sheets of a grid dataset: 2 cm source map cropped to the
                    grid (black obstacle, green roofed floor, purple unroofed floor) next
                    to the task preview; flagged maps first
    svg_sheets      contact sheets of an SVG dataset for one robot: free white, obstacle
                    dark grey, outdoor floor purple; real footprint at every start
                    (green, heading arrow) and the success disk at every goal (red)
    robots_svg      every robot footprint to scale (1:10) in one SVG
    episode_svg     one oracle episode as SVG (real map, footprint along the trajectory)
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .. import paths
from ..core.geometry import place
from ..core.gridcore import MOVES, bfs
from ..core.svgmap import SvgMap, rasterize
from ..robots import footprint_polygon, height_tag

TILE_W, HEAD_H, PAD = 640, 40, 10


def _font():
    try:
        return ImageFont.load_default(size=15)
    except TypeError:                      # Pillow < 10.1
        return ImageFont.load_default()


# ------------------------------------------------------------------ grid

def bfs_path(free, s, g):
    dist = bfs(free, g)
    path = [tuple(s)]
    r, c = s
    while (r, c) != tuple(g):
        for dr, dc in MOVES:
            nr, nc = r + dr, c + dc
            if 0 <= nr < free.shape[0] and 0 <= nc < free.shape[1] and dist[nr, nc] == dist[r, c] - 1:
                r, c = nr, nc
                break
        path.append((r, c))
    return path


def grid_map_png(state, main, tasks, path, px=10):
    """tasks: schema 2.0 grid tasks (start/goal carry "cell")."""
    col = np.array([[255, 255, 255], [30, 30, 30], [150, 150, 150], [205, 185, 235]], np.uint8)
    img = col[state]
    img[(state == 0) & ~main] = (215, 215, 215)
    im = Image.fromarray(img.repeat(px, 0).repeat(px, 1))
    d = ImageDraw.Draw(im)
    free = state == 0
    for t in tasks:
        pts = [(c * px + px / 2, r * px + px / 2)
               for r, c in bfs_path(free, t["start"]["cell"], t["goal"]["cell"])]
        d.line(pts, fill=(120, 160, 230), width=1)
    for t in tasks:
        for (r, c), fill in ((t["start"]["cell"], (0, 170, 0)), (t["goal"]["cell"], (220, 0, 0))):
            d.rectangle([c * px + 2, r * px + 2, (c + 1) * px - 3, (r + 1) * px - 3], fill=fill)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def grid_source_view(ws, meta, height):
    """2 cm map cropped to the grid, coloured obstacle / roofed floor / unroofed floor."""
    d = paths.slice_dir(ws) / meta["scene"] / "maps"
    k_st, htag = meta["storey"], height_tag(height)
    ob = np.asarray(Image.open(d / f"storey{k_st}_{htag}.png")) == 0
    fl = np.asarray(Image.open(d / f"storey{k_st}_floor.png")) > 0
    rp = d / f"storey{k_st}_roof_{htag}.png"
    rf = np.asarray(Image.open(rp)) > 0 if rp.exists() else np.ones_like(fl)
    img = np.full(ob.shape + (3,), 255, np.uint8)
    img[fl] = (205, 185, 235)
    img[fl & rf] = (130, 205, 130)
    img[ob] = (0, 0, 0)
    # pixel area covered by the cropped grid (see gridify.cell_states)
    k = meta["k"]
    y0 = meta["oy"] - k + meta["r0"] * k
    x0 = meta["ox"] - k + meta["c0"] * k
    H, W = meta["shape"][0] * k, meta["shape"][1] * k
    canvas = np.full((H, W, 3), 255, np.uint8)
    ys, xs = max(y0, 0), max(x0, 0)
    ye, xe = min(y0 + H, img.shape[0]), min(x0 + W, img.shape[1])
    canvas[ys - y0:ye - y0, xs - x0:xe - x0] = img[ys:ye, xs:xe]
    return Image.fromarray(canvas)


def _fit(im, w, h):
    im = im.copy()
    im.thumbnail((w, h), Image.NEAREST)
    return im


def _pages(ids, root_dir, tiles, per_page, cols, tile_h):
    root_dir.mkdir(parents=True, exist_ok=True)
    for old in root_dir.glob("page_*.png"):
        old.unlink()
    rows_max = -(-per_page // cols)
    pages = [ids[i:i + per_page] for i in range(0, len(ids), per_page)]
    for n, page in enumerate(pages, 1):
        sheet = Image.new("RGB", (cols * TILE_W + (cols + 1) * PAD, rows_max * (tile_h + PAD) + PAD),
                          (200, 200, 200))
        for i, m in enumerate(page):
            r, c = divmod(i, cols)
            sheet.paste(tiles(m), (PAD + c * (TILE_W + PAD), PAD + r * (tile_h + PAD)))
        sheet.save(root_dir / f"page_{n:02d}.png")
    return len(pages)


def grid_sheets(ws, ds, robot, per_page=12, cols=3, img_h=300) -> int:
    """preview/<robot>/<map_id>.png for every map + preview/<robot>/sheets/page_NN.png."""
    root = ds.root
    height = ds.robot(robot).height
    stats = ds.manifest["robots"][robot]["maps"]
    pdir = root / "preview" / robot
    for m in ds.maps(robot):
        z = ds.grid_arrays(m)
        grid_map_png(z["state"], z["main"], ds.tasks(robot, m), pdir / f"{m}.png")
    ids = sorted(ds.maps(robot), key=lambda m: (not stats.get(m, {}).get("flags"), m))
    font = _font()

    def tile(m):
        meta, info = ds.grid_meta(m), stats.get(m, {})
        t = Image.new("RGB", (TILE_W, HEAD_H + img_h), (255, 255, 255))
        d = ImageDraw.Draw(t)
        lv = info.get("levels", {})
        d.text((6, 3), f"{m}   {len(ds.tasks(robot, m))} tasks  E{lv.get('easy', 0)} M{lv.get('medium', 0)} "
                       f"H{lv.get('hard', 0)} S{lv.get('starved', 0)}   {meta['main_m2']:.0f} m2",
               fill=(0, 0, 0), font=font)
        if info.get("flags"):
            d.text((6, 21), "FLAG: " + ", ".join(info["flags"]), fill=(200, 0, 0), font=font)
        half = (TILE_W - 3 * PAD) // 2
        views = [Image.open(pdir / f"{m}.png").convert("RGB")]
        if ws is not None:
            views.insert(0, grid_source_view(ws, meta, height))
        for i, im in enumerate(views):
            im = _fit(im, half, img_h - PAD)
            t.paste(im, (PAD + i * (half + PAD) + (half - im.width) // 2, HEAD_H))
        return t
    return _pages(ids, pdir / "sheets", tile, per_page, cols, HEAD_H + img_h)


# ------------------------------------------------------------------ svg

def draw_svg_map(m, tasks, robot, res, success_radius):
    mask, x0, y1, r = m.raster(res)
    img = np.where(mask[..., None], 255, 90).astype(np.uint8).repeat(3, 2)
    for rings in m.viz.values():
        out = rasterize(rings, x0, y1, r, *mask.shape)
        img[out & ~mask] = (205, 185, 235)

    def px(P):
        return np.stack([(P[:, 0] - x0) / r, (y1 - P[:, 1]) / r], 1).round().astype(np.int32)

    for t in tasks:
        g = t["goal"]
        cv2.circle(img, tuple(px(np.array([[g["x"], g["y"]]]))[0]),
                   max(2, int(success_radius / r)), (220, 40, 40), -1)
    for t in tasks:
        s = t["start"]
        P = px(place(robot.footprint, s["x"], s["y"], s["theta"]))
        cv2.polylines(img, [P], True, (0, 150, 0), 1)
        a = px(np.array([[s["x"], s["y"]],
                         [s["x"] + np.cos(s["theta"]) * robot.r_circ * 1.4,
                          s["y"] + np.sin(s["theta"]) * robot.r_circ * 1.4]]))
        cv2.arrowedLine(img, tuple(a[0]), tuple(a[1]), (0, 150, 0), 1, tipLength=0.4)
    return Image.fromarray(img)


def svg_sheets(ds, robot, per_page=12, cols=3, img_h=320, res=0.03) -> int:
    rb = ds.robot(robot)
    stats = ds.manifest["robots"][robot]["maps"]
    ids = sorted(ds.maps(robot), key=lambda m: (not stats.get(m, {}).get("flags"), m))
    font = _font()

    def tile(mid):
        tf = ds.task_file(robot, mid)
        m = SvgMap.load(ds.svg_path(robot, mid))
        im = draw_svg_map(m, tf["tasks"], rb, res, tf["success_radius"])
        im.thumbnail((TILE_W - 2 * PAD, img_h - PAD), Image.NEAREST)
        t = Image.new("RGB", (TILE_W, HEAD_H + img_h), (255, 255, 255))
        d = ImageDraw.Draw(t)
        geo = [x["geodesic_m"] for x in tf["tasks"]]
        d.text((6, 3), f"{mid}   {len(geo)} tasks  geodesic {min(geo):.1f}-{max(geo):.1f} m"
                       f"   IoU {m.meta.get('vectorize', {}).get('iou', float('nan')):.3f}",
               fill=(0, 0, 0), font=font)
        flags = stats.get(mid, {}).get("flags")
        if flags:
            d.text((6, 21), "FLAG: " + ", ".join(flags), fill=(200, 0, 0), font=font)
        t.paste(im, ((TILE_W - im.width) // 2, HEAD_H))
        return t
    return _pages(ids, ds.root / "preview" / robot / "sheets", tile, per_page, cols, HEAD_H + img_h)


# ------------------------------------------------------------------ robots

CELL, COLS = 0.9, 3            # 0.9 m grid cell per robot
STYLE = (".cell{fill:#fafafa;stroke:#ddd;stroke-width:0.003}"
         ".fp{fill:#bfe3bf;stroke:#1f7a1f;stroke-width:0.004}"
         ".old{fill:none;stroke:#d62728;stroke-width:0.003;stroke-dasharray:0.012 0.008}"
         ".circ{fill:none;stroke:#999;stroke-width:0.002;stroke-dasharray:0.01 0.008}"
         ".ctr{stroke:#000;stroke-width:0.004}"
         ".dir{stroke:#1f7a1f;stroke-width:0.006}.ruler{stroke:#000;stroke-width:0.006}"
         "text{font-family:sans-serif;text-anchor:middle}.t1{font-size:0.032px;font-weight:bold}"
         ".t2{font-size:0.024px}.t3{font-size:0.018px;fill:#a0522d}")


def _dims(r):
    fp = r.raw.get("footprint_simple", r.raw["footprint"])
    if fp["type"] == "rectangle":
        return f"{fp['length'] * 1000:.0f} x {fp['width'] * 1000:.0f} mm"
    if fp["type"] == "circle":
        return f"D{fp['radius'] * 2000:.0f} mm"
    return "polygon"


def robots_svg(robots) -> str:
    """Footprints to scale: green = collision shape (URDF section when available), red
    dashed = simple spec-sheet shape, grey dashed = circumscribed circle around the
    rotation centre (C_disk), + = rotation centre, bar = forward (+x). Units m, 1:10."""
    def pts(P, cx, cy):
        return " ".join(f"{cx + x:.4f},{cy - y:.4f}" for x, y in P)
    rows = math.ceil(len(robots) / COLS)
    W, H = COLS * CELL, rows * CELL
    parts = []
    for i, r in enumerate(robots):
        cx = (i % COLS + 0.5) * CELL
        cy = (i // COLS + 0.5) * CELL + 0.05
        old = ""
        if "footprint_simple" in r.raw:
            old = f'<polygon points="{pts(footprint_polygon(r.raw["footprint_simple"]), cx, cy)}" class="old"/>'
        src = "URDF section" if r.raw["footprint"]["type"] == "polygon" else "simple shape (no URDF yet)"
        est = [k for k, v in r.raw["sources"].items() if v["kind"] == "estimated"]
        spd = f"v {r.v_max:g} m/s, w {r.w_max:g} rad/s" + (f", vy {r.vy_max:g}" if r.omni else "")
        top, bot = cy - CELL / 2, cy + CELL / 2
        parts.append(
            f'<g id="{r.id}">'
            f'<rect x="{cx - CELL / 2 + 0.01:.3f}" y="{top - 0.04:.3f}" width="{CELL - 0.02:.3f}" '
            f'height="{CELL - 0.02:.3f}" class="cell"/>'
            f'<circle cx="{cx:.4f}" cy="{cy:.4f}" r="{r.r_circ:.4f}" class="circ"/>'
            f'<polygon points="{pts(r.footprint, cx, cy)}" class="fp"/>{old}'
            f'<line x1="{cx:.4f}" y1="{cy:.4f}" x2="{cx + r.r_circ * 0.8:.4f}" y2="{cy:.4f}" class="dir"/>'
            f'<path d="M {cx - 0.015:.4f} {cy:.4f} H {cx + 0.015:.4f} M {cx:.4f} {cy - 0.015:.4f} '
            f'V {cy + 0.015:.4f}" class="ctr"/>'
            f'<text x="{cx:.4f}" y="{top + 0.03:.4f}" class="t1">{r.name}</text>'
            f'<text x="{cx:.4f}" y="{top + 0.065:.4f}" class="t2">{src}</text>'
            f'<text x="{cx:.4f}" y="{bot - 0.13:.4f}" class="t2">{_dims(r)}, height '
            f'{r.height * 1000:.0f} mm, {r.drive}</text>'
            f'<text x="{cx:.4f}" y="{bot - 0.09:.4f}" class="t2">{spd}</text>'
            f'<text x="{cx:.4f}" y="{bot - 0.05:.4f}" class="t3">'
            f'{"estimated: " + ", ".join(est) if est else "all parameters from official sources"}</text>'
            f'</g>')
    parts.append(f'<g id="scale"><line x1="0.05" y1="{H + 0.05:.3f}" x2="0.55" y2="{H + 0.05:.3f}" '
                 f'class="ruler"/><text x="0.30" y="{H + 0.1:.3f}" class="t2">0.5 m</text></g>')
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.3f} {H + 0.14:.3f}" '
            f'width="{W * 10:.1f}cm" height="{(H + 0.14) * 10:.1f}cm">\n<style>{STYLE}</style>\n'
            + "\n".join(parts) + "\n</svg>\n")


# ------------------------------------------------------------------ episode

def run_episode(ds, robot, map_id, task_idx=None, **env_kwargs):
    """Run the oracle on one task (default: the longest geodesic of the map)."""
    from ..envs.oracle import GridOracle, OracleFollower
    env = ds.make_env(robot=robot, map_id=map_id, **env_kwargs)
    tasks = ds.tasks(robot, map_id)
    i = task_idx if task_idx is not None else max(range(len(tasks)), key=lambda k: tasks[k]["geodesic_m"])
    env.reset(options={"map_id": map_id, "task_idx": i})
    oracle = GridOracle(env) if ds.env_type == "grid" else OracleFollower(env)
    done = False
    while not done:
        _, _, term, trunc, info = env.step(oracle.act())
        done = term or trunc
    return env, i, info


def episode_svg(ds, robot, map_id, task_idx=None, every=15) -> tuple[str, int, dict]:
    if ds.env_type != "svg":
        raise ValueError("episode SVG needs an SVG dataset (use render with --png for grids)")
    env, i, info = run_episode(ds, robot, map_id, task_idx, reward="sparse")
    return env.render_svg(every=every), i, info


def episode_png(ds, robot, map_id, task_idx=None) -> tuple[np.ndarray, int, dict]:
    env, i, info = run_episode(ds, robot, map_id, task_idx, render_mode="rgb_array")
    return env.render(), i, info
