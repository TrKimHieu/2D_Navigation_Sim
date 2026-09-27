"""Manual review of storey detection (``hm3d review``): scroll through the layers,
pick layers to merge, fix the up axis of flipped scenes, then rebuild the maps of each
storey.

Every reviewed scene gets ``<workspace>/stages/slice/<scene>/review.json`` (up axis,
floor/ceiling height of every storey, status). The slice stage applies it instead of
the automatic detection on the next build.
"""

from __future__ import annotations

import base64
import csv
import io
import json
import threading
import time
import traceback
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

from ... import paths
from ...robots import HEIGHT_CLASSES
from .. import slice as S

HERE = Path(__file__).resolve().parent
# set by serve()
OUT: Path = Path(".")
GLB_DIR: Path = Path(".")
CACHE_DIR: Path = Path(".")
ROBOT_H = HEIGHT_CLASSES["h63"]

STATUS = ("todo", "ok", "fixed", "bad")

# meshes are slow to read and large: keep only the last few scenes
LOCK = threading.Lock()
_mesh = {}
_slice_cache = {}
_slice_lock = threading.Lock()


def default_args(res, robot_height, oversample=4.0, floor_mask=False):
    """Slice parameters for interactive rendering."""
    return S.SliceParams(res=res, robot_height=robot_height, oversample=oversample,
                         floor_mask=floor_mask)


def mesh(scene):
    with LOCK:
        if scene not in _mesh:
            if len(_mesh) >= 3:
                _mesh.clear()
            _mesh[scene] = S.load_mesh(GLB_DIR / (scene + ".glb"), CACHE_DIR)
        return _mesh[scene]


# --------------------------------------------------------------- read results

def scene_names():
    return sorted(d.name for d in OUT.iterdir()
                  if d.is_dir() and (d / "profile.csv").exists())


def read_profile(scene):
    rows = []
    p = OUT / scene / "profile.csv"
    if not p.exists():
        return rows
    with open(p, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append({"i": int(r["layer"]), "z": float(r["z_m"]),
                         "ratio": float(r["black_ratio"]),
                         "slab": int(r["is_slab"])})
    return rows


def read_index():
    """index.csv is appended by every run: keep the latest record."""
    idx = {}
    p = OUT / "index.csv"
    if not p.exists():
        return idx
    with open(p, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            idx.setdefault(r["scene"], {})[int(r["storey"])] = {
                "k": int(r["storey"]),
                "floor_z": float(r["floor_z"]), "ceil_z": float(r["ceil_z"]),
                "h": float(r["storey_h"]), "free_m2": float(r["free_m2"]),
                "png": r["png"],
            }
    return {s: [v[k] for k in sorted(v)] for s, v in idx.items()}


def read_review(scene):
    p = OUT / scene / "review.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def auto_up(scene):
    """Up axis used by the automatic slice; None if unknown.

    Scenes sliced before auto.json existed fall back to storeyK.json.
    """
    for p in [OUT / scene / "auto.json",
              *sorted((OUT / scene / "maps").glob("storey*.json"))]:
        try:
            return int(json.loads(p.read_text(encoding="utf-8"))["up"])
        except Exception:
            continue
    return None


def slice_files(scene):
    """Layer image paths in layer order.

    The background thumbnail thread calls this concurrently with browser requests,
    so the cache is per scene and locked.
    """
    d = OUT / scene / "slices"
    stamp = d.stat().st_mtime if d.exists() else 0
    with _slice_lock:
        hit = _slice_cache.get(scene)
    if hit and hit[0] == stamp:
        return hit[1]
    files = sorted(d.glob("L*.png")) if d.exists() else []
    with _slice_lock:
        _slice_cache[scene] = (stamp, files)
    return files


def audit(prof, storeys):
    """Suspicion signals, so the most suspicious scenes are reviewed first."""
    flags = []
    if prof:
        span = prof[-1]["z"] - prof[0]["z"]
        top = max(r["ratio"] for r in prof)
        if top < 0.35:
            flags.append([f"density peak only {top:.2f} - up axis suspect", 3])
        if span > 12.0:
            flags.append([f"height {span:.1f} m - up axis suspect", 2])
    if not storeys:
        flags.append(["no storey detected", 3])
    for st in storeys:
        if st["h"] > 4.5:
            flags.append([f"storey {st['k']} is {st['h']:.1f} m high", 2])
        if st["free_m2"] < 5:
            flags.append([f"storey {st['k']} only {st['free_m2']:.0f} m2", 2])
    return flags


def scene_rows():
    idx = read_index()
    rows = []
    for name in scene_names():
        prof = read_profile(name)
        st = idx.get(name, [])
        rv = read_review(name) or {}
        flags = audit(prof, st)
        rows.append({
            "name": name,
            "layers": len(prof),
            "span": round(prof[-1]["z"] - prof[0]["z"], 2) if prof else 0,
            "auto_storeys": len(st),
            "status": rv.get("status", "todo"),
            "n_fixed": len(rv.get("storeys", [])),
            "score": sum(f[1] for f in flags),
            "flags": [f[0] for f in flags],
        })
    return rows


# --------------------------------------------------------------- compute

def render_map(scene, up, floor_z, ceil_z, res, robot_h, oversample,
               floor_mask=False):
    """Real map of one storey: black = material inside the robot band."""
    args = default_args(res, robot_h, oversample, floor_mask)
    V, F = mesh(scene)
    frame = S.Frame(V, up, res)
    with LOCK:
        m, stats = S.build_map(V, F, frame, floor_z, ceil_z, args,
                               args.oversample / (res ** 2))
    return m, stats


def axis_probe(scene, res=0.04, n_pts=1_200_000):
    """For every axis, project the densest layer perpendicular to it.

    The true up axis gives a FLOOR PLAN (solid, covering the building); the other two
    give vertical sections (sparse, a few lines). Obvious at a glance, no need to
    interpret numbers.
    """
    V, F = mesh(scene)
    area, _ = S.face_areas_normals(V, F)
    rng = np.random.default_rng(0)
    with LOCK:
        P = np.concatenate(list(S.sample_faces(V[F], area, n_pts, rng)))
        auto = int(S.detect_up_axis(V, F))
    lo, hi = P.min(axis=0), P.max(axis=0)
    cands = []
    for a in range(3):
        c = P[:, a]
        nb = max(8, int((hi[a] - lo[a]) / 0.10))
        h, edges = np.histogram(c, bins=nb)
        k = int(h.argmax())
        z0, z1 = edges[k] - 0.06, edges[k + 1] + 0.06
        Q = P[(c >= z0) & (c < z1)]
        ua, va = [x for x in range(3) if x != a]
        W = int((hi[ua] - lo[ua]) / res) + 1
        H = int((hi[va] - lo[va]) / res) + 1
        img = np.full((H, W), 255, np.uint8)
        if len(Q):
            ci = np.clip(((Q[:, ua] - lo[ua]) / res).astype(np.int32), 0, W - 1)
            ri = np.clip(((Q[:, va] - lo[va]) / res).astype(np.int32), 0, H - 1)
            img[H - 1 - ri, ci] = 0
        cands.append({"axis": a, "name": "XYZ"[a],
                      "fill": round(float((img == 0).sum()) / (W * H), 4),
                      "z": round(float((z0 + z1) / 2), 2),
                      "span": round(float(hi[a] - lo[a]), 2),
                      "png": png_b64(img)})
    return {"auto": auto, "cands": cands}


def suggest_storeys(scene, up, min_storey_m=1.8):
    """Suggest storeys from RELATIVE peaks of the density profile, not a fixed threshold.

    The slice stage's min_ratio divides by the union footprint of all layers, so a
    low sprawling house or one with a garden inflates the denominator, pushes every
    ratio below the threshold and loses all storeys (e.g. 00803: 9 storeys, 31 m,
    one slab found). Prominence against the neighbourhood does not depend on it.
    """
    from scipy.ndimage import median_filter
    from scipy.signal import find_peaks

    prof = read_profile(scene)
    if len(prof) < 3:
        return {"storeys": [], "peaks": []}
    z = np.array([r["z"] for r in prof])
    ratio = np.array([r["ratio"] for r in prof])
    step = float(z[1] - z[0])

    # local baseline = median over a 2 m window; slabs rise several times above it and
    # the ratio does not depend on the footprint denominator (works on sprawling houses)
    base = median_filter(ratio, size=max(3, int(2.0 / step) | 1))
    # pad zeros at both ends: find_peaks never reports the first/last layer, and the
    # ground floor often falls exactly on layer 0
    pk, _ = find_peaks(np.concatenate([[0.0], ratio, [0.0]]), prominence=0.015,
                       distance=max(1, int(round(1.2 / step))))
    pk = [i - 1 for i in pk if ratio[i - 1] >= max(3.0 * base[i - 1], 0.05)]

    slabs = []                       # a 20-30 cm slab gives two adjacent peaks
    for i in pk:
        if slabs and z[i] - z[slabs[-1][-1]] <= 0.6:
            slabs[-1].append(int(i))
        else:
            slabs.append([int(i)])

    V, F = mesh(scene)
    out = []
    with LOCK:
        for g, h in zip(slabs, slabs[1:]):
            a, b = g[-1], h[0]
            fz = S.refine_surface_z(V, F, up, z[a], z[a] + step)
            cz = S.refine_surface_z(V, F, up, z[b], z[b] + step)
            fz = float(z[a]) if fz is None else float(fz)
            cz = float(z[b]) if cz is None else float(cz)
            if cz - fz >= min_storey_m:
                out.append({"floor_z": round(fz, 3), "ceil_z": round(cz, 3)})
    return {"storeys": out, "peaks": [int(i) for i in pk]}


def reslice(scene, up, res, step, oversample, say):
    """Re-cut every layer along another up axis; overwrites slices/ + profile.csv."""
    args = default_args(res, ROBOT_H, oversample)
    scene_dir = OUT / scene
    V, F = mesh(scene)
    frame = S.Frame(V, up, res)
    say("sampling the surface...")
    with LOCK:
        grids, heights = S.build_layers(V, F, frame, step,
                                        oversample / (res ** 2), 0)
    L = len(grids)
    footprint = grids.any(axis=0)
    ratio = grids.reshape(L, -1).sum(axis=1) / max(int(footprint.sum()), 1)
    slabs = S.find_slabs(ratio, heights, args.min_ratio)
    is_slab = {i for g in slabs for i in g}

    S.save_bw(footprint, scene_dir / "footprint.png")
    with open(scene_dir / "profile.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["layer", "z_m", "black_px", "black_ratio", "is_slab"])
        for i in range(L):
            w.writerow([i, f"{heights[i]:.3f}", int(grids[i].sum()),
                        f"{ratio[i]:.4f}", int(i in is_slab)])

    say(f"writing {L} layer images...")
    sdir = scene_dir / "slices"
    sdir.mkdir(parents=True, exist_ok=True)
    for old in sdir.glob("L*.png"):
        old.unlink()
    for old in (scene_dir / ".thumbs").glob("*.png"):
        old.unlink()
    for i in range(L):
        S.save_bw(grids[i], sdir / f"L{i:03d}_z{heights[i]:+.2f}.png")
    _slice_cache.clear()
    _warmed.discard(scene)
    return {"layers": L, "slabs": len(slabs), "up": up}


def render_all(scene, up, storeys, robot_h, res, oversample, say,
               floor_mask=False):
    """Build the real maps of every reviewed storey; overwrites <scene>/maps/."""
    maps_dir = OUT / scene / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    tag = f"h{int(round(robot_h * 100)):02d}"
    for old in maps_dir.glob(f"storey*_{tag}.png"):
        old.unlink()
    out = []
    for k, st in enumerate(storeys, 1):
        say(f"storey {k}/{len(storeys)}...")
        m, stats = render_map(scene, up, st["floor_z"], st["ceil_z"],
                              res, robot_h, oversample, floor_mask)
        name = f"storey{k}_{tag}.png"
        S.save_map(m, maps_dir / name)
        # floor mask + frame for the grid/vector stages, as in the slice stage
        V, F = mesh(scene)
        with LOCK:
            S.save_storey_extras(V, F, S.Frame(V, up, res), k, st["floor_z"],
                                 st["ceil_z"], default_args(res, robot_h, oversample),
                                 maps_dir, scene)
        out.append({"k": k, "png": name,
                    "floor_z": st["floor_z"], "ceil_z": st["ceil_z"],
                    "free_m2": round(stats["free_m2"], 1),
                    "blocked_m2": round(stats["blocked_m2"], 1)})
    return {"maps": out, "res": res, "robot_h": robot_h}


# --------------------------------------------------------------- images

def png_bytes(arr):
    buf = io.BytesIO()
    Image.fromarray(arr, "L").save(buf, "PNG")
    return buf.getvalue()


def png_b64(arr):
    return "data:image/png;base64," + base64.b64encode(png_bytes(arr)).decode()


def thumb(scene, i, w=104):
    """Thumbnail of one layer, cached in <scene>/.thumbs/."""
    tdir = OUT / scene / ".thumbs"
    tp = tdir / f"L{i:03d}.png"
    files = slice_files(scene)
    if i >= len(files):
        return None
    # a new slice run may overwrite slices/ (e.g. another up axis): the old thumbnail
    # of that layer is then stale
    if tp.exists() and tp.stat().st_mtime >= files[i].stat().st_mtime:
        return tp.read_bytes()
    im = Image.open(files[i]).convert("L")
    h = max(1, int(im.height * w / im.width))
    # BOX averages the whole cell: keeps material density, thin walls do not vanish
    im = im.resize((w, h), Image.BOX)
    tdir.mkdir(exist_ok=True)
    im.save(tp)
    return tp.read_bytes()


_warmed = set()


def warm_thumbs(scene, n):
    """Pre-generate thumbnails when a scene opens so the strip does not trickle in."""
    if scene in _warmed:
        return
    _warmed.add(scene)

    def run():
        for i in range(n):
            try:
                thumb(scene, i)
            except Exception:
                return

    threading.Thread(target=run, daemon=True).start()


# --------------------------------------------------------------- background jobs

JOBS = {}


def start_job(fn):
    jid = uuid.uuid4().hex[:8]
    JOBS[jid] = {"state": "run", "msg": "running...", "t0": time.time()}

    def say(m):
        JOBS[jid]["msg"] = m

    def run():
        try:
            JOBS[jid]["result"] = fn(say)
            JOBS[jid]["state"] = "done"
            JOBS[jid]["msg"] = f"done ({time.time() - JOBS[jid]['t0']:.1f}s)"
        except Exception as e:
            traceback.print_exc()
            JOBS[jid]["state"] = "error"
            JOBS[jid]["msg"] = f"{type(e).__name__}: {e}"

    threading.Thread(target=run, daemon=True).start()
    return jid


# --------------------------------------------------------------- http

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def send(self, code, ctype, body, headers=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, obj, code=200):
        self.send(code, "application/json; charset=utf-8", json.dumps(obj))

    def send_file(self, path, ctype, cache=True):
        path = Path(path)
        if not path.exists():
            return self.send(404, "text/plain", "khong co " + path.name)
        h = {"Cache-Control": "max-age=86400"} if cache else {
            "Cache-Control": "no-store"}
        self.send(200, ctype, path.read_bytes(), h)

    # ------------------------------------------------------------- GET
    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        p = u.path
        try:
            if p in ("/", "/index.html"):
                return self.send_file(HERE / "review.html",
                                      "text/html; charset=utf-8", cache=False)
            if p == "/api/scenes":
                return self.send_json({"scenes": scene_rows()})
            if p == "/api/scene":
                return self.send_json(self.scene_payload(q["name"]))
            if p == "/api/job":
                return self.send_json(JOBS.get(q["id"], {"state": "error",
                                                         "msg": "no such job"}))
            if p == "/api/axis":
                name = q["name"]
                return self.send_json({"id": start_job(
                    lambda say: axis_probe(name))})
            if p.startswith("/slice/"):
                _, _, scene, i = p.split("/", 3)
                files = slice_files(scene)
                i = int(i)
                if i >= len(files):
                    return self.send(404, "text/plain", "het lop")
                return self.send_file(files[i], "image/png")
            if p.startswith("/thumb/"):
                _, _, scene, i = p.split("/", 3)
                b = thumb(scene, int(i))
                if b is None:
                    return self.send(404, "text/plain", "het lop")
                return self.send(200, "image/png", b,
                                 {"Cache-Control": "max-age=86400"})
            if p.startswith("/footprint/"):
                scene = p.split("/", 2)[2]
                return self.send_file(OUT / scene / "footprint.png", "image/png")
            if p.startswith("/map/"):
                _, _, scene, name = p.split("/", 3)
                return self.send_file(OUT / scene / "maps" / name, "image/png",
                                      cache=False)
            if p == "/api/suggest":
                return self.send_json(suggest_storeys(q["name"], int(q["up"])))
            if p == "/api/refine":
                return self.send_json(self.refine(q))
            if p == "/api/preview.png":
                return self.preview(q)
            return self.send(404, "text/plain", "404 " + p)
        except Exception as e:
            traceback.print_exc()
            self.send_json({"error": f"{type(e).__name__}: {e}"}, 500)

    def scene_payload(self, name):
        prof = read_profile(name)
        warm_thumbs(name, len(prof))
        idx = read_index().get(name, [])
        rv = read_review(name)
        first = slice_files(name)
        W = H = 0
        if first:
            with Image.open(first[0]) as im:
                W, H = im.size
        return {
            "name": name, "W": W, "H": H,
            "layers": prof,
            "step": round(prof[1]["z"] - prof[0]["z"], 3) if len(prof) > 1 else 0.1,
            "auto": idx,
            "auto_up": auto_up(name),
            "review": rv,
            "flags": [f[0] for f in audit(prof, idx)],
        }

    def refine(self, q):
        """Layer edge -> real floor/ceiling height, as the slice stage does before cutting the band."""
        scene, up = q["name"], int(q["up"])
        z, step = float(q["z"]), float(q.get("step", 0.10))
        V, F = mesh(scene)
        with LOCK:
            r = S.refine_surface_z(V, F, up, z, z + step)
        return {"z": z if r is None else float(r)}

    def preview(self, q):
        """Real map of one storey at preview resolution."""
        scene = q["name"]
        up = int(q["up"])
        res = float(q.get("res", 0.05))
        ov = float(q.get("oversample", 2.0))
        arr, stats = render_map(scene, up, float(q["floor_z"]),
                                float(q["ceil_z"]), res,
                                float(q.get("robot_h", ROBOT_H)), ov,
                                q.get("floor_mask") == "1")
        arr = np.where(arr == S.FREE, 255, 0).astype(np.uint8)
        stats = {k: round(v, 2) for k, v in stats.items()}
        self.send(200, "image/png", png_bytes(arr),
                  {"Cache-Control": "no-store",
                   "X-Stats": json.dumps(stats)})

    # ------------------------------------------------------------- POST
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or "{}")
        p = urlparse(self.path).path
        try:
            if p == "/api/review":
                return self.send_json(self.save_review(body))
            if p == "/api/reslice":
                b = body
                return self.send_json({"id": start_job(
                    lambda say: reslice(b["name"], int(b["up"]),
                                        float(b.get("res", 0.02)),
                                        float(b.get("step", 0.10)),
                                        float(b.get("oversample", 4.0)), say))})
            if p == "/api/render":
                b = body
                return self.send_json({"id": start_job(
                    lambda say: render_all(b["name"], int(b["up"]), b["storeys"],
                                           float(b.get("robot_h", ROBOT_H)),
                                           float(b.get("res", 0.02)),
                                           float(b.get("oversample", 4.0)), say,
                                           bool(b.get("floor_mask"))))})
            return self.send(404, "text/plain", "404 " + p)
        except Exception as e:
            traceback.print_exc()
            self.send_json({"error": f"{type(e).__name__}: {e}"}, 500)

    def save_review(self, b):
        name = b["name"]
        rv = {
            "scene": name,
            "up": int(b["up"]),
            "storeys": [{"floor_z": round(float(s["floor_z"]), 3),
                         "ceil_z": round(float(s["ceil_z"]), 3)}
                        for s in b.get("storeys", [])],
            "status": b.get("status", "fixed"),
            "note": b.get("note", ""),
            "robot_h": float(b.get("robot_h", ROBOT_H)),
            "ts": time.strftime("%Y-%m-%d %H:%M"),
        }
        if rv["status"] not in STATUS:
            rv["status"] = "fixed"
        (OUT / name / "review.json").write_text(
            json.dumps(rv, ensure_ascii=False, indent=1), encoding="utf-8")
        return rv


def serve(ws, port=8765, open_browser=True, robot_height=None):
    """Run the review server on 127.0.0.1:<port> until Ctrl+C."""
    global OUT, GLB_DIR, CACHE_DIR, ROBOT_H
    ws = Path(ws)
    OUT, GLB_DIR, CACHE_DIR = paths.slice_dir(ws), paths.glb_dir(ws), paths.cache_dir(ws)
    if robot_height:
        ROBOT_H = robot_height
    if not OUT.exists():
        raise FileNotFoundError(f"{OUT} does not exist; run the slice stage first (hm3d slice)")
    url = f"http://127.0.0.1:{port}/"
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Review: {url}   (Ctrl+C to stop)")
    print(f"{len(scene_names())} scenes in {OUT}")
    if open_browser:
        threading.Timer(0.7, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
