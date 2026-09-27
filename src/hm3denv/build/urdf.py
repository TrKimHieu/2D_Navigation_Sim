"""Real horizontal cross-section of a robot from the manufacturer's URDF/xacro + meshes.

  1. download the description packages (xacro/urdf/yaml) from the manufacturer's GitHub
     and expand xacro without ROS ($(find pkg) points into the download cache)
  2. assemble the robot along its joints: joints at 0 (or the arm tuck pose `joints`)
  3. project EVERY visual geometry onto the floor in the root frame (base_footprint:
     rotation centre, x forward) at 1 mm, take the outer contour, simplify by 1 mm
  4. compare the bounding box with the preset's official dimensions: beyond `max_dev`
     it is an error (wrong mesh units / axes) instead of silently writing bad data
  5. write the preset: footprint = polygon (possibly concave), source = model URL; the
     old footprint is kept as footprint_simple for comparison

Projecting the whole body is CONSERVATIVE: a part at any height may collide.
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

import cv2
import numpy as np

RES = 0.001

# description packages: name -> (repo, branch, package directory inside the repo)
PACKAGES = {
    "turtlebot3_description": ("ROBOTIS-GIT/turtlebot3", "humble", "turtlebot3_description"),
    "turtlebot4_description": ("turtlebot/turtlebot4", "humble", "turtlebot4_description"),
    "irobot_create_description": ("iRobotEducation/create3_sim", "humble",
                                  "irobot_create_common/irobot_create_description"),
    "irobot_create_control": ("iRobotEducation/create3_sim", "humble",
                              "irobot_create_common/irobot_create_control"),
    "kobuki_description": ("kobuki-base/kobuki_ros", "devel", "kobuki_description"),
    "jackal_description": ("jackal/jackal", "noetic-devel", "jackal_description"),
    "limo_description": ("agilexrobotics/limo_ros", "master", "limo_description"),
    "pmb2_description": ("pal-robotics/pmb2_robot", "humble-devel", "pmb2_description"),
    "tiago_description": ("pal-robotics/tiago_robot", "humble-devel", "tiago_description"),
    "pal_urdf_utils": ("pal-robotics/pal_urdf_utils", "humble-devel", ""),
    "pal_gripper_description": ("pal-robotics/pal_gripper", "humble-devel",
                                "pal_gripper_description"),
}

# robot -> model; args = xacro arguments, joints = non-zero joint values
MODELS = {
    "turtlebot3_burger": {"entry": "turtlebot3_description/urdf/turtlebot3_burger.urdf"},
    "turtlebot3_waffle_pi": {"entry": "turtlebot3_description/urdf/turtlebot3_waffle_pi.urdf"},
    "turtlebot4": {"entry": "turtlebot4_description/urdf/standard/turtlebot4.urdf.xacro"},
    "turtlebot4_lite": {"entry": "turtlebot4_description/urdf/lite/turtlebot4.urdf.xacro"},
    "kobuki": {"entry": "kobuki_description/urdf/kobuki_standalone.urdf.xacro"},
    "clearpath_jackal": {"entry": "jackal_description/urdf/jackal.urdf.xacro"},
    "agilex_limo": {"entry": "limo_description/urdf/limo_four_diff.xacro",
                    "args": {"robot_namespace": "/"}},
    # arm in PAL's "home" pose (last waypoint of tiago_bringup/config/motions/
    # tiago_motions_general.yaml); optional sensors off (small, inside the body)
    "pal_tiago": {"entry": "tiago_description/robots/tiago.urdf.xacro",
                  "args": {"end_effector": "pal-gripper", "ft_sensor": "no-ft-sensor",
                           "laser_model": "no-laser", "camera_model": "no-camera"},
                  "joints": {"torso_lift_joint": 0.15, "arm_1_joint": 0.50,
                             "arm_2_joint": -1.34, "arm_3_joint": -0.48, "arm_4_joint": 1.94,
                             "arm_5_joint": -1.49, "arm_6_joint": 1.37, "arm_7_joint": 0.0}},
}


# ------------------------------------------------------------------ download

_trees = {}


def repo_tree(cache, repo, ref):
    if (repo, ref) not in _trees:
        f = cache / "_trees" / f"{repo.replace('/', '__')}__{ref}.json"
        if not f.exists():
            f.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://api.github.com/repos/{repo}/git/trees/{ref}?recursive=1"
            with urllib.request.urlopen(url) as r:
                f.write_bytes(r.read())
        _trees[(repo, ref)] = [t["path"] for t in json.loads(f.read_text())["tree"]
                               if t["type"] == "blob"]
    return _trees[(repo, ref)]


def fetch(cache, pkg, rel):
    """Download one package file into <cache>/<pkg>/<rel> (once)."""
    repo, ref, sub = PACKAGES[pkg]
    dst = cache / pkg / rel
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://raw.githubusercontent.com/{repo}/{ref}/{sub + '/' if sub else ''}{rel}"
        with urllib.request.urlopen(url) as r:
            dst.write_bytes(r.read())
    return dst


def fetch_text_files(cache, pkg):
    """Every xacro/urdf/yaml file of a package (small); xacro needs them before running."""
    repo, ref, sub = PACKAGES[pkg]
    for p in repo_tree(cache, repo, ref):
        rel = p[len(sub) + 1:] if sub else p
        if (not sub or p.startswith(sub + "/")) and p.endswith(
                (".xacro", ".urdf", ".yaml", ".yml", ".gazebo", ".xml")):
            fetch(cache, pkg, rel)
    return cache / pkg


def expand(cache, model):
    """Expanded URDF (XML string). No ROS needed: $(find pkg) points into the cache."""
    pkg, rel = model["entry"].split("/", 1)
    fetch_text_files(cache, pkg)
    path = cache / pkg / rel
    if path.suffix == ".urdf":
        # the TurtleBot3 URDF leaves ${namespace} for its launch file
        return path.read_text(encoding="utf-8").replace("${namespace}", "")
    import xacro
    from xacro import substitution_args as sa

    def find(name):
        if name in PACKAGES:
            return str(fetch_text_files(cache, name))
        # packages outside PACKAGES (e.g. Jackal's optional sensors) are EMPTY stubs:
        # if the model really calls a macro from one, xacro fails, so geometry can
        # never be dropped silently
        return str(cache / "_stub" / name)

    for f in list(cache.glob("*/**/*.xacro")) + list(cache.glob("*/**/*.urdf")):
        for name, rel in re.findall(r"\$\(find ([\w-]+)\)/([^\s\"'<>]+)", f.read_text("utf-8", "ignore")):
            stub = cache / "_stub" / name / rel
            if name not in PACKAGES and not stub.exists():
                stub.parent.mkdir(parents=True, exist_ok=True)
                stub.write_text('<robot xmlns:xacro="http://www.ros.org/wiki/xacro"/>\n')
    sa._eval_find = find
    doc = xacro.process_file(str(path), mappings=model.get("args", {}))
    return doc.toxml()


# ------------------------------------------------------------------ assemble

def origin(el):
    import trimesh.transformations as tf
    if el is None:
        return np.eye(4)
    xyz = [float(v) for v in el.get("xyz", "0 0 0").split()]
    rpy = [float(v) for v in el.get("rpy", "0 0 0").split()]
    return tf.compose_matrix(angles=rpy, translate=xyz)


def load_geom(cache, geom):
    import trimesh
    m = geom.find("mesh")
    if m is not None:
        fn = m.get("filename").replace("\\", "/")
        base = cache.resolve().as_posix().rstrip("/") + "/"
        if fn.startswith("file://") and base.lower() in fn.lower():   # $(find pkg) in the cache
            fn = "package://" + fn[fn.lower().index(base.lower()) + len(base):]
        mm = re.match(r"package://([^/]+)/(.+)", fn)
        if not mm:
            raise ValueError(f"cannot resolve mesh {fn}")
        f = fetch(cache, mm.group(1), mm.group(2))
        mesh = trimesh.load(f, force="mesh")
        s = [float(v) for v in m.get("scale", "1 1 1").split()]
        mesh.apply_scale(s)
        return mesh
    if geom.find("box") is not None:
        return trimesh.creation.box([float(v) for v in geom.find("box").get("size").split()])
    if geom.find("cylinder") is not None:
        c = geom.find("cylinder")
        return trimesh.creation.cylinder(float(c.get("radius")), float(c.get("length")))
    if geom.find("sphere") is not None:
        return trimesh.creation.icosphere(radius=float(geom.find("sphere").get("radius")))
    return None


def assemble(cache, urdf_xml, joints=None):
    """(triangles (T,3,3) in the root frame, root frame name)."""
    import xml.etree.ElementTree as ET
    import trimesh.transformations as tf
    root = ET.fromstring(urdf_xml)
    joints = joints or {}
    par = {}
    for j in root.findall("joint"):          # top level only: skips <ros2_control> joints
        T = origin(j.find("origin"))
        q = joints.get(j.get("name"), 0.0)
        if q and j.get("type") in ("revolute", "continuous"):
            ax = [float(v) for v in (j.find("axis").get("xyz") if j.find("axis") is not None
                                     else "1 0 0").split()]
            T = T @ tf.rotation_matrix(q, ax)
        elif q and j.get("type") == "prismatic":
            ax = np.array([float(v) for v in j.find("axis").get("xyz").split()])
            T = T @ tf.translation_matrix(q * ax)
        par[j.find("child").get("link")] = (j.find("parent").get("link"), T)
    links = [l.get("name") for l in root.findall("link")]
    base = next(l for l in links if l not in par)

    def world(link):
        T = np.eye(4)
        while link in par:
            p, J = par[link]
            T = J @ T
            link = p
        return T

    tris = []
    for L in root.findall("link"):
        for v in L.findall("visual"):
            m = load_geom(cache, v.find("geometry"))
            if m is None:
                continue
            m.apply_transform(world(L.get("name")) @ origin(v.find("origin")))
            tris.append(m.triangles)
    return np.concatenate(tris), base


def silhouette(tris, res=RES, close_px=2, eps_px=1.0):
    """Outer contour (CCW, m) of the projection of every triangle onto the XY plane."""
    xy = tris[:, :, :2]
    lo = xy.reshape(-1, 2).min(0) - 0.01
    W, H = np.ceil((xy.reshape(-1, 2).max(0) + 0.01 - lo) / res).astype(int)
    img = np.zeros((H, W), np.uint8)
    for t in np.round((xy - lo) / res).astype(np.int32):
        cv2.fillConvexPoly(img, t, 1)
    if close_px:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_px + 1,) * 2)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k)
    cs, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = cv2.approxPolyDP(max(cs, key=cv2.contourArea), eps_px, True)[:, 0, :].astype(float)
    P = c * res + lo
    area2 = np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
    return P if area2 > 0 else P[::-1]


def official_box(raw):
    fp = raw.get("footprint_simple", raw["footprint"])
    if fp["type"] == "rectangle":
        return fp["length"], fp["width"]
    if fp["type"] == "circle":
        return 2 * fp["radius"], 2 * fp["radius"]
    return None


def import_urdf(raw: dict, cache: Path, max_dev: float = 0.25) -> tuple[dict, str]:
    """(updated preset dict, report line). Raises ValueError when the model's bounding
    box deviates more than `max_dev` from the official dimensions."""
    rid = raw["id"]
    if rid not in MODELS:
        raise KeyError(f"no URDF model registered for {rid}; known: {sorted(MODELS)}")
    model = MODELS[rid]
    cache = Path(cache)
    tris, base = assemble(cache, expand(cache, model), model.get("joints"))
    P = silhouette(tris)
    L, Wd = P[:, 0].max() - P[:, 0].min(), P[:, 1].max() - P[:, 1].min()
    box = official_box(raw)
    dev = max(abs(L - box[0]) / box[0], abs(Wd - box[1]) / box[1]) if box else 0.0
    area = 0.5 * abs(np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1]))
    cx = (P[:, 0].max() + P[:, 0].min()) / 2
    msg = (f"{rid}: root '{base}', {len(P)} vertices, box {L * 1000:.0f} x {Wd * 1000:.0f} mm"
           + (f" (official {box[0] * 1000:.0f} x {box[1] * 1000:.0f})" if box else "")
           + f", model height {tris[:, :, 2].max() * 1000:.0f} mm, area {area * 1e4:.0f} cm2, "
           f"rotation centre {cx * 1000:+.0f} mm from box centre in x, box deviation {dev:.0%}")
    if dev > max_dev:
        raise ValueError(msg + f" -> above max_dev {max_dev:.0%}, not written")
    repo, ref, sub = PACKAGES[model["entry"].split("/", 1)[0]]
    url = f"https://github.com/{repo}/blob/{ref}/{sub}/{model['entry'].split('/', 1)[1]}"
    out = json.loads(json.dumps(raw))
    if "footprint_simple" not in out:
        out["footprint_simple"] = out["footprint"]
        out["sources"]["footprint_simple"] = out["sources"]["footprint"]
    out["footprint"] = {"type": "polygon", "points": [[round(x, 4), round(y, 4)] for x, y in P]}
    out["sources"]["footprint"] = {
        "kind": "official", "url": url,
        "method": "all URDF visual geometry (joints = 0"
                  + (", arm tucked" if model.get("joints") else "")
                  + f") projected onto the floor of frame '{base}' at 1 mm, outer contour, "
                    "simplified by 1 mm (hm3d robots import-urdf)"}
    return out, msg
