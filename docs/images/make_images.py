"""Regenerate docs/images/episode.gif: an oracle TurtleBot 4 driving in a bundled demo map,
with its LiDAR rays, its path and the goal. Needs Pillow (`./setup.sh --build`).

    python docs/images/make_images.py
"""

from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
from PIL import Image

import hm3denv  # noqa: F401
from hm3denv.core.geometry import place
from hm3denv.envs import OracleFollower

OUT = Path(__file__).resolve().parent / "episode.gif"
MAP, TASK, ROBOT = "demo-S001_s0", 7, "turtlebot4"
RES, EVERY, SIZE = 0.025, 5, 520          # m per pixel, steps per frame, output width (px)
FREE, WALL = (250, 250, 247), (52, 58, 70)
PATH, RAY, HIT, BODY, GOAL = (60, 120, 216), (255, 170, 60), (235, 110, 20), (40, 150, 70), (214, 40, 40)


def frame(u, obs, bg, x0, y1, path):
    def px(P):
        P = np.asarray(P, float).reshape(-1, 2)
        return np.stack([(P[:, 0] - x0) / RES, (y1 - P[:, 1]) / RES], 1).round().astype(np.int32)

    img = bg.copy()
    x, y, th = u.pose
    a = th + u.beam_angles
    ends = np.stack([x + obs["lidar"] * np.cos(a), y + obs["lidar"] * np.sin(a)], 1)
    rays = img.copy()
    c = tuple(px([x, y])[0])
    for e in px(ends):
        cv2.line(rays, c, tuple(e), RAY, 1, cv2.LINE_AA)
    img = cv2.addWeighted(rays, 0.6, img, 0.4, 0)
    for e in px(ends):
        cv2.circle(img, tuple(e), 2, HIT, -1, cv2.LINE_AA)
    if len(path) > 1:
        cv2.polylines(img, [px(path)], False, PATH, 3, cv2.LINE_AA)
    g = u.task["goal"]
    cv2.circle(img, tuple(px([g["x"], g["y"]])[0]), max(5, int(u.success_radius / RES)), GOAL, -1, cv2.LINE_AA)
    cv2.fillPoly(img, [px(place(u.robot.footprint, x, y, th))], BODY, cv2.LINE_AA)
    nose = px([x + 0.25 * np.cos(th), y + 0.25 * np.sin(th)])[0]
    cv2.line(img, c, tuple(nose), (255, 255, 255), 2, cv2.LINE_AA)
    return img


def main():
    env = gym.make("HM3D/Svg-v0", dataset="demo-svg", robot=ROBOT)
    u = env.unwrapped
    obs, _ = env.reset(seed=0, options={"map_id": MAP, "task_idx": TASK})
    mask, x0, y1, _ = u.map.raster(RES)
    bg = np.where(mask[..., None], np.array(FREE, np.uint8), np.array(WALL, np.uint8))
    oracle = OracleFollower(env)
    frames, path, done, t = [], [tuple(u.pose[:2])], False, 0
    while not done:
        if t % EVERY == 0:
            frames.append(frame(u, obs, bg, x0, y1, path))
        obs, _, term, trunc, info = env.step(oracle.act())
        path.append(tuple(u.pose[:2]))
        done, t = term or trunc, t + 1
    frames += [frame(u, obs, bg, x0, y1, path)] * 12              # hold the last frame
    h, w = frames[0].shape[:2]
    size = (SIZE, round(SIZE * h / w))
    ims = [Image.fromarray(cv2.resize(f, size, interpolation=cv2.INTER_AREA)) for f in frames]
    pal = ims[-1].quantize(colors=32, method=Image.Quantize.MEDIANCUT)
    ims = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in ims]
    ims[0].save(OUT, save_all=True, append_images=ims[1:], duration=60, loop=0, optimize=True)
    print(f"{info['termination']} in {t} steps, {len(ims)} frames -> {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
