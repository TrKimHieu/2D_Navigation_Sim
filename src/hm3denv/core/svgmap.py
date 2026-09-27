"""Read/write SVG maps.

The file stores REAL map-frame coordinates in metres: content sits in
``<g transform="scale(1,-1)">`` so y points up (REP-103); drawing scale 1:100
(1 m = 1 cm when printed). The ``free`` path (fill-rule evenodd) is the region a
robot may occupy and the only layer environments read; ``viz`` paths are for
display. Metadata is JSON inside ``<metadata id="hm3denv">``.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET

import cv2
import numpy as np

SCHEMA_VERSION = "2.0"
READABLE_SCHEMAS = ("1.0", "2.0")
SVG_NS = "http://www.w3.org/2000/svg"
_NUM = re.compile(r"[MLZ]|-?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?")


def rings_to_d(rings, nd: int = 4) -> str:
    parts = []
    for R in rings:
        pts = " L ".join(f"{x:.{nd}f},{y:.{nd}f}" for x, y in R)
        parts.append(f"M {pts} Z")
    return " ".join(parts)


def d_to_rings(d: str) -> list[np.ndarray]:
    """Parse a path made only of M/L/Z commands (the format rings_to_d writes)."""
    rings, cur, nums = [], [], []
    for tok in _NUM.findall(d.replace(",", " ")):
        if tok in "MLZ":
            if tok in "MZ" and cur:
                rings.append(np.array(cur, float))
                cur = []
            continue
        nums.append(float(tok))
        if len(nums) == 2:
            cur.append(nums)
            nums = []
    if cur:
        rings.append(np.array(cur, float))
    return rings


def rasterize(rings, x0, y1, res, H, W) -> np.ndarray:
    """Fill rings into an H x W image with the even-odd rule (XOR of every filled ring).
    Pixel (r, c) has centre (x0 + (c+.5)res, y1 - (r+.5)res)."""
    mask = np.zeros((H, W), np.uint8)
    SH = 4                                       # 1/16 px vertex precision
    for R in rings:
        px = (R[:, 0] - x0) / res - 0.5
        py = (y1 - R[:, 1]) / res - 0.5
        c0, c1 = max(int(np.floor(px.min())) - 1, 0), min(int(px.max()) + 2, W)
        r0, r1 = max(int(np.floor(py.min())) - 1, 0), min(int(py.max()) + 2, H)
        if c1 <= c0 or r1 <= r0:
            continue
        sub = np.zeros((r1 - r0, c1 - c0), np.uint8)
        pts = np.round(np.stack([px - c0, py - r0], 1) * (1 << SH)).astype(np.int32)
        cv2.fillPoly(sub, [pts], 1, lineType=cv2.LINE_8, shift=SH)
        mask[r0:r1, c0:c1] ^= sub
    return mask.astype(bool)


class SvgMap:
    """The ``free`` region as a set of rings under the even-odd rule, in metres."""

    def __init__(self, rings, meta: dict, viz: dict | None = None):
        self.rings = [np.asarray(r, float) for r in rings if len(r) >= 3]
        self.meta = dict(meta)
        self.viz = viz or {}
        pts = np.concatenate(self.rings) if self.rings else np.zeros((1, 2))
        self.bbox = (*pts.min(axis=0), *pts.max(axis=0))   # xmin, ymin, xmax, ymax
        self._raster: dict = {}

    @property
    def segments(self) -> np.ndarray:
        """(N, 4) x1, y1, x2, y2 of every edge."""
        segs = [np.hstack([R, np.roll(R, -1, axis=0)]) for R in self.rings]
        return np.concatenate(segs) if segs else np.zeros((0, 4))

    def raster(self, res: float = 0.02, pad: float = 0.2):
        """(mask, x0, y1, res): mask[r, c] = pixel centre (x0 + (c+.5)res, y1 - (r+.5)res) is free."""
        key = (round(res, 6), round(pad, 6))
        if key not in self._raster:
            xmin, ymin, xmax, ymax = self.bbox
            x0, y1 = xmin - pad, ymax + pad
            W = int(np.ceil((xmax + pad - x0) / res))
            H = int(np.ceil((y1 - (ymin - pad)) / res))
            self._raster[key] = (rasterize(self.rings, x0, y1, res, H, W), x0, y1, res)
        return self._raster[key]

    def to_svg(self, style: bool = True) -> str:
        xmin, ymin, xmax, ymax = self.bbox
        pad = 0.2
        w, h = xmax - xmin + 2 * pad, ymax - ymin + 2 * pad
        vb = f"{xmin - pad:.4f} {-(ymax + pad):.4f} {w:.4f} {h:.4f}"
        meta = json.dumps(self.meta, ensure_ascii=False, indent=1)
        viz = "".join(
            f'\n    <path id="{k}" class="{k}" fill-rule="evenodd" d="{rings_to_d(v)}"/>'
            for k, v in self.viz.items())
        css = ("\n  <style>.free{fill:#ffffff;stroke:#000;stroke-width:0.01}"
               ".outdoor{fill:#cdb9eb} .bg{fill:#505050}</style>") if style else ""
        return (f'<?xml version="1.0" encoding="UTF-8"?>\n'
                f'<svg xmlns="{SVG_NS}" viewBox="{vb}" width="{w:.3f}cm" height="{h:.3f}cm">'
                f'{css}\n  <metadata id="hm3denv"><![CDATA[\n{meta}\n]]></metadata>'
                f'\n  <g id="map" transform="scale(1,-1)">'
                f'\n    <rect class="bg" x="{xmin - pad:.4f}" y="{ymin - pad:.4f}" '
                f'width="{w:.4f}" height="{h:.4f}"/>'
                f'\n    <g id="viz">{viz}\n    </g>'
                f'\n    <path id="free" class="free" fill-rule="evenodd" d="{rings_to_d(self.rings)}"/>'
                f'\n  </g>\n</svg>\n')

    def save(self, path) -> None:
        with open(path, "w", encoding="utf-8", newline="\n") as fh:   # same bytes on every OS
            fh.write(self.to_svg().replace("\r\n", "\n"))

    @classmethod
    def load(cls, path) -> "SvgMap":
        root = ET.parse(path).getroot()
        md = root.find(f"{{{SVG_NS}}}metadata")
        meta = json.loads(md.text) if md is not None and md.text else {}
        if meta.get("schema_version") not in (None, *READABLE_SCHEMAS):
            raise ValueError(f"{path}: unsupported schema_version {meta.get('schema_version')}")
        free, viz = None, {}
        for p in root.iter(f"{{{SVG_NS}}}path"):
            if p.get("id") == "free":
                free = d_to_rings(p.get("d", ""))
            elif p.get("id"):
                viz[p.get("id")] = d_to_rings(p.get("d", ""))
        if free is None:
            raise ValueError(f"{path}: no path with id='free'")
        return cls(free, meta, viz)
