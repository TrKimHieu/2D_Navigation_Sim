"""A 6 x 4 m box room (Z up) with one 0.8 m box, as a GLB: the synthetic scene of the build
tests, also used by CI to try build-map (`python tests/room_glb.py room.glb`). Needs trimesh."""

import sys


def room(obstacle_x):
    """6 x 4 m room (Z up): floor z=0, ceiling z=2.5, four walls, one 0.8 m box."""
    import trimesh
    parts = []
    for ext, c in (([6, 4, 0.1], [3, 2, -0.05]), ([6, 4, 0.1], [3, 2, 2.55]),
                   ([0.1, 4, 2.5], [-0.05, 2, 1.25]), ([0.1, 4, 2.5], [6.05, 2, 1.25]),
                   ([6, 0.1, 2.5], [3, -0.05, 1.25]), ([6, 0.1, 2.5], [3, 4.05, 1.25]),
                   ([0.8, 0.8, 0.8], [obstacle_x, 2, 0.4])):
        b = trimesh.creation.box(extents=ext)
        b.apply_translation(c)
        parts.append(b)
    return trimesh.util.concatenate(parts)


if __name__ == "__main__":
    room(2.0).export(sys.argv[1] if len(sys.argv) > 1 else "room.glb")
