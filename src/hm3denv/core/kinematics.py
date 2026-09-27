"""Exact integration of planar motion with constant body velocity (vx, vy, w) over dt.
Differential drive is the vy = 0 case (circular arc); omnidirectional (mecanum) uses vy.

    p(dt) = p0 + integral_0^dt R(theta0 + w t) [vx, vy] dt
"""

import math

W_EPS = 1e-9


def integrate(x, y, th, vx, vy, w, dt):
    th1 = th + w * dt
    if abs(w) < W_EPS:
        c, s = math.cos(th) * dt, math.sin(th) * dt
    else:
        c = (math.sin(th1) - math.sin(th)) / w          # integral of cos
        s = -(math.cos(th1) - math.cos(th)) / w         # integral of sin
    return (x + vx * c - vy * s, y + vx * s + vy * c, wrap(th1))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi
