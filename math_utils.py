import math

"""
import sympy

w, x, y, z = sympy.symbols('w x y z')
q = sympy.Quaternion(w, x, y, z)
q.set_norm(1)
project_gravity = sympy.Quaternion.rotate_point((0, 0, -1), q.inverse())

gx = 2 * w * y - 2 * x * z
gy = -2 * w * x - 2 * y * z
gz = -w * w + x * x + y * y - z * z

assert project_gravity == (gx, gy, gz)
"""


def project_gravity(quaternion: 'list[float]'):
    w, x, y, z = quaternion  # assume normalized
    gx = 2 * w * y - 2 * x * z
    gy = -2 * w * x - 2 * y * z
    gz = -w * w + x * x + y * y - z * z
    return [gx, gy, gz]


def wrap_to_pi(x: float):
    # wrap ℝ to [-π, π] while preserving cos(x) and sin(x)
    # (x ± math.pi) % (2 * math.pi) ± math.pi
    return math.atan2(math.sin(x), math.cos(x))
