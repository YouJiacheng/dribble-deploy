import math

import torch


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor):
    w1, u1 = q1[..., :1], q1[..., 1:]
    w2, u2 = q2[..., :1], q2[..., 1:]

    scalar = w1 * w2 - torch.einsum('...i,...i', u1, u2).unsqueeze(-1)
    vector = w1 * u2 + w2 * u1 + torch.cross(u1, u2, dim=-1)

    return torch.cat([scalar, vector], dim=-1)


def quaternion_conjugate(q: torch.Tensor):
    w, u = q[..., :1], q[..., 1:]
    return torch.cat([w, -u], dim=-1)


def rotate_vector_by_quaternion(v: torch.Tensor, q: torch.Tensor):
    v_quaternion = torch.cat([v.new_zeros(*v.shape[:-1], 1), v], dim=-1)
    rotated_quaternion = quaternion_multiply(q, quaternion_multiply(v_quaternion, quaternion_conjugate(q)))
    return rotated_quaternion[..., 1:]


def project_gravity(quaternion: torch.Tensor):
    gravity = torch.tensor([0.0, 0.0, -1], dtype=torch.float32)
    return rotate_vector_by_quaternion(gravity, quaternion_conjugate(quaternion))


def wrap_to_pi(x: float):
    # wrap ℝ to [-π, π] while preserving cos(x) and sin(x)
    # (x ± math.pi) % (2 * math.pi) ± math.pi
    return math.atan2(math.sin(x), math.cos(x))
