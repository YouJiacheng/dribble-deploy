import math
from pathlib import Path

import torch
import zmq


class BallDetector:
    def __init__(self):
        ctx: 'zmq.Context[zmq.Socket]' = zmq.Context.instance()
        self.socket = sock = ctx.socket(zmq.DEALER)
        sock.set(zmq.CONFLATE, 1)
        sock.connect('tcp://127.0.0.1:5555')
        self.box_corner = None

    def refresh(self):
        try:
            score, box_corner = self.socket.recv_pyobj(flags=zmq.NOBLOCK)
        except zmq.error.Again:
            return

        if score + 1 > 0.5:
            self.box_corner = box_corner

    def get_ball_pos(self):
        if self.box_corner is None:
            return torch.tensor([0.5, 0.0, 0.0], dtype=torch.float32)
        x0, y0, x1, y1 = self.box_corner
        image_width = 480
        offset = (x0 + x1) / 2 - image_width / 2
        w = x1 - x0
        h = y1 - y0
        size = math.sqrt(w * h)

        r = 20 / size  # ball at 1m distance has size 20px
        θ = offset * 0.3 * (math.pi / 180)  # 0.3 deg per pixel
        return torch.tensor([r * math.cos(θ), r * math.sin(θ), 0.0], dtype=torch.float32)


def load_policy(root: Path):
    body = torch.jit.load(root / 'body.jit', map_location='cpu')
    adaptation_module = torch.jit.load(root / 'adaptation_module.jit', map_location='cpu')

    @torch.no_grad()
    def policy(stacked_history: torch.Tensor):
        # stacked_history: (H, d) = (15, 75)
        history = stacked_history.reshape(1, 1125)
        latent = adaptation_module(history)  # (1, 6)
        composed = torch.cat((history, latent), dim=-1)
        action = body(composed)
        return action

    return policy


def env_transformed(history_len: int):
    from math_utils import quaternion_conjugate, rotate_vector_by_quaternion
    from robot import Robot, RobotObservation
    robot = Robot()
    ball_detector = BallDetector()

    obs_dim = 75
    act_dim = 12

    assert history_len > 0
    buffer_len = history_len * 3
    buffer = torch.zeros(buffer_len, obs_dim, dtype=torch.float32)
    t = history_len

    action_t = torch.zeros(act_dim, dtype=torch.float32)
    action_t_minus1 = torch.zeros(act_dim, dtype=torch.float32)

    # gait type parameters
    phase = 0.5
    offset = 0
    bound = 0

    duration = 0.5  # duration = stance / (stance + swing)
    step_frequency = 3

    control_decimation = 4
    simulation_dt = 0.005
    dt = control_decimation * simulation_dt

    gait_index = torch.zeros(1, dtype=torch.float32)
    foot_gait_offset = torch.tensor([phase + offset + bound, offset, bound, phase], dtype=torch.float32)

    def store_obs(obs: torch.Tensor):
        nonlocal t
        if t == buffer_len:
            buffer[:history_len] = buffer[t - history_len:t].clone()
            t = history_len

        buffer[t] = obs
        t += 1

    def time_step():
        gait_index.add_(dt * step_frequency).remainder_(1)

    def wrap_to_pi_inplace(x: torch.Tensor):
        x.remainder_(2 * torch.pi)
        x[x > torch.pi] -= 2 * torch.pi
        return x

    def project_gravity(quaternion: torch.Tensor) -> torch.Tensor:
        gravity = torch.tensor([0.0, 0.0, -1], dtype=torch.float32)
        return rotate_vector_by_quaternion(gravity, quaternion_conjugate(quaternion))

    def transform(robot_obs: RobotObservation) -> torch.Tensor:
        ball_pos = ball_detector.get_ball_pos()
        projected_gravity = project_gravity(robot_obs.quaternion)
        commands = torch.tensor([
            robot_obs.lx,  # x vel
            robot_obs.ly,  # y vel
            0,  # yaw vel
            0,  # body height
            3,  # step frequency
            phase,
            offset,
            bound,
            duration,
            0.09,  # foot swing height
            0,  # pitch
            0,  # roll
            0,  # stance_width
            0.1 / 2,  # stance length
            0.01 / 2,  # unknown
        ], dtype=torch.float32)
        dof_pos = robot_obs.joint_position
        dof_vel = robot_obs.joint_velocity
        action = action_t
        last_action = action_t_minus1
        clock = torch.sin(2 * torch.pi * (gait_index + foot_gait_offset))
        yaw = wrap_to_pi_inplace(robot_obs.rpy[None, 2].clone())
        timing = gait_index
        return torch.cat([
            ball_pos,
            projected_gravity,
            commands,
            dof_pos, dof_vel,
            action, last_action,
            clock,
            yaw,
            timing,
        ])

    def step(action: torch.Tensor):
        action_t_minus1[:] = action_t
        action_t[:] = action

        robot_obs = robot.step(action)
        time_step()  # gait clock
        ball_detector.refresh()
        obs = transform(robot_obs)
        store_obs(obs)
        return buffer[t - history_len:t]

    return step
