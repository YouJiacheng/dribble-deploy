import math
from pathlib import Path

import torch
import zmq

from math_utils import project_gravity, wrap_to_pi_
from robot import Robot, RobotObservation


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
        return action[0]

    return policy


class RealtimeEnv:
    def observe(self): ...
    def advance(self, action): ...


class DribbleEnv(RealtimeEnv):
    obs_dim = 75
    act_dim = 12

    # gait type parameters
    phase = 0.5
    offset = 0
    bound = 0

    foot_gait_offsets = torch.tensor([phase + offset + bound, offset, bound, phase], dtype=torch.float32)

    duration = 0.5  # duration = stance / (stance + swing)
    step_frequency = 3

    control_decimation = 4
    simulation_dt = 0.005
    dt = control_decimation * simulation_dt

    action_scale = 0.25
    hip_scale_reduction = torch.tensor([0.5, 1, 1] * 4, dtype=torch.float32)

    def __init__(self, history_len: int):
        assert history_len > 0

        self.history_len = history_len
        self.buffer = torch.zeros(history_len * 3, self.obs_dim, dtype=torch.float32)
        self.t = history_len

        self.action_t = torch.zeros(self.act_dim, dtype=torch.float32)
        self.action_t_minus1 = torch.zeros(self.act_dim, dtype=torch.float32)

        self.gait_index = torch.zeros(1, dtype=torch.float32)

        self.robot = Robot()
        self.ball_detector = BallDetector()

    def observe(self):
        self.ball_detector.refresh()
        robot_obs = self.robot.get_obs()
        obs = self.make_obs(robot_obs)
        self.store_obs(obs)
        return self.buffer[self.t - self.history_len:self.t]

    def advance(self, action):
        self.action_t_minus1[:] = self.action_t
        self.action_t[:] = action

        action_scaled = action * self.action_scale * self.hip_scale_reduction
        self.robot.set_act(action_scaled)
        self.gait_index.add_(self.dt * self.step_frequency).remainder_(1)

    def store_obs(self, obs: torch.Tensor):
        h, buffer, t = self.history_len, self.buffer, self.t
        if t == buffer.shape[0]:
            buffer[:h] = buffer[t - h:t].clone()
            t = h
        buffer[t] = obs
        self.t = t + 1

    def make_obs(self, robot_obs: RobotObservation) -> torch.Tensor:
        ball_pos = self.ball_detector.get_ball_pos()
        projected_gravity = project_gravity(robot_obs.quaternion)
        commands = torch.tensor([
            # rocker x: left/right
            # rocker y: forward/backward
            robot_obs.ly * 2,   # x vel
            robot_obs.lx * 2,   # y vel
            0 * 0.25,           # yaw vel
            0 * 2,              # body height
            self.step_frequency,
            self.phase,
            self.offset,
            self.bound,
            self.duration,
            0.09 * 0.15,        # foot swing height
            0 * 0.3,            # pitch
            0 * 0.3,            # roll
            0,                  # stance_width
            0.1 / 2,            # stance length
            0.01 / 2,           # unknown
        ], dtype=torch.float32)
        dof_pos = robot_obs.joint_position
        dof_vel = robot_obs.joint_velocity * 0.05
        action = self.action_t
        last_action = self.action_t_minus1
        clock = torch.sin(2 * torch.pi * self.foot_indices())
        yaw = wrap_to_pi_(robot_obs.rpy[None, 2].clone())
        timing = self.gait_index

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

    def foot_indices(self):
        return self.gait_index + self.foot_gait_offsets


