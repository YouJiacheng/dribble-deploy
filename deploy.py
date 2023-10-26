import math
from pathlib import Path

import torch
import zmq

from math_utils import project_gravity, wrap_to_pi
from robot import Robot, RobotObservation


class BallDetector:
    def __init__(self):
        ctx: 'zmq.Context[zmq.Socket]' = zmq.Context.instance()
        self.socket = sock = ctx.socket(zmq.DEALER)
        sock.set(zmq.CONFLATE, 1)
        sock.bind('tcp://127.0.0.1:5555')
        self.box_corner = None

    def refresh(self):
        try:
            score, box_corner = self.socket.recv_pyobj(flags=zmq.NOBLOCK)
        except zmq.error.Again:
            return

        self.box_corner = box_corner if score > 0.7 else None

    def get_ball_pos(self):
        if self.box_corner is None:
            return [0.2, 0.0, 0.0]
        x0, y0, x1, y1 = self.box_corner
        image_width = 480
        offset_from_center = (x0 + x1) / 2 - image_width / 2
        # the line y = 0 (the x-axis) has offset_from_center ≈ -40
        offset_from_x_axis = offset_from_center + 40

        w = x1 - x0
        h = y1 - y0
        size = math.sqrt(w * h)

        # ball at 1m distance has size 20px
        r = 20 / size

        # the positive direction of offset points to the right
        # the positive direction of θ is counter-clockwise
        # the ray θ = 0 points forward

        θ = -math.radians(0.4 * offset_from_x_axis)  # 0.4° per pixel

        # right-handed coordinate system
        # the positive x-axis points forward
        # the positive y-axis points to the left
        # the positive z-axis points up
        return [r * math.cos(θ), r * math.sin(θ), 0.0]


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
    offset = 0.0
    bound = 0.0

    foot_gait_offsets = [phase + offset + bound, offset, bound, phase]

    duration = 0.5  # duration = stance / (stance + swing)
    step_frequency = 3.0

    control_decimation = 4
    simulation_dt = 0.005
    dt = control_decimation * simulation_dt

    action_scale = 0.25
    hip_scale_reduction = torch.tensor([0.5, 1, 1] * 4, dtype=torch.float32)

    def __init__(self, history_len: int, robot: Robot, ball_detector: BallDetector):
        assert history_len > 0

        self.history_len = history_len
        self.buffer = torch.zeros(history_len * 3, self.obs_dim, dtype=torch.float32)
        self.t = history_len

        self.action_t = torch.zeros(self.act_dim, dtype=torch.float32)
        self.action_t_minus1 = torch.zeros(self.act_dim, dtype=torch.float32)

        self.gait_index = 0.0

        self.yaw_init = 0.0

        self.robot = robot
        self.ball_detector = ball_detector

    def observe(self):
        self.ball_detector.refresh()
        robot_obs = self.robot.get_obs()
        obs = self.make_obs(robot_obs)
        self.store_obs(obs)
        return self.buffer[self.t - self.history_len:self.t], robot_obs

    def advance(self, action: torch.Tensor):
        self.action_t_minus1[:] = self.action_t
        self.action_t[:] = action

        action_scaled = action * self.action_scale * self.hip_scale_reduction
        self.robot.set_act(action_scaled.tolist())
        self.gait_index = (self.gait_index + self.step_frequency * self.dt) % 1

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
        commands = [
            # rocker x: left/right
            # rocker y: forward/backward
            robot_obs.ly * 2,   # x vel
            robot_obs.lx * 2,   # y vel
            0.0 * 0.25,         # yaw vel
            0.0 * 2,            # body height
            self.step_frequency,
            self.phase,
            self.offset,
            self.bound,
            self.duration,
            0.09 * 0.15,        # foot swing height
            0.0 * 0.3,          # pitch
            0.0 * 0.3,          # roll
            0.0,                # stance_width
            0.1 / 2,            # stance length
            0.01 / 2,           # unknown
        ]
        dof_pos = robot_obs.joint_position
        dof_vel = [v * 0.05 for v in robot_obs.joint_velocity]
        action = self.action_t
        last_action = self.action_t_minus1

        clock = self.clock()
        yaw = wrap_to_pi(robot_obs.yaw - self.yaw_init)
        timing = self.gait_index

        return torch.cat([
            torch.tensor([
                *ball_pos,
                *projected_gravity,
                *commands,
                *dof_pos,
                *dof_vel,
            ], dtype=torch.float32),
            action, last_action,
            torch.tensor([*clock, yaw, timing], dtype=torch.float32),
        ])

    def clock(self):
        return [
            math.sin(2 * math.pi * (self.gait_index + offset))
            for offset in self.foot_gait_offsets
        ]


def main():
    import time
    policy = load_policy(Path(__file__).resolve().parent)
    robot = Robot()
    ball_detector = BallDetector()
    env = DribbleEnv(history_len=15, robot=robot, ball_detector=ball_detector)

    robot.init()
    while True:
        obs, robot_obs = env.observe()
        env.advance(torch.zeros(12, dtype=torch.float32))
        if robot_obs.L1:
            break
        time.sleep(0.02)

    env.yaw_init = robot_obs.yaw

    while True:
        obs, robot_obs = env.observe()
        action = policy(obs)
        env.advance(action)
        if robot_obs.L2:
            break
        time.sleep(0.02)

    robot.stopped.set()
    robot.background_thread.join()


if __name__ == '__main__':
    main()
