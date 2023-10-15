import importlib.machinery
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from threading import Thread

import numpy as np
import torch

id_to_real_index = {
    'FR_0': 0, 'FR_1': 1, 'FR_2': 2,
    'FL_0': 3, 'FL_1': 4, 'FL_2': 5,
    'RR_0': 6, 'RR_1': 7, 'RR_2': 8,
    'RL_0': 9, 'RL_1': 10, 'RL_2': 11,
}

name_to_id = {
    'FL_hip_joint': 'FL_0', 'FL_thigh_joint': 'FL_1', 'FL_calf_joint': 'FL_2',
    'FR_hip_joint': 'FR_0', 'FR_thigh_joint': 'FR_1', 'FR_calf_joint': 'FR_2',
    'RL_hip_joint': 'RL_0', 'RL_thigh_joint': 'RL_1', 'RL_calf_joint': 'RL_2',
    'RR_hip_joint': 'RR_0', 'RR_thigh_joint': 'RR_1', 'RR_calf_joint': 'RR_2',
}

sim_index_to_name = [
    'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
    'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
    'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
    'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
]

name_to_q0 = {
    'FL_hip_joint': 0.1,
    'RL_hip_joint': 0.1,
    'FR_hip_joint': -0.1,
    'RR_hip_joint': -0.1,
    'FL_thigh_joint': 0.8,
    'RL_thigh_joint': 1.0,
    'FR_thigh_joint': 0.8,
    'RR_thigh_joint': 1.0,
    'FL_calf_joint': -1.5,
    'RL_calf_joint': -1.5,
    'FR_calf_joint': -1.5,
    'RR_calf_joint': -1.5,
}

num_joints = len(sim_index_to_name)

# joint_real_to_sim[i] = j means that the i-th joint in sim corresponds to the j-th joint in real
joint_real_to_sim = torch.tensor([id_to_real_index[name_to_id[name]] for name in sim_index_to_name])

# joint_sim_to_real[i] = j means that the i-th joint in real corresponds to the j-th joint in sim
joint_sim_to_real = torch.argsort(joint_real_to_sim)

q0 = torch.tensor([name_to_q0[name] for name in sim_index_to_name], dtype=torch.float32)


@dataclass
class RobotObservation:
    joint_position: torch.Tensor
    joint_velocity: torch.Tensor
    gyroscope: torch.Tensor
    quaternion: torch.Tensor
    rpy: torch.Tensor
    lx: torch.Tensor
    ly: torch.Tensor
    rx: torch.Tensor
    ry: torch.Tensor


class Robot:
    LOWLEVEL = 0xFF

    def __init__(self):
        finder = importlib.machinery.PathFinder()
        spec = finder.find_spec(
            'robot_interface',
            [str(Path(__file__).resolve().parent)]
        )
        if spec is None:
            raise
        sdk = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sdk)
        self.sdk = sdk
        self.udp = sdk.UDP(Robot.LOWLEVEL, 8080, "192.168.123.10", 8007)

        self.safe = sdk.Safety(sdk.LeggedType.Go1)
        self.power_protect_state = sdk.LowState()

        self.background_thread = Thread(target=self._poll, daemon=True)
        self.background_thread.start()

    def _poll(self):
        self.udp.Send()
        self.udp.Recv()

    def step(self, action: torch.Tensor):
        self.set_act(action)
        return self.get_obs()

    def get_obs(self):
        dtype = torch.float32

        state = self.retrieve_state()

        joint_position_real = torch.tensor([state.motorState[i].q for i in range(12)], dtype=dtype)
        joint_velocity_real = torch.tensor([state.motorState[i].dq for i in range(12)], dtype=dtype)

        joint_position_sim = joint_position_real[joint_real_to_sim]
        joint_velocity_sim = joint_velocity_real[joint_real_to_sim]

        # pybind11 will convert std::array into python list
        gyroscope = torch.tensor(state.imu.gyroscope, dtype=dtype)  # rpy order, rad/s
        quaternion = torch.tensor(state.imu.quaternion, dtype=dtype)  # (w, x, y, z) order, normalized
        rpy = torch.tensor(state.imu.rpy, dtype=dtype)  # rpy order, rad

        keydata = torch.from_numpy(np.array(state.wirelessRemote, dtype='uint8')[4:24].view(np.float32))
        lx, rx, ry, _, ly = keydata

        return RobotObservation(
            joint_position=joint_position_sim,
            joint_velocity=joint_velocity_sim,
            gyroscope=gyroscope,
            quaternion=quaternion,
            rpy=rpy,
            lx=lx,
            ly=ly,
            rx=rx,
            ry=ry,
        )

    def set_act(self, action: torch.Tensor):
        Kp = 20.0
        Kd = 0.5

        cmd = self.sdk.LowCmd()
        self.udp.InitCmdData(cmd)

        q = (action + q0)[joint_sim_to_real]

        for i in range(12):
            cmd.motorCmd[i].q = q[i]    # expected position
            cmd.motorCmd[i].dq = 0.0    # expected velocity
            cmd.motorCmd[i].Kp = Kp     # stiffness
            cmd.motorCmd[i].Kd = Kd     # damping
            cmd.motorCmd[i].tau = 0.0   # expected torque

        self.safe.PowerProtect(cmd, self.retrieve_state(), 5)
        self.udp.SetSend(cmd)

    def retrieve_state(self):
        state = self.sdk.LowState()
        self.udp.GetRecv(state)
        return state
