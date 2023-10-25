import importlib.machinery
import importlib.util
import struct
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread

import torch

finder = importlib.machinery.PathFinder()
spec = finder.find_spec(
    'robot_interface',
    [str(Path(__file__).resolve().parent)]
)
if spec is None:
    raise
sdk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sdk)

id_to_real_index = {
    'FR_0': 0, 'FR_1': 1, 'FR_2': 2,
    'FL_0': 3, 'FL_1': 4, 'FL_2': 5,
    'RR_0': 6, 'RR_1': 7, 'RR_2': 8,
    'RL_0': 9, 'RL_1': 10, 'RL_2': 11,
}

real_index_to_id = {v: k for k, v in id_to_real_index.items()}

name_to_id = {
    'FL_hip_joint': 'FL_0', 'FL_thigh_joint': 'FL_1', 'FL_calf_joint': 'FL_2',
    'FR_hip_joint': 'FR_0', 'FR_thigh_joint': 'FR_1', 'FR_calf_joint': 'FR_2',
    'RL_hip_joint': 'RL_0', 'RL_thigh_joint': 'RL_1', 'RL_calf_joint': 'RL_2',
    'RR_hip_joint': 'RR_0', 'RR_thigh_joint': 'RR_1', 'RR_calf_joint': 'RR_2',
}

id_to_name = {v: k for k, v in name_to_id.items()}

sim_index_to_name = [
    'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
    'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
    'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
    'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
]

name_to_sim_index = {name: i for i, name in enumerate(sim_index_to_name)}

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

# sim index to real index mapping is for real-to-sim conversion
# for sim_idx, real_idx in enumerate(sim_index_to_real_index):
#     assert real_joints[real_idx] == sim_joints[sim_idx]
sim_idx_to_real_idx = [id_to_real_index[name_to_id[name]] for name in sim_index_to_name]

# real index to sim index mapping is for sim-to-real conversion
# even if dict is ordered in Python 3.7 or later, it is not sorted by its key
# thus, we need to use `real_index_to_id[i] for i in range(num_joints)`
# instead of `for id in real_index_to_id.values()`
real_idx_to_sim_idx = [name_to_sim_index[id_to_name[real_index_to_id[i]]] for i in range(num_joints)]

# 模: sim, 真: real
q0_模 = [name_to_q0[name] for name in sim_index_to_name]
q0_真 = [name_to_q0[id_to_name[real_index_to_id[i]]] for i in range(num_joints)]


@dataclass
class RobotObservation:
    joint_position: torch.Tensor
    joint_velocity: torch.Tensor
    gyroscope: torch.Tensor
    quaternion: torch.Tensor
    roll: float
    pitch: float
    yaw: float
    lx: float
    ly: float
    rx: float
    ry: float
    L1: bool
    L2: bool


class Robot:
    LOWLEVEL = 0xFF

    def __init__(self):
        self.state = sdk.LowState()

        # pybind11 will convert std::array into python list
        self.motor_state_实 = self.state.motorState[:12]  # std::array<MotorState, 20>
        self.motor_state_模 = [self.motor_state_实[i] for i in sim_idx_to_real_idx]

        self.imu = self.state.imu

        # the struct module does have compiled format cache
        # reuse the Struct object explicitly for clarity, not performance
        self.rocker_struct = struct.Struct('@5f')

        self.Δq_真 = [float('NaN') for _ in range(12)]

        self.stopped = Event()
        self.background_thread = Thread(target=self._send_recv_loop, daemon=True)
        self.background_thread.start()

    def _send_recv_loop(self):
        # parameter
        Kp = 20.0
        Kd = 0.5

        # shorthand
        state = self.state
        stopped = self.stopped

        # setup
        udp = sdk.UDP(Robot.LOWLEVEL, 8080, '192.168.123.10', 8007)
        safe = sdk.Safety(sdk.LeggedType.Go1)

        udp.Send()  # make MCU happy
        while not stopped.wait(0.005):
            udp.Recv()
            udp.GetRecv(state)
            if state.tick != 0:
                self.Δq_真 = [ms.q - q0 for ms, q0 in zip(state.motorState, q0_真)]
                break

        # === Loop Invariant Code Motion ===
        cmd = sdk.LowCmd()
        udp.InitCmdData(cmd)
        motor_cmd = cmd.motorCmd[:12]
        # PositionLimit and PowerProtect won't modify dq, Kp, Kd, tau
        for mc in motor_cmd:
            mc.dq = 0.0     # expected velocity
            mc.Kp = Kp      # stiffness
            mc.Kd = Kd      # damping
            mc.tau = 0.0    # expected torque
        # === Loop Invariant Code Motion ===

        # Microbenchmark for torch.Tensor versus builtin.list
        # for len=12, torch.float32 and builtin.float(f64)
        # on Go1.NX CPU and Python 3.6.9
        # copy:
        #   tolist + copy from list cost ~(1.5+2.3) μs
        #   directly copy from torch tensor to list cost ~70 μs
        # permute:
        #   permute tensor by tensor cost ~11 μs
        #   permute list by list cost ~2.5 μs (assignment) / ~1.9 μs (comprenhension)
        # add:
        #   add tensor by tensor cost ~5.9us
        #   add list by list cost ~3.1us
        #
        # additional observation:
        #   tensor has a much higher peak latency than list
        #
        # conclusion:
        #   for len=12, use list instead of tensor

        # Microbenchmark for pybind11 STL container automatic conversion performance
        # NOTE: This automatic conversions involve a copy operation
        #       that prevents pass-by-reference semantics.
        #       However, the elements in the container are still passed by reference.
        # NOTE: state.motorState[i] will first construct a list, then get item from it.
        #       Thus, [state.motorState[i].q for i in range(12)] will construct the list 12 times.
        # on Go1.NX CPU and Python 3.6.9
        # x = [state.motorState[i].q for i in range(12)] cost ~175 μs
        # x = [ms.q for ms in motor_state] cost ~9 μs
        # cmd.motorCmd[i].q = qs[i] cost ~185 μs
        # mc.q = q for mc, q in zip(motor_cmd, qs) cost ~12 μs

        while not stopped.wait(0.005):
            for mc, q0, Δq in zip(motor_cmd, q0_真, self.Δq_真):
                mc.q = q0 + Δq  # expected position
            safe.PositionLimit(cmd)
            udp.Recv()
            udp.GetRecv(state)
            safe.PowerProtect(cmd, state, 5)
            udp.SetSend(cmd)
            udp.Send()

    def get_obs(self):
        dtype = torch.float32

        # shorthand
        motor_state_模 = self.motor_state_模

        joint_position = torch.tensor([ms.q - q0 for ms, q0 in zip(motor_state_模, q0_模)], dtype=dtype)
        joint_velocity = torch.tensor([ms.dq for ms in motor_state_模], dtype=dtype)

        # imu.gyroscope & .rpy: std::array<float, 3>
        # imu.quaternion: std::array<float, 4>
        # => list[float]
        imu = self.imu
        gyroscope = torch.tensor(imu.gyroscope, dtype=dtype)  # rpy order, rad/s
        quaternion = torch.tensor(imu.quaternion, dtype=dtype)  # (w, x, y, z) order, normalized
        roll, pitch, yaw = imu.rpy  # rpy order, rad

        rc = self.state.wirelessRemote  # std::array<uint8_t, 40> => list[int]
        # stdlib struct.unpack is faster than convert tensor then .view(torch.float32)
        # and torch<=1.10 only support .view to dtype with same size
        lx, rx, ry, _, ly = self.rocker_struct.unpack(bytes(rc[4:24]))

        # LSB -> MSB
        # rc[2] = [R1, L1, start, select][R2, L2, F1, F2]
        # rc[3] = [A, B, X, Y][up, right, down, left]
        button_L1 = rc[2] & 0b0000_0010
        button_L2 = rc[2] & 0b0010_0000

        return RobotObservation(
            joint_position=joint_position,  # in sim order, relative to q0
            joint_velocity=joint_velocity,  # in sim order
            gyroscope=gyroscope,
            quaternion=quaternion,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            lx=lx,
            ly=ly,
            rx=rx,
            ry=ry,
            L1=bool(button_L1),
            L2=bool(button_L2),
        )

    def set_act(self, action: torch.Tensor):
        Δq_模 = action.tolist()
        self.Δq_真 = [Δq_模[i] for i in real_idx_to_sim_idx]

    def init(self):
        import math
        stopped = self.stopped
        while any(math.isnan(Δq) for Δq in self.Δq_真) and not stopped.wait(0.05):
            pass

        Δq_sequence = []
        Δq_t = self.Δq_真
        while any(abs(Δq) > 0.01 for Δq in Δq_t):
            # reduce Δq magnitude by 0.05 rad per step
            Δq_t = [math.copysign(max(0, abs(Δq) - 0.05), Δq) for Δq in Δq_t]
            Δq_sequence.append(Δq_t)

        for Δq_t in Δq_sequence:
            self.Δq_真 = Δq_t
            # 0.05 sec per step
            if stopped.wait(0.05):
                break
