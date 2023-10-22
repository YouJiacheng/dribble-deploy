import importlib.machinery
import importlib.util
from pathlib import Path

from time import perf_counter, sleep
from tqdm import tqdm


def benchmark_udp(sdk, udp):
    state = sdk.LowState()
    # UDP Recv cost
    with tqdm() as pbar:
        while True:
            udp.Recv()
            pbar.update(1)
            if pbar.n == int(1e5):
                break

    with tqdm() as pbar:
        while True:
            udp.Recv()
            udp.GetRecv(state)
            pbar.update(1)
            if pbar.n == int(1e5):
                break

    # UDP update frequency
    with tqdm() as pbar:
        q = tuple(0.0 for _ in range(12))
        while True:
            udp.Recv()
            udp.GetRecv(state)
            q_new = tuple(m.q for m in state.motorState)
            if q_new == q:
                continue
            pbar.update(1)
            q = q_new
            if pbar.n == 3000:
                break

    with tqdm() as pbar:
        imu = state.imu
        r = 0
        while True:
            udp.Recv()
            udp.GetRecv(state)
            r_new = imu.rpy[0]
            if r_new == r:
                continue
            pbar.update(1)
            r = r_new
            if pbar.n == 3000:
                break


def benchmark_get_state(sdk, udp):
    state = sdk.LowState()
    motor_state = state.motorState[:12]
    x = [0.0 for _ in range(12)]

    # test correctness
    udp.GetRecv(state)
    assert all(state.motorState[i].q == motor_state[i].q for i in range(12))

    t = perf_counter()
    for _ in range(int(1e4)):
        for i in range(12):
            x[i] = state.motorState[i].q
    Δt_μs = (perf_counter() - t) * 100
    print(f'x[i] = state.motorState[i].q loop: {Δt_μs} μs')

    t = perf_counter()
    for _ in range(int(1e4)):
        x = [state.motorState[i].q for i in range(12)]
    Δt_μs = (perf_counter() - t) * 100
    print(f'x = [state.motorState[i].q for i in range(12)]: {Δt_μs} μs')

    t = perf_counter()
    for _ in range(int(1e4)):
        for i in range(12):
            x[i] = motor_state[i].q
    Δt_μs = (perf_counter() - t) * 100
    print(f'x[i] = motor_state[i].q loop: {Δt_μs} μs')

    t = perf_counter()
    for _ in range(int(1e4)):
        for i, ms in enumerate(motor_state):
            x[i] = ms.q
    Δt_μs = (perf_counter() - t) * 100
    print(f'x[i] = ms.q enumerate loop: {Δt_μs} μs')

    t = perf_counter()
    for _ in range(int(1e4)):
        x = [ms.q for ms in motor_state]
    Δt_μs = (perf_counter() - t) * 100
    print(f'x = [ms.q for ms in motor_state]: {Δt_μs} μs')


def benchmark_set_command(sdk, udp):
    cmd = sdk.LowCmd()
    udp.InitCmdData(cmd)
    motor_cmd = cmd.motorCmd[:12]

    # test correctness
    motor_cmd_0 = motor_cmd[0]
    motor_cmd_0.q = 1e9
    assert cmd.motorCmd[0].q == 1e9
    motor_cmd_0.q = 0
    assert cmd.motorCmd[0].q == 0

    qs = [0.0 for _ in range(12)]

    t = perf_counter()
    for _ in range(int(1e4)):
        for i in range(12):
            cmd.motorCmd[i].q = qs[i]
    Δt_μs = (perf_counter() - t) * 100
    print(f'cmd.motorCmd[i].q = qs[i] loop: {Δt_μs} μs')

    t = perf_counter()
    for _ in range(int(1e4)):
        for i in range(12):
            motor_cmd[i].q = qs[i]
    Δt_μs = (perf_counter() - t) * 100
    print(f'motor_cmd[i].q = qs[i] loop: {Δt_μs} μs')

    t = perf_counter()
    for _ in range(int(1e4)):
        for mc, q in zip(motor_cmd, qs):
            mc.q = q
    Δt_μs = (perf_counter() - t) * 100
    print(f'mc.q = q for mc, q in zip(motor_cmd, qs): {Δt_μs} μs')


def benchmark_interface():
    finder = importlib.machinery.PathFinder()
    spec = finder.find_spec(
        'robot_interface',
        [str(Path(__file__).resolve().parent)]
    )
    if spec is None:
        raise
    sdk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sdk)

    udp = sdk.UDP(0xFF, 8080, '192.168.123.10', 8007)
    state = sdk.LowState()

    udp.Send()
    with tqdm() as pbar:
        while True:
            udp.Recv()
            udp.GetRecv(state)
            if state.tick != 0:
                break
            pbar.update(1)
            sleep(0.01)

    t = perf_counter()
    for _ in range(int(1e4)):
        pass
    Δt_μs = (perf_counter() - t) * 100
    print(f'pass: {Δt_μs} μs')

    benchmark_udp(sdk, udp)
    benchmark_get_state(sdk, udp)
    benchmark_set_command(sdk, udp)


def benchmark_list_v_tensor():
    import torch
    x = torch.zeros(12, dtype=torch.float32)
    y = torch.zeros(12, dtype=torch.float32)
    z = torch.zeros(12, dtype=torch.float32)
    k = torch.tensor([i for i in range(12)])
    xx = x.tolist()
    yy = y.tolist()
    zz = z.tolist()
    kk = k.tolist()

    def benchmark_add():
        t = perf_counter()
        for _ in range(int(1e5)):
            w = x + y
        Δt_μs = (perf_counter() - t) * 10
        print(f'tensor add: {Δt_μs} μs')

        t = perf_counter()
        for _ in range(int(1e5)):
            for i in range(12):
                zz[i] = xx[i] + yy[i]
        Δt_μs = (perf_counter() - t) * 10
        print(f'list add with nonlocal variable: {Δt_μs} μs')

        def add(xx, yy):
            for i in range(12):
                zz[i] = xx[i] + yy[i]

        t = perf_counter()
        for _ in range(int(1e5)):
            add(xx, yy)
        Δt_μs = (perf_counter() - t) * 10
        print(f'list add with local variable (in function): {Δt_μs} μs')

        class A:
            def __init__(self, xx, yy, zz):
                self.xx = xx
                self.yy = yy
                self.zz = zz

            def add(self):
                for i in range(12):
                    self.zz[i] = self.xx[i] + self.yy[i]

        a = A(xx, yy, zz)
        t = perf_counter()
        for _ in range(int(1e5)):
            a.add()
        Δt_μs = (perf_counter() - t) * 10
        print(f'list add with object attribute: {Δt_μs} μs')

        class B:
            def __init__(self, xx, yy, zz):
                self.xx = xx
                self.yy = yy
                self.zz = zz

            def add(self):
                xx = self.xx
                yy = self.yy
                zz = self.zz
                for i in range(12):
                    zz[i] = xx[i] + yy[i]

        b = B(xx, yy, zz)
        t = perf_counter()
        for _ in range(int(1e5)):
            b.add()
        Δt_μs = (perf_counter() - t) * 10
        print(f'list add with local var from attr: {Δt_μs} μs')

    benchmark_add()

    def benchmark_permute_preallocated():
        t = perf_counter()
        for _ in range(int(1e4)):
            torch.index_put_(y, (k,), x)
        Δt_μs = (perf_counter() - t) * 100
        print(f'tensor permute pre-allocated (index_put_): {Δt_μs} μs')

        t = perf_counter()
        for _ in range(int(1e5)):
            for i in range(12):
                yy[i] = xx[kk[i]]
        Δt_μs = (perf_counter() - t) * 10
        print(f'list yy[i] = xx[kk[i]] loop: {Δt_μs} μs')

        t = perf_counter()
        for _ in range(int(1e5)):
            for i, kkk in enumerate(kk):
                yy[i] = xx[kkk]
        Δt_μs = (perf_counter() - t) * 10
        print(f'yy[i] = xx[kkk] enumerate loop: {Δt_μs} μs')

        t = perf_counter()
        for _ in range(int(1e5)):
            for i, kkk in zip(range(12), kk):
                yy[i] = xx[kkk]
        Δt_μs = (perf_counter() - t) * 10
        print(f'yy[i] = xx[kkk] zip loop: {Δt_μs} μs')

        def permute(xx, yy, kk):
            for i in range(12):
                yy[i] = xx[kk[i]]

        t = perf_counter()
        for _ in range(int(1e5)):
            permute(xx, yy, kk)
        Δt_μs = (perf_counter() - t) * 10
        print(f'yy[i] = xx[kk[i]] w/ local var: {Δt_μs} μs')

        def permute(xx, yy, kk):
            for i, kkk in enumerate(kk):
                yy[i] = xx[kkk]

        t = perf_counter()
        for _ in range(int(1e5)):
            permute(xx, yy, kk)
        Δt_μs = (perf_counter() - t) * 10
        print(f'yy[i] = xx[kkk] enumerate w/ local var: {Δt_μs} μs')

        class A:
            def __init__(self, xx, yy, kk):
                self.xx = xx
                self.yy = yy
                self.kk = kk

            def permute(self):
                for i, kkk in enumerate(self.kk):
                    self.yy[i] = self.xx[kkk]

        a = A(xx, yy, kk)
        t = perf_counter()
        for _ in range(int(1e5)):
            a.permute()
        Δt_μs = (perf_counter() - t) * 10
        print(f'yy[i] = xx[kkk] enumerate w/ obj attr: {Δt_μs} μs')

        class B:
            def __init__(self, xx, yy, kk):
                self.xx = xx
                self.yy = yy
                self.kk = kk

            def permute(self):
                xx = self.xx
                yy = self.yy
                for i, kkk in enumerate(self.kk):
                    yy[i] = xx[kkk]

        b = B(xx, yy, kk)
        t = perf_counter()
        for _ in range(int(1e5)):
            b.permute()
        Δt_μs = (perf_counter() - t) * 10
        print(f'yy[i] = xx[kkk] enumerate w/ local var from obj attr: {Δt_μs} μs')

    benchmark_permute_preallocated()

    def benchmark_permute_new():
        t = perf_counter()
        for _ in range(int(1e5)):
            w = x[k]
        Δt_μs = (perf_counter() - t) * 10
        print(f'Tensor permute new (w[k]): {Δt_μs} μs')

        t = perf_counter()
        for _ in range(int(1e5)):
            ww = [xx[kk[i]] for i in range(12)]
        Δt_μs = (perf_counter() - t) * 10
        print(f'[xx[kk[i]] for i in range(12)]: {Δt_μs} μs')

        t = perf_counter()
        for _ in range(int(1e5)):
            ww = [xx[kkk] for kkk in kk]
        Δt_μs = (perf_counter() - t) * 10
        print(f'[xx[kkk] for kkk in kk]: {Δt_μs} μs')

        def permute(xx, kk):
            return [xx[kkk] for kkk in kk]

        t = perf_counter()
        for _ in range(int(1e5)):
            permute(xx, kk)
        Δt_μs = (perf_counter() - t) * 10
        print(f'[xx[kkk] for kkk in kk] w/ local var: {Δt_μs} μs')

        class A:
            def __init__(self, xx, kk):
                self.xx = xx
                self.kk = kk

            def permute(self):
                return [self.xx[kkk] for kkk in self.kk]

        a = A(xx, kk)
        t = perf_counter()
        for _ in range(int(1e5)):
            a.permute()
        Δt_μs = (perf_counter() - t) * 10
        print(f'[xx[kkk] for kkk in kk] w/ obj attr: {Δt_μs} μs')

        class B:
            def __init__(self, xx, kk):
                self.xx = xx
                self.kk = kk

            def permute(self):
                xx = self.xx
                return [xx[kkk] for kkk in self.kk]

        b = B(xx, kk)
        t = perf_counter()
        for _ in range(int(1e5)):
            b.permute()
        Δt_μs = (perf_counter() - t) * 10
        print(f'[xx[kkk] for kkk in kk] w/ local var from obj attr: {Δt_μs} μs')

    benchmark_permute_new()

    def benchmark_copy():
        u = y.tolist()
        t = perf_counter()
        for _ in range(int(1e5)):
            v = x.tolist()
            for i in range(12):
                u[i] = v[i]
        Δt_μs = (perf_counter() - t) * 10
        print(f'tolist + list copy: {Δt_μs} μs')

        t = perf_counter()
        for _ in range(int(1e5)):
            v = x.tolist()
        Δt_μs = (perf_counter() - t) * 10
        print(f'tensor tolist: {Δt_μs} μs')

        t = perf_counter()
        for _ in range(int(1e4)):
            for i in range(12):
                u[i] = x[i].item()
        Δt_μs = (perf_counter() - t) * 100
        print(f'directly copy (u[i] = x[i].item()): {Δt_μs} μs')

        t = perf_counter()
        for _ in range(int(1e4)):
            for i in range(12):
                u[i] = float(x[i])
        Δt_μs = (perf_counter() - t) * 100
        print(f'directly copy (u[i] = float(x[i])): {Δt_μs} μs')

    benchmark_copy()


if __name__ == '__main__':
    benchmark_list_v_tensor()
    benchmark_interface()
