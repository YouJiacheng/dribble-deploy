from pathlib import Path

import tensorrt as trt
import torch

torch_dtype_from_trt = {
    trt.int8: torch.int8,
    trt.int32: torch.int32,
    trt.float16: torch.float16,
    trt.float32: torch.float32,
}


class Detector:
    def __init__(self, engine_path: Path):
        logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(logger)
        trt.init_libnvinfer_plugins(logger, '')
        self.engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        assert tuple(self.engine) == ('image', 'num', 'boxes', 'scores', 'classes')
        self.image_size = tuple(self.engine.get_tensor_shape('image'))[2:]
        print(self.image_size)
        self.context = self.engine.create_execution_context()

    def allocate_tensor(self, name: str):
        shape = tuple(self.engine.get_tensor_shape(name))
        dtype = torch_dtype_from_trt[self.engine.get_tensor_dtype(name)]
        return torch.empty(shape, dtype=dtype, device='cuda')

    def detect(self, image: torch.Tensor):
        bindings = [None for _ in range(5)]
        assert image.shape[2:] == self.image_size
        image_dtype = torch_dtype_from_trt[self.engine.get_tensor_dtype('image')]
        image = image.to(device='cuda', dtype=image_dtype, non_blocking=True)
        bindings[0] = image.data_ptr()
        num = self.allocate_tensor('num')
        bindings[1] = num.data_ptr()
        boxes = self.allocate_tensor('boxes')
        bindings[2] = boxes.data_ptr()
        scores = self.allocate_tensor('scores')
        bindings[3] = scores.data_ptr()
        classes = self.allocate_tensor('classes')
        bindings[4] = classes.data_ptr()
        self.context.execute_async_v2(bindings=bindings, stream_handle=torch.cuda.current_stream().cuda_stream)
        return num, boxes, scores, classes

    def get_fps(self):
        import time
        image = torch.ones((1, 3, *self.image_size), dtype=torch.float32, device='cpu')
        for _ in range(5):  # warmup
            _ = self.detect(image)

        t0 = time.perf_counter()
        for _ in range(100):
            _ = self.detect(image)
        print(100 / (time.perf_counter() - t0), 'FPS')


if __name__ == '__main__':
    detector = Detector(Path('best.trt'))
    detector.get_fps()
