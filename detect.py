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
        self.image_shape, self.image_dtype = self.get_shape_dtype('image')
        print(f'Input image shape: {self.image_shape}, dtype: {self.image_dtype}')
        self.context = self.engine.create_execution_context()

    def get_shape_dtype(self, name: str):
        e = self.engine
        if hasattr(e, 'get_tensor_shape') and hasattr(e, 'get_tensor_dtype'):
            shape = e.get_tensor_shape(name)
            dtype = e.get_tensor_dtype(name)
        else:  # fallback for TensorRT < 8.5
            shape = e.get_binding_shape(name)
            dtype = e.get_binding_dtype(name)
        return tuple(shape), torch_dtype_from_trt[dtype]

    def allocate_tensor(self, name: str):
        shape, dtype = self.get_shape_dtype(name)
        return torch.empty(shape, dtype=dtype, device='cuda')

    def detect(self, image: torch.Tensor):
        bindings = [None for _ in range(5)]
        assert image.shape == self.image_shape
        image = image.to(device='cuda', dtype=self.image_dtype, non_blocking=True)
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
        image = torch.ones(self.image_shape, dtype=self.image_dtype, device='cpu')
        for _ in range(5):  # warmup
            _ = self.detect(image)

        t = time.perf_counter()
        for _ in range(100):
            _ = self.detect(image)
        print(f'{100 / (time.perf_counter() - t)} FPS')


if __name__ == '__main__':
    import cv2
    import torchvision.transforms as T
    import zmq
    cap = cv2.VideoCapture(
        'udpsrc address=192.168.123.15 port=9201 '
        '! application/x-rtp,media=video,encoding-name=H264 '
        '! rtph264depay ! h264parse ! omxh264dec ! videoconvert ! appsink'
    )

    ctx: 'zmq.Context[zmq.Socket]' = zmq.Context.instance()
    socket = ctx.socket(zmq.DEALER)
    socket.set(zmq.CONFLATE, 1)
    socket.bind('tcp://127.0.0.1:5555')

    detector = Detector(Path('best.trt'))
    detector.get_fps()

    while True:
        rv, image = cap.read()
        if not rv:
            continue
        image = T.ToTensor()(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image = T.Pad((8, 40))(image)
        num, boxes, scores, classes = detector.detect(image[None])
        score = scores[0, 0].item()
        box_corner = boxes[0, 0].tolist()
        socket.send_pyobj((score, box_corner))
