from pathlib import Path

import numpy as np
import tensorrt as trt


def define_network(builder, logger, onnx_path: Path, score_thres: float, iou_thres: float, max_out: int):
    # Parse a serialized ONNX model into the TensorRT network
    # And add postprocessing (EfficientNMS_TRT)

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    successful = parser.parse(onnx_path.read_bytes())
    if not successful:
        exit(1)

    assert network.num_inputs == 1 and network.num_outputs == 1, 'network should satisfy: concat(box, obj_score, class_scores) = network(image)'

    print("ONNX I/O:")
    net_in = network.get_input(0)
    print(f'  Input {net_in.name} with shape {net_in.shape} and dtype {net_in.dtype}')
    net_out = network.get_output(0)
    print(f'  Output {net_out.name} with shape {net_out.shape} and dtype {net_out.dtype}')

    # Add postprocessing (EfficientNMS_TRT)
    # NOTE: This can be done either in
    #       the torch-to-ONNX stage or
    #       the ONNX-to-TRT stage,
    #       but not both.

    all_boxes = network.get_output(0)
    batch_size, number_boxes, number_outputs = all_boxes.shape

    # number_outputs = box(4) + obj_score(1) + class_scores(number_classes)
    number_classes = number_outputs - 5

    network.unmark_output(all_boxes)

    ld = (batch_size, number_boxes)
    unit = (1, 1, 1)
    box = network.add_slice(all_boxes, start=(0, 0, 0), shape=(*ld, 4), stride=unit)
    obj_score = network.add_slice(all_boxes, start=(0, 0, 4), shape=(*ld, 1), stride=unit)
    class_scores = network.add_slice(all_boxes, start=(0, 0, 5), shape=(*ld, number_classes), stride=unit)
    # scores = obj_score * class_scores : (bs, num_boxes, nc)
    scores = network.add_elementwise(obj_score.get_output(0), class_scores.get_output(0), trt.ElementWiseOperation.PROD)

    # EfficientNMS
    #   plugin_version: "1"
    # Parameters:
    #   score_threshold: score_thres
    #       The scalar threshold for score (low scoring boxes are removed).
    #   iou_threshold: iou_thres
    #       The scalar threshold for IOU (additional boxes that have high IOU overlap with previously selected boxes are removed).
    #   max_output_boxes: max_out
    #       The maximum number of detections to output per image.
    #   background_class: -1
    #       The label ID for the background class. If there is no background class, set it to -1.
    #   score_activation: False
    #       Set to true to apply sigmoid activation to the confidence scores during NMS operation.
    #   box_coding: 1
    #       Coding type used for boxes (and anchors if applicable), 0 = BoxCorner, 1 = BoxCenterSize.
    #       BoxCenterSize: [x, y, w, h]

    creator = trt.get_plugin_registry().get_plugin_creator("EfficientNMS_TRT", "1")
    field_collection = trt.PluginFieldCollection([
        trt.PluginField("score_threshold", np.array([score_thres], dtype=np.float32), trt.PluginFieldType.FLOAT32),
        trt.PluginField("iou_threshold", np.array([iou_thres], dtype=np.float32), trt.PluginFieldType.FLOAT32),
        trt.PluginField("max_output_boxes", np.array([max_out], dtype=np.int32), trt.PluginFieldType.INT32),
        trt.PluginField("background_class", np.array([-1], dtype=np.int32), trt.PluginFieldType.INT32),
        trt.PluginField("score_activation", np.array([0], dtype=np.int32), trt.PluginFieldType.INT32),
        trt.PluginField("box_coding", np.array([1], dtype=np.int32), trt.PluginFieldType.INT32),
    ])
    nms_plugin = creator.create_plugin("nms", field_collection)

    nms = network.add_plugin_v2([box.get_output(0), scores.get_output(0)], nms_plugin)
    for i in range(4):
        network.mark_output(nms.get_output(i))

    network.get_input(0).name = 'image'
    network.get_output(0).name = 'num'
    network.get_output(1).name = 'boxes'
    network.get_output(2).name = 'scores'
    network.get_output(3).name = 'classes'

    return network


def serialize_network(builder, network, workspace_GiB: float, enable_fp16: bool, engine_path: Path):
    config = builder.create_builder_config()
    if hasattr(config, 'set_memory_pool_limit'):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_GiB * (2 ** 30)))
    else:  # fallback for TensorRT < 8.4
        config.max_workspace_size = int(workspace_GiB * (2 ** 30))
    if enable_fp16:
        # Enable fp16 layer selection
        config.set_flag(trt.BuilderFlag.FP16)
        if not builder.platform_has_fast_fp16:
            print('The platform has NO fast native fp16')

    engine_path.write_bytes(builder.build_serialized_network(network, config))


def main():
    import argparse
    argparser = argparse.ArgumentParser()

    argparser.add_argument('-f', '--onnx_path', help='ONNX model path')
    argparser.add_argument('-v', '--verbose', action='store_true')
    argparser.add_argument('--fp16', action='store_true', help='Enable fp16 layer selection')
    argparser.add_argument('-w', '--workspace', default=1.0, type=float,
                           help='WORKSPACE is used by TensorRT to store intermediate buffers within an operation.')

    argparser.add_argument('--score_thres', default=0.35, type=float,
                           help='NMS: The scalar threshold for score (low scoring boxes are removed).')
    argparser.add_argument('--iou_thres', default=0.65, type=float,
                           help='NMS: The scalar threshold for IOU (additional boxes that have high IOU overlap with previously selected boxes are removed).')
    argparser.add_argument('--max_out', default=1, type=int,
                           help='The maximum number of detections to output per image.')

    args = argparser.parse_args()

    min_severity = trt.Logger.INFO.VERBOSE if args.verbose else trt.Logger.INFO
    logger = trt.Logger(min_severity)
    trt.init_libnvinfer_plugins(logger, namespace='')
    builder = trt.Builder(logger)

    onnx_path = Path(args.onnx_path).resolve()
    network = define_network(builder, logger, onnx_path, args.score_thres, args.iou_thres, args.max_out)
    serialize_network(builder, network, args.workspace, args.fp16, onnx_path.with_suffix('.trt'))


if __name__ == "__main__":
    main()
