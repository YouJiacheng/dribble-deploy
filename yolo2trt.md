First convert Pytorch model to ONNX
```bash
python yolov7/export.py --weights best.pt --grid --simplify --img-size 480 480
```
Do NOT specify `--end2end`, we handle this in `onnx2trt.py`.
Thus there is no need to specify `--conf-thres`,`--iou-thres` and `--topk-all` for NMS.

Then convert ONNX to TensorRT
```bash
python onnx2trt.py -f best.onnx -v --fp16 --score_thres 0.35 --iou_thres 0.65 --max_out 1
```
