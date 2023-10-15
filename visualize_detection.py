from pathlib import Path

import torchvision.transforms as T
from PIL import Image, ImageDraw

from detect import Detector


def draw_boxes(image, boxes, scores):
    # boxes in BoxCorner format
    draw = ImageDraw.Draw(image)
    for box, score in zip(boxes, scores):
        draw.rectangle(box, outline='red', width=2)
        draw.text((box[0], box[1] - 10), f'{score * 100:.1f}', fill='red')


detector = Detector(Path('best.trt'))

image = Image.open('test.png')
image = T.ToTensor()(image)
image = T.Pad((8, 40))(image)
num, boxes, scores, classes = detector.detect(image[None])

boxes = boxes[0].cpu().numpy()
scores = scores[0].cpu().numpy()
image: Image.Image = T.ToPILImage()(image)
draw_boxes(image, boxes, scores)
image.save('result.png')
