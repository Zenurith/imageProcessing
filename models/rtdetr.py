import numpy as np
from models.base import BaseDetector
from models.registry import register

WEIGHT_PATH = "runs/rtdetr/train/weights/best.pt"


@register
class RTDETRDetector(BaseDetector):
    name = "RT-DETR"
    description = "Real-Time Detection Transformer trained on TACO dataset. Outputs bounding boxes only."

    def __init__(self):
        self._model = None

    def load(self) -> None:
        from ultralytics import RTDETR
        self._model = RTDETR(WEIGHT_PATH)

    def predict(self, image: np.ndarray, conf_threshold: float) -> dict:
        results = self._model.predict(image, conf=conf_threshold, verbose=False)
        boxes_obj = results[0].boxes

        boxes, labels, scores = [], [], []
        for box in boxes_obj:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes.append([x1, y1, x2, y2])
            scores.append(float(box.conf[0]))
            labels.append(int(box.cls[0]))

        return {"boxes": boxes, "labels": labels, "scores": scores, "masks": None}
