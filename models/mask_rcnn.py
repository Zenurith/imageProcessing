import numpy as np
import cv2
import torch
from models.base import BaseDetector
from models.registry import register

WEIGHT_PATH = "runs/mask_rcnn/best_mask_rcnn.pth"
MAX_SIZE = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@register
class MaskRCNNDetector(BaseDetector):
    name = "Mask R-CNN"
    description = "Instance segmentation model trained on TACO dataset. Outputs bounding boxes and pixel masks."

    def __init__(self):
        self._model = None

    def load(self) -> None:
        from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
        model = maskrcnn_resnet50_fpn_v2(weights=None, num_classes=10)  # background + 9 classes
        ckpt = torch.load(WEIGHT_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(DEVICE)
        model.eval()
        self._model = model

    def predict(self, image: np.ndarray, conf_threshold: float) -> dict:
        orig_h, orig_w = image.shape[:2]
        scale = min(MAX_SIZE / max(orig_h, orig_w), 1.0)

        if scale < 1.0:
            new_h, new_w = int(orig_h * scale), int(orig_w * scale)
            resized = cv2.resize(image, (new_w, new_h))
        else:
            resized = image

        tensor = torch.as_tensor(resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
        tensor = tensor.to(DEVICE)

        with torch.no_grad():
            pred = self._model([tensor])[0]

        keep = pred["scores"] >= conf_threshold
        raw_boxes = pred["boxes"][keep].cpu().numpy()
        raw_labels = pred["labels"][keep].cpu().numpy()
        raw_scores = pred["scores"][keep].cpu().numpy()
        raw_masks = pred["masks"][keep].cpu().numpy()

        boxes, labels, scores, masks = [], [], [], []
        for i in range(len(raw_boxes)):
            x1, y1, x2, y2 = [v / scale for v in raw_boxes[i]]
            boxes.append([x1, y1, x2, y2])
            labels.append(int(raw_labels[i]) - 1)  # 1-indexed → 0-indexed
            scores.append(float(raw_scores[i]))

            binary = (raw_masks[i, 0] > 0.5).astype(np.uint8)
            if scale < 1.0:
                binary = cv2.resize(binary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            masks.append(binary)

        return {"boxes": boxes, "labels": labels, "scores": scores, "masks": masks if masks else None}
