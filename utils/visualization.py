import cv2
import numpy as np

CLASS_NAMES = ["Bottle", "Cigarette", "Foam", "Glass", "Metal", "Other", "Paper", "Plastic", "Unlabeled"]

# One distinct BGR color per class
CLASS_COLORS = [
    (0, 114, 189),   # Bottle     - blue
    (217, 83, 25),   # Cigarette  - orange
    (237, 177, 32),  # Foam       - yellow
    (126, 47, 142),  # Glass      - purple
    (119, 172, 48),  # Metal      - green
    (77, 190, 238),  # Other      - cyan
    (162, 20, 47),   # Paper      - dark red
    (255, 153, 0),   # Plastic    - amber
    (128, 128, 128), # Unlabeled  - grey
]


def draw_results(
    image: np.ndarray,
    boxes: list,
    labels: list,
    scores: list,
    masks: list | None = None,
) -> np.ndarray:
    """Draw detection boxes, optional masks and labels on a copy of the RGB image."""
    out = image.copy()
    overlay = out.copy()

    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        label = max(0, min(label, len(CLASS_NAMES) - 1))
        color_bgr = CLASS_COLORS[label]
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])

        if masks is not None and i < len(masks):
            mask = masks[i]
            for c in range(3):
                overlay[:, :, c] = np.where(mask == 1, color_rgb[c], overlay[:, :, c])

        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(out, (x1, y1), (x2, y2), color_rgb, 2)

        text = f"{CLASS_NAMES[label]} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color_rgb, -1)
        cv2.putText(out, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    if masks is not None:
        out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)

    return out
