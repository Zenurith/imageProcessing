from abc import ABC, abstractmethod
import numpy as np


class BaseDetector(ABC):
    name: str = ""
    description: str = ""

    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory."""
        ...

    @abstractmethod
    def predict(self, image: np.ndarray, conf_threshold: float) -> dict:
        """
        Run inference on an RGB image (H, W, 3) uint8.

        Returns a dict with keys:
          boxes   - list of [x1, y1, x2, y2] in original pixel coords
          labels  - list of int class indices (0-indexed into CLASS_NAMES)
          scores  - list of float confidence values
          masks   - list of binary np.ndarray (H, W) uint8, or None if unsupported
        """
        ...
