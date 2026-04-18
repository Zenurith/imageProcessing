from models.base import BaseDetector

# Maps display name → detector class
MODEL_REGISTRY: dict[str, type[BaseDetector]] = {}


def register(cls: type[BaseDetector]) -> type[BaseDetector]:
    MODEL_REGISTRY[cls.name] = cls
    return cls


# Import all model modules here so their @register decorators run.
# Add a new import line here whenever a new model module is created.
from models import rtdetr      # noqa: E402, F401
from models import mask_rcnn   # noqa: E402, F401
