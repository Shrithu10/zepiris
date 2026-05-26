"""PyTorch model architecture definitions."""

from zepiris.ml_inference.models.adaface import AdaFaceIR50
from zepiris.ml_inference.models.blur import ResNetBlurDetector
from zepiris.ml_inference.models.spoof import MobileNetV3LSpoof

__all__ = [
    "AdaFaceIR50",
    "ResNetBlurDetector",
    "MobileNetV3LSpoof",
]
