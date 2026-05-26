"""FastAPI dependency injection for ML inference routes."""

from typing import Annotated, TypeVar

from fastapi import Depends, HTTPException, Request

from zepiris.ml_inference.adaface_embedding import AdaFaceEmbeddingService
from zepiris.ml_inference.blur_detection import BlurDetectionService
from zepiris.ml_inference.face_embedding import FaceEmbeddingService
from zepiris.ml_inference.image_quality_assessment import (
    ImageQualityAssessmentService,
)
from zepiris.ml_inference.nsfw_detection import NSFWDetectionService
from zepiris.ml_inference.spoof_detection import SpoofDetectionService

_MODEL_HINT = (
    "Model failed to load at startup. Check ML container logs: "
    "docker logs zepiris-ml-inference. "
    "For IQA, all three .pth files must load (paths: ML_SERVICE_*_LOCAL_MODEL_PATH). "
    "Rebuild the ML image if weights are missing: make build-ml."
)

_T = TypeVar("_T")


def _require(name: str, svc: _T | None, *, hint: str = _MODEL_HINT) -> _T:
    if svc is None:
        raise HTTPException(
            status_code=503,
            detail={"message": f"{name}_unavailable", "hint": hint},
        )
    return svc


def _nsfw(request: Request) -> NSFWDetectionService:
    return _require("nsfw", request.app.state.nsfw_service)


def _spoof(request: Request) -> SpoofDetectionService:
    return _require("spoof", request.app.state.spoof_service)


def _blur(request: Request) -> BlurDetectionService:
    return _require("blur", request.app.state.blur_service)


def _face_embedding(request: Request) -> FaceEmbeddingService | AdaFaceEmbeddingService:
    return _require(
        "face_embedding",
        request.app.state.face_embedding_service,
        hint=(
            "Face embedding service failed to load. "
            "Check ML_SERVICE_FACE_EMBEDDING_BACKEND and the corresponding model path. "
            "See ML container startup logs."
        ),
    )


def _iqa(request: Request) -> ImageQualityAssessmentService:
    svc: ImageQualityAssessmentService | None = request.app.state.iqa_service
    if svc is None:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "iqa_unavailable",
                "hint": _MODEL_HINT,
            },
        )
    return svc


NSFWDep = Annotated[NSFWDetectionService, Depends(_nsfw)]
SpoofDep = Annotated[SpoofDetectionService, Depends(_spoof)]
BlurDep = Annotated[BlurDetectionService, Depends(_blur)]
FaceEmbeddingDep = Annotated[
    FaceEmbeddingService | AdaFaceEmbeddingService, Depends(_face_embedding)
]
IQADep = Annotated[ImageQualityAssessmentService, Depends(_iqa)]
