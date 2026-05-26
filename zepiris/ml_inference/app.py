"""FastAPI application for ML inference microservice.

Run standalone:
    uvicorn zepiris.ml_inference.app:app --host 0.0.0.0 --port 8001

Or via the entry point:
    zepiris-ml-inference-api
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import FastAPI
from pydantic_settings import BaseSettings, SettingsConfigDict

from zepiris.ml_inference.adaface_embedding import AdaFaceEmbeddingService
from zepiris.ml_inference.blur_detection import BlurDetectionService
from zepiris.ml_inference.face_embedding import FaceEmbeddingService
from zepiris.ml_inference.image_quality_assessment import (
    ImageQualityAssessmentService,
)
from zepiris.ml_inference.nsfw_detection import NSFWDetectionService
from zepiris.ml_inference.spoof_detection import SpoofDetectionService
from zepiris.version import __version__ as package_version

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Settings (env prefix ML_SERVICE_ so the inference container has its own env)
# ---------------------------------------------------------------------------
class MLServiceSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ML_SERVICE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "0.0.0.0"
    port: int = 8001

    ml_device: str = "cpu"

    nsfw_model_source: str = "auto"
    nsfw_hf_repo_id: str = ""
    nsfw_hf_model_file: str = "nsfw_model.pth"
    nsfw_local_model_path: str = "/app/models/nsfw_model.pth"
    nsfw_threshold: float = 0.5

    spoof_model_source: str = "auto"
    spoof_hf_repo_id: str = ""
    spoof_hf_model_file: str = "spoof_model.pth"
    spoof_local_model_path: str = "/app/models/spoof_model.pth"
    spoof_threshold: float = 0.5

    blur_model_source: str = "auto"
    blur_hf_repo_id: str = ""
    blur_hf_model_file: str = "blur_model.pth"
    blur_local_model_path: str = "/app/models/blur_model.pth"
    blur_threshold: float = 0.5

    face_embedding_dim: int = 512
    face_detection_width: int = 640
    face_detection_height: int = 640
    face_area_threshold: float = 0.01

    # Embedding backend: "insightface" (buffalo_l/ArcFace) or "adaface" (IR-50, CVPR 2022)
    # AdaFace significantly outperforms ArcFace on low-quality images (blur, low-light, occlusion)
    face_embedding_backend: str = "insightface"

    adaface_model_source: str = "auto"
    adaface_hf_repo_id: str = ""
    adaface_hf_model_file: str = "adaface_ir50.ckpt"
    adaface_local_model_path: str = "/app/models/adaface_ir50.ckpt"


@lru_cache
def get_ml_settings() -> MLServiceSettings:
    return MLServiceSettings()


# ---------------------------------------------------------------------------
# Lifespan — instantiate every model service once on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_ml_settings()
    device = s.ml_device

    logger.info("Loading ML models on device=%s …", device)
    failed: list[str] = []

    try:
        app.state.nsfw_service = NSFWDetectionService(
            huggingface_repo_id=s.nsfw_hf_repo_id,
            huggingface_model_file=s.nsfw_hf_model_file,
            local_model_path=s.nsfw_local_model_path or None,
            model_source=s.nsfw_model_source,
            nsfw_threshold=s.nsfw_threshold,
            device=device,
        )
        app.state.nsfw_service.load_model()
    except Exception:
        logger.exception("Failed to load NSFWDetectionService")
        app.state.nsfw_service = None
        failed.append("nsfw")

    try:
        app.state.spoof_service = SpoofDetectionService(
            huggingface_repo_id=s.spoof_hf_repo_id,
            huggingface_model_file=s.spoof_hf_model_file,
            local_model_path=s.spoof_local_model_path or None,
            model_source=s.spoof_model_source,
            spoof_threshold=s.spoof_threshold,
            device=device,
        )
        app.state.spoof_service.load_model()
    except Exception:
        logger.exception("Failed to load SpoofDetectionService")
        app.state.spoof_service = None
        failed.append("spoof")

    try:
        app.state.blur_service = BlurDetectionService(
            huggingface_repo_id=s.blur_hf_repo_id,
            huggingface_model_file=s.blur_hf_model_file,
            local_model_path=s.blur_local_model_path or None,
            model_source=s.blur_model_source,
            blur_threshold=s.blur_threshold,
            device=device,
        )
        app.state.blur_service.load_model()
    except Exception:
        logger.exception("Failed to load BlurDetectionService")
        app.state.blur_service = None
        failed.append("blur")

    try:
        if s.face_embedding_backend == "adaface":
            logger.info("Face embedding backend: AdaFace IR-50 (CVPR 2022)")
            app.state.face_embedding_service = AdaFaceEmbeddingService(
                huggingface_repo_id=s.adaface_hf_repo_id,
                huggingface_model_file=s.adaface_hf_model_file,
                local_model_path=s.adaface_local_model_path or None,
                model_source=s.adaface_model_source,
                embedding_dim=s.face_embedding_dim,
                detection_size=(s.face_detection_width, s.face_detection_height),
                facial_area_threshold=s.face_area_threshold,
                device=device,
            )
        else:
            logger.info("Face embedding backend: InsightFace buffalo_l (ArcFace)")
            app.state.face_embedding_service = FaceEmbeddingService(
                embedding_dim=s.face_embedding_dim,
                detection_size=(s.face_detection_width, s.face_detection_height),
                facial_area_threshold=s.face_area_threshold,
                device=device,
            )
        app.state.face_embedding_service.load_model()
    except Exception:
        logger.exception(
            "Failed to load face embedding service (backend=%s)", s.face_embedding_backend
        )
        app.state.face_embedding_service = None
        failed.append("face_embedding")

    nsfw = app.state.nsfw_service
    spoof = app.state.spoof_service
    blur = app.state.blur_service
    if nsfw is not None and spoof is not None and blur is not None:
        app.state.iqa_service = ImageQualityAssessmentService(
            nsfw_service=nsfw,
            spoof_service=spoof,
            blur_service=blur,
        )
    else:
        app.state.iqa_service = None
        logger.error(
            "IQA disabled: need all three models loaded (nsfw=%s spoof=%s blur=%s)",
            nsfw is None,
            spoof is None,
            blur is None,
        )

    if failed:
        logger.warning("ML service started with degraded models: %s", ", ".join(failed))
    else:
        logger.info("All ML models loaded successfully.")
    yield
    logger.info("ML inference service shutting down.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    from zepiris.ml_inference.routes import router

    application = FastAPI(
        title="ZepIris ML Inference Service",
        version=package_version,
        lifespan=lifespan,
    )
    application.include_router(router)
    return application


app = create_app()


def run() -> None:
    """Entry point for ``zepiris-ml-inference-api`` console script."""
    import uvicorn

    s = get_ml_settings()
    uvicorn.run(
        "zepiris.ml_inference.app:app",
        host=s.host,
        port=s.port,
        reload=False,
    )
