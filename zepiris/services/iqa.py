"""IQA for the main API: always delegates to the ML inference service ``POST /v1/iqa/assess``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import numpy as np
from fastapi import HTTPException

from zepiris.schemas.ml_inference import ImageQualityAssessmentResult

if TYPE_CHECKING:
    from zepiris.services.ml_client import MLInferenceClient


class MLInferenceIQAService:
    """Calls :meth:`MLInferenceClient.assess_image_quality` (HTTP ``/v1/iqa/assess``) for every request."""

    def __init__(self, client: MLInferenceClient) -> None:
        self._client = client

    async def assess(self, image_bgr: np.ndarray, image_b64: str) -> ImageQualityAssessmentResult:
        """Run ML IQA. ``image_bgr`` is unused (kept for call-site compatibility with decoding pipeline)."""
        _ = image_bgr
        try:
            return await self._client.assess_image_quality(image_b64)
        except httpx.HTTPStatusError as e:
            try:
                upstream = e.response.json()
            except Exception:
                upstream = e.response.text[:2000]
            status = 503 if e.response.status_code == 503 else 502
            raise HTTPException(
                status_code=status,
                detail={
                    "message": "ml_inference_assess_failed",
                    "upstream_status": e.response.status_code,
                    "upstream": upstream,
                },
            ) from e
