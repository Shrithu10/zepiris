"""Async HTTP client for calling the ML inference microservice.

The main application uses this client to send requests to the separate
ML inference container (running on port 8001 by default).

Example:
    import base64
    import cv2

    async with MLInferenceClient("http://localhost:8001") as client:
        image_bgr = cv2.imread("photo.jpg")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_bytes = cv2.imencode(".jpg", image_rgb)[1].tobytes()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        nsfw_result = await client.detect_nsfw(image_b64)
"""

from __future__ import annotations

from typing import Any

import httpx

from zepiris.schemas.ml_inference import (
    BlurDetectionResult,
    FaceEmbeddingResult,
    ImageQualityAssessmentResult,
    NSFWDetectionResult,
    SpoofDetectionResult,
)


def _upstream_error_detail(response: httpx.Response) -> dict[str, Any]:
    try:
        upstream: Any = response.json()
    except Exception:
        upstream = response.text[:2000]
    return {
        "message": "ml_inference_request_failed",
        "upstream_status": response.status_code,
        "upstream": upstream,
    }


class MLInferenceClient:
    """Async HTTP client for calling the remote ML inference service.

    All network methods are coroutines — await them inside async FastAPI
    route handlers or lifespan hooks. Use as an async context manager or
    call ``aclose()`` explicitly during application shutdown.
    """

    def __init__(self, base_url: str) -> None:
        """Initialize ML inference client.

        Args:
            base_url: Base URL of the ML inference service, e.g. "http://localhost:8001"
        """
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(base_url=self.base_url)

    def _prepare_image_json(self, image_b64: str) -> dict:
        """Prepare JSON payload for an image in base64 format.

        Args:
            image_b64: Image as base64-encoded string

        Returns:
            dict: JSON payload with base64-encoded image
        """
        return {"image_b64": image_b64}

    async def detect_nsfw(self, image_b64: str) -> NSFWDetectionResult:
        """Run NSFW detection on an image.

        Args:
            image_b64: Image as base64-encoded string

        Returns:
            NSFWDetectionResult: Detection result
        """
        payload = self._prepare_image_json(image_b64)
        response = await self.client.post("/v1/iqa/nsfw_check", json=payload)
        response.raise_for_status()
        return NSFWDetectionResult(**response.json())

    async def detect_spoof(self, image_b64: str) -> SpoofDetectionResult:
        """Run spoof detection on an image.

        Args:
            image_b64: Image as base64-encoded string

        Returns:
            SpoofDetectionResult: Detection result
        """
        payload = self._prepare_image_json(image_b64)
        response = await self.client.post("/v1/iqa/spoof_check", json=payload)
        response.raise_for_status()
        return SpoofDetectionResult(**response.json())

    async def detect_blur(self, image_b64: str) -> BlurDetectionResult:
        """Run blur detection on an image.

        Args:
            image_b64: Image as base64-encoded string

        Returns:
            BlurDetectionResult: Detection result
        """
        payload = self._prepare_image_json(image_b64)
        response = await self.client.post("/v1/iqa/blur_check", json=payload)
        response.raise_for_status()
        return BlurDetectionResult(**response.json())

    async def embed_face(self, image_b64: str) -> FaceEmbeddingResult:
        """Generate face embedding from an image.

        Args:
            image_b64: Image as base64-encoded string

        Returns:
            FaceEmbeddingResult: Embedding result
        """
        payload = self._prepare_image_json(image_b64)
        response = await self.client.post("/v1/face/embed", json=payload)
        response.raise_for_status()
        return FaceEmbeddingResult(**response.json())

    async def assess_image_quality(self, image_b64: str) -> ImageQualityAssessmentResult:
        """Run combined image quality assessment (NSFW + spoof + blur).

        Args:
            image_b64: Image as base64-encoded string

        Returns:
            ImageQualityAssessmentResult: Combined assessment result
        """
        payload = self._prepare_image_json(image_b64)
        response = await self.client.post("/v1/iqa/assess", json=payload)
        response.raise_for_status()
        return ImageQualityAssessmentResult(**response.json())

    async def healthz(self) -> dict[str, str]:
        """Check service health.

        Returns:
            dict: Health status response
        """
        response = await self.client.get("/healthz")
        response.raise_for_status()
        return response.json()

    async def aclose(self) -> None:
        """Close the underlying async HTTP client."""
        await self.client.aclose()

    async def __aenter__(self) -> MLInferenceClient:
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args) -> None:
        """Async context manager exit."""
        await self.aclose()
