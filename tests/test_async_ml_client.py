"""Tests for the async MLInferenceClient and its downstream service consumers.

All network calls are mocked via AsyncMock so no running ML inference service
is required. Tests are driven with asyncio.run() — no pytest-asyncio dependency.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import numpy as np
import pytest

from zepiris.services.embedding import MLInferenceEmbeddingService, StubFaceEmbeddingService
from zepiris.services.iqa import MLInferenceIQAService
from zepiris.services.ml_client import MLInferenceClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    return asyncio.run(coro)


def _mock_client(base_url: str = "http://test") -> MLInferenceClient:
    """Return an MLInferenceClient whose inner httpx.AsyncClient is never used."""
    c = MLInferenceClient.__new__(MLInferenceClient)
    c.base_url = base_url
    c.client = AsyncMock(spec=httpx.AsyncClient)
    return c


def _json_response(data: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.json.return_value = data
    resp.raise_for_status = MagicMock()
    resp.status_code = status_code
    return resp


# ---------------------------------------------------------------------------
# MLInferenceClient — unit tests
# ---------------------------------------------------------------------------


class TestMLInferenceClientAsync:
    def test_prepare_image_json(self) -> None:
        c = _mock_client()
        assert c._prepare_image_json("abc") == {"image_b64": "abc"}

    def test_uses_async_client(self) -> None:
        c = MLInferenceClient("http://localhost:8001")
        assert isinstance(c.client, httpx.AsyncClient)
        _run(c.aclose())

    def test_base_url_strips_trailing_slash(self) -> None:
        c = MLInferenceClient("http://localhost:8001/")
        assert c.base_url == "http://localhost:8001"
        _run(c.aclose())

    def test_detect_nsfw_success(self) -> None:
        async def _inner():
            c = _mock_client()
            c.client.post = AsyncMock(
                return_value=_json_response({"is_safe": True, "probability": 0.99})
            )
            result = await c.detect_nsfw("b64img")
            assert result.is_safe is True
            assert result.probability == pytest.approx(0.99)
            c.client.post.assert_awaited_once()

        _run(_inner())

    def test_detect_spoof_success(self) -> None:
        async def _inner():
            c = _mock_client()
            c.client.post = AsyncMock(
                return_value=_json_response({"is_live": True, "probability": 0.95})
            )
            result = await c.detect_spoof("b64img")
            assert result.is_live is True

        _run(_inner())

    def test_detect_blur_success(self) -> None:
        async def _inner():
            c = _mock_client()
            c.client.post = AsyncMock(
                return_value=_json_response({"is_sharp": True, "probability": 0.88})
            )
            result = await c.detect_blur("b64img")
            assert result.is_sharp is True

        _run(_inner())

    def test_embed_face_success(self) -> None:
        async def _inner():
            c = _mock_client()
            c.client.post = AsyncMock(
                return_value=_json_response(
                    {
                        "face_detected": True,
                        "embedding": [0.1] * 512,
                        "embedding_dim": 512,
                    }
                )
            )
            result = await c.embed_face("b64img")
            assert result.face_detected is True
            assert result.embedding_dim == 512

        _run(_inner())

    def test_assess_image_quality_success(self) -> None:
        async def _inner():
            c = _mock_client()
            payload = {
                "passed": True,
                "nsfw": {"is_safe": True, "probability": 0.99},
                "spoof": {"is_live": True, "probability": 0.98},
                "blur": {"is_sharp": True, "probability": 0.97},
            }
            c.client.post = AsyncMock(return_value=_json_response(payload))
            result = await c.assess_image_quality("b64img")
            assert result.passed is True

        _run(_inner())

    def test_healthz_success(self) -> None:
        async def _inner():
            c = _mock_client()
            c.client.get = AsyncMock(return_value=_json_response({"status": "ok"}))
            result = await c.healthz()
            assert result == {"status": "ok"}

        _run(_inner())

    def test_aclose_is_awaitable(self) -> None:
        c = _mock_client()
        _run(c.aclose())  # must not raise

    def test_async_context_manager(self) -> None:
        async def _inner():
            async with _mock_client() as c:
                assert isinstance(c, MLInferenceClient)

        _run(_inner())

    def test_http_error_propagates(self) -> None:
        async def _inner():
            c = _mock_client()
            bad_resp = MagicMock(spec=httpx.Response)
            bad_resp.status_code = 503
            bad_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                "upstream down", request=MagicMock(), response=bad_resp
            )
            c.client.post = AsyncMock(return_value=bad_resp)
            with pytest.raises(httpx.HTTPStatusError):
                await c.detect_nsfw("b64img")

        _run(_inner())


# ---------------------------------------------------------------------------
# MLInferenceIQAService — async assess()
# ---------------------------------------------------------------------------


class TestMLInferenceIQAServiceAsync:
    def test_assess_delegates_to_client(self) -> None:
        async def _inner():
            c = _mock_client()
            payload = {
                "passed": True,
                "nsfw": {"is_safe": True, "probability": 0.99},
                "spoof": {"is_live": True, "probability": 0.98},
                "blur": {"is_sharp": True, "probability": 0.97},
            }
            c.client.post = AsyncMock(return_value=_json_response(payload))
            svc = MLInferenceIQAService(c)
            result = await svc.assess(np.zeros((10, 10, 3), dtype=np.uint8), "b64")
            assert result.passed is True

        _run(_inner())

    def test_assess_raises_http_exception_on_503(self) -> None:
        from fastapi import HTTPException

        async def _inner():
            c = _mock_client()
            bad_resp = MagicMock(spec=httpx.Response)
            bad_resp.status_code = 503
            bad_resp.json.return_value = {"detail": "service unavailable"}
            bad_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                "503", request=MagicMock(), response=bad_resp
            )
            c.client.post = AsyncMock(return_value=bad_resp)
            svc = MLInferenceIQAService(c)
            with pytest.raises(HTTPException) as exc_info:
                await svc.assess(np.zeros((10, 10, 3), dtype=np.uint8), "b64")
            assert exc_info.value.status_code == 503

        _run(_inner())


# ---------------------------------------------------------------------------
# FaceEmbeddingProvider — async embed()
# ---------------------------------------------------------------------------


class TestEmbeddingServicesAsync:
    def test_stub_embed_returns_valid_result(self) -> None:
        async def _inner():
            svc = StubFaceEmbeddingService(dim=512)
            image = np.zeros((112, 112, 3), dtype=np.uint8)
            result = await svc.embed(image)
            assert result.face_detected is True
            assert result.embedding_dim == 512
            assert len(result.embedding) == 512

        _run(_inner())

    def test_stub_embed_is_deterministic(self) -> None:
        async def _inner():
            svc = StubFaceEmbeddingService(dim=512)
            image = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
            r1 = await svc.embed(image)
            r2 = await svc.embed(image)
            assert r1.embedding == r2.embedding

        _run(_inner())

    def test_stub_embed_is_l2_normalized(self) -> None:
        async def _inner():
            svc = StubFaceEmbeddingService(dim=512)
            image = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
            result = await svc.embed(image)
            norm = np.linalg.norm(result.embedding)
            assert abs(norm - 1.0) < 1e-5

        _run(_inner())

    def test_ml_inference_embedding_calls_client(self) -> None:
        async def _inner():
            c = _mock_client()
            c.client.post = AsyncMock(
                return_value=_json_response(
                    {"face_detected": True, "embedding": [0.0] * 512, "embedding_dim": 512}
                )
            )
            svc = MLInferenceEmbeddingService(c)
            image = np.zeros((112, 112, 3), dtype=np.uint8)
            result = await svc.embed(image)
            assert result.face_detected is True
            c.client.post.assert_awaited_once()

        _run(_inner())

    def test_ml_inference_embedding_encode_failure_raises(self) -> None:
        from zepiris.exceptions import ImageEncodeError

        async def _inner():
            c = _mock_client()
            svc = MLInferenceEmbeddingService(c)
            image = np.zeros((112, 112, 3), dtype=np.uint8)
            with patch("cv2.imencode", return_value=(False, None)), pytest.raises(ImageEncodeError):
                await svc.embed(image)

        _run(_inner())
