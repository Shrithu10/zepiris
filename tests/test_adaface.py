"""Tests for AdaFace IR-50 backbone and AdaFaceEmbeddingService.

Architecture tests run without a checkpoint (random weights are fine for
shape/normalization checks). Service tests mock the detector and backbone
so no InsightFace models or GPU are required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed; skipping AdaFace tests")

from zepiris.ml_inference.adaface_embedding import AdaFaceEmbeddingService  # noqa: E402
from zepiris.ml_inference.models.adaface import AdaFaceIR50  # noqa: E402
from zepiris.schemas.ml_inference import FaceEmbeddingResult  # noqa: E402

# ---------------------------------------------------------------------------
# AdaFaceIR50 — architecture tests (no checkpoint, random weights)
# ---------------------------------------------------------------------------


class TestAdaFaceIR50Architecture:
    """Shape, normalization, and determinism checks — no model weights needed."""

    @pytest.fixture()
    def model(self) -> AdaFaceIR50:
        m = AdaFaceIR50()
        m.eval()
        return m

    def test_instantiation(self, model: AdaFaceIR50) -> None:
        assert model is not None

    def test_single_image_output_shape(self, model: AdaFaceIR50) -> None:
        x = torch.randn(1, 3, 112, 112)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 512)

    def test_batch_output_shape(self, model: AdaFaceIR50) -> None:
        x = torch.randn(4, 3, 112, 112)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 512)

    def test_output_is_l2_normalized(self, model: AdaFaceIR50) -> None:
        x = torch.randn(3, 3, 112, 112)
        with torch.no_grad():
            out = model(x)
        norms = out.norm(p=2, dim=1)
        assert torch.allclose(norms, torch.ones(3), atol=1e-5)

    def test_same_input_produces_same_output(self, model: AdaFaceIR50) -> None:
        x = torch.randn(1, 3, 112, 112)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)

    def test_different_inputs_produce_different_outputs(self, model: AdaFaceIR50) -> None:
        x1 = torch.randn(1, 3, 112, 112)
        x2 = torch.randn(1, 3, 112, 112)
        with torch.no_grad():
            out1 = model(x1)
            out2 = model(x2)
        assert not torch.allclose(out1, out2)

    def test_input_range_invariance(self, model: AdaFaceIR50) -> None:
        """AdaFace expects [-1, 1] input — verify normalization layer absorbs scale."""
        x_norm = torch.clamp(torch.randn(1, 3, 112, 112), -1, 1)
        with torch.no_grad():
            out = model(x_norm)
        norms = out.norm(p=2, dim=1)
        assert torch.allclose(norms, torch.ones(1), atol=1e-5)


# ---------------------------------------------------------------------------
# AdaFaceEmbeddingService — unit tests (mocked detector + backbone)
# ---------------------------------------------------------------------------


class TestAdaFaceEmbeddingService:
    """Service-level tests without real InsightFace models or checkpoints."""

    @pytest.fixture()
    def service(self) -> AdaFaceEmbeddingService:
        return AdaFaceEmbeddingService(device="cpu")

    def test_default_embedding_dim(self, service: AdaFaceEmbeddingService) -> None:
        assert service._embedding_dim == 512

    def test_default_facial_area_threshold(self, service: AdaFaceEmbeddingService) -> None:
        assert service._facial_area_threshold == pytest.approx(0.01)

    def test_backbone_not_loaded_at_init(self, service: AdaFaceEmbeddingService) -> None:
        assert service._backbone is None

    def test_detector_not_loaded_at_init(self, service: AdaFaceEmbeddingService) -> None:
        assert service._detector is None

    def test_postprocess_no_face_returns_zero_embedding(
        self, service: AdaFaceEmbeddingService
    ) -> None:
        zero = np.zeros(512, dtype=np.float32)
        result = service.postprocess((zero, False))
        assert isinstance(result, FaceEmbeddingResult)
        assert result.face_detected is False
        assert result.embedding_dim == 512
        assert len(result.embedding) == 512
        assert all(v == 0.0 for v in result.embedding)

    def test_postprocess_with_face_returns_normalized_embedding(
        self, service: AdaFaceEmbeddingService
    ) -> None:
        emb = np.random.randn(512).astype(np.float32)
        result = service.postprocess((emb, True))
        assert result.face_detected is True
        assert result.embedding_dim == 512
        norm = np.linalg.norm(result.embedding)
        assert abs(norm - 1.0) < 1e-5

    def test_postprocess_normalizes_unnormalized_embedding(
        self, service: AdaFaceEmbeddingService
    ) -> None:
        emb = np.ones(512, dtype=np.float32) * 3.0
        result = service.postprocess((emb, True))
        norm = np.linalg.norm(result.embedding)
        assert abs(norm - 1.0) < 1e-5

    def test_postprocess_returns_float_list(self, service: AdaFaceEmbeddingService) -> None:
        emb = np.random.randn(512).astype(np.float32)
        result = service.postprocess((emb, True))
        assert isinstance(result.embedding, list)
        assert all(isinstance(v, float) for v in result.embedding)

    def test_preprocess_blank_image_no_face(self, service: AdaFaceEmbeddingService) -> None:
        mock_detector = MagicMock()
        mock_detector.det_model.detect.return_value = (np.array([]), None)
        service._detector = mock_detector

        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        out = service.preprocess(blank)

        assert out["face_detected"] is False
        assert out["tensor"] is None

    def test_preprocess_empty_kps_no_face(self, service: AdaFaceEmbeddingService) -> None:
        mock_detector = MagicMock()
        mock_detector.det_model.detect.return_value = (np.zeros((1, 5)), None)
        service._detector = mock_detector

        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        out = service.preprocess(blank)

        assert out["face_detected"] is False

    def test_preprocess_face_detected_returns_tensor(
        self, service: AdaFaceEmbeddingService
    ) -> None:
        mock_detector = MagicMock()
        bboxes = np.array([[50.0, 50.0, 350.0, 350.0, 0.99]])
        kpss = np.array(
            [[[96.0, 110.0], [192.0, 110.0], [144.0, 150.0], [110.0, 200.0], [178.0, 200.0]]]
        )
        mock_detector.det_model.detect.return_value = (bboxes, kpss)
        service._detector = mock_detector

        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        with patch("zepiris.ml_inference.adaface_embedding.norm_crop") as mock_crop:
            mock_crop.return_value = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
            out = service.preprocess(image)

        assert out["face_detected"] is True
        assert out["tensor"] is not None
        assert out["tensor"].shape == (1, 3, 112, 112)
        assert out["tensor"].dtype == torch.float32

    def test_preprocess_tensor_value_range(self, service: AdaFaceEmbeddingService) -> None:
        """AdaFace normalization must map [0,255] → [-1, 1]."""
        mock_detector = MagicMock()
        bboxes = np.array([[50.0, 50.0, 350.0, 350.0, 0.99]])
        kpss = np.array(
            [[[96.0, 110.0], [192.0, 110.0], [144.0, 150.0], [110.0, 200.0], [178.0, 200.0]]]
        )
        mock_detector.det_model.detect.return_value = (bboxes, kpss)
        service._detector = mock_detector

        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        with patch("zepiris.ml_inference.adaface_embedding.norm_crop") as mock_crop:
            mock_crop.return_value = np.ones((112, 112, 3), dtype=np.uint8) * 255
            out = service.preprocess(image)

        assert torch.allclose(out["tensor"], torch.ones(1, 3, 112, 112), atol=1e-5)

    def test_predict_no_face_returns_zero_vector(self, service: AdaFaceEmbeddingService) -> None:
        emb, face_detected = service.predict({"tensor": None, "face_detected": False})
        assert face_detected is False
        assert emb.shape == (512,)
        assert np.all(emb == 0.0)

    def test_predict_with_face_calls_backbone(self, service: AdaFaceEmbeddingService) -> None:
        expected = torch.randn(1, 512)
        expected = expected / expected.norm(p=2, dim=1, keepdim=True)

        mock_backbone = MagicMock(return_value=expected)
        service._backbone = mock_backbone

        dummy_tensor = torch.randn(1, 3, 112, 112)
        emb, face_detected = service.predict({"tensor": dummy_tensor, "face_detected": True})

        assert face_detected is True
        assert emb.shape == (512,)
        mock_backbone.assert_called_once()

    def test_predict_returns_numpy_float32(self, service: AdaFaceEmbeddingService) -> None:
        expected = torch.randn(1, 512)
        expected = expected / expected.norm(p=2, dim=1, keepdim=True)
        service._backbone = MagicMock(return_value=expected)

        emb, _ = service.predict({"tensor": torch.randn(1, 3, 112, 112), "face_detected": True})
        assert isinstance(emb, np.ndarray)
        assert emb.dtype == np.float32

    def test_embed_is_alias_for_forward(self, service: AdaFaceEmbeddingService) -> None:
        fake_result = FaceEmbeddingResult(
            face_detected=False, embedding=[0.0] * 512, embedding_dim=512
        )
        with patch.object(service, "forward", return_value=fake_result) as mock_fwd:
            image = np.zeros((112, 112, 3), dtype=np.uint8)
            result = service.embed(image)
            mock_fwd.assert_called_once_with(image)
            assert result is fake_result


# ---------------------------------------------------------------------------
# MLServiceSettings — backend switching
# ---------------------------------------------------------------------------


class TestMLServiceSettings:
    @pytest.fixture(autouse=True)
    def clear_cache(self) -> None:
        from zepiris.ml_inference.app import get_ml_settings

        get_ml_settings.cache_clear()
        yield
        get_ml_settings.cache_clear()

    def test_default_backend_is_insightface(
        self, tmp_path: pytest.fixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        from zepiris.ml_inference.app import MLServiceSettings

        s = MLServiceSettings()
        assert s.face_embedding_backend == "insightface"

    def test_adaface_backend_set_via_env(
        self, tmp_path: pytest.fixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("ML_SERVICE_FACE_EMBEDDING_BACKEND", "adaface")
        from zepiris.ml_inference.app import MLServiceSettings

        s = MLServiceSettings()
        assert s.face_embedding_backend == "adaface"

    def test_adaface_default_local_model_path(
        self, tmp_path: pytest.fixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        from zepiris.ml_inference.app import MLServiceSettings

        s = MLServiceSettings()
        assert s.adaface_local_model_path == "/app/models/adaface_ir50.ckpt"

    def test_adaface_default_hf_model_file(
        self, tmp_path: pytest.fixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        from zepiris.ml_inference.app import MLServiceSettings

        s = MLServiceSettings()
        assert s.adaface_hf_model_file == "adaface_ir50.ckpt"

    def test_adaface_local_path_overridable(
        self, tmp_path: pytest.fixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("ML_SERVICE_ADAFACE_LOCAL_MODEL_PATH", "/custom/path/model.ckpt")
        from zepiris.ml_inference.app import MLServiceSettings

        s = MLServiceSettings()
        assert s.adaface_local_model_path == "/custom/path/model.ckpt"
