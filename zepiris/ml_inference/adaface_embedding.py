"""Face embedding service using AdaFace IR-50 (CVPR 2022)."""

from __future__ import annotations

import io

import numpy as np
import torch
from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop

from zepiris.ml_inference.base import ModelService, ModelServiceConfig
from zepiris.ml_inference.models.adaface import AdaFaceIR50
from zepiris.schemas.ml_inference import FaceEmbeddingResult

# AdaFace normalization: [-1, 1] (not ImageNet)
_ADAFACE_MEAN = 0.5
_ADAFACE_STD = 0.5
_ALIGNED_SIZE = 112


class AdaFaceEmbeddingService(ModelService):
    """Face embedding service using AdaFace IR-50.

    Pipeline:
        1. RetinaFace (InsightFace) detects faces + extracts 5-pt landmarks.
        2. norm_crop aligns the selected face to a 112×112 canonical crop.
        3. AdaFace IR-50 backbone produces a 512-d L2-normalized embedding.

    The embedding vector is compatible with the existing Milvus COSINE index
    — no schema migration needed when switching backends.
    """

    def __init__(
        self,
        huggingface_repo_id: str = "",
        huggingface_model_file: str = "adaface_ir50.ckpt",
        local_model_path: str | None = None,
        model_source: str = "auto",
        embedding_dim: int = 512,
        detection_size: tuple[int, int] = (640, 640),
        facial_area_threshold: float = 0.01,
        device: str = "cpu",
    ) -> None:
        """Initialize AdaFace embedding service.

        Args:
            huggingface_repo_id: HuggingFace repo containing the checkpoint
            huggingface_model_file: Checkpoint filename inside the repo
            local_model_path: Optional local path to the .ckpt file
            model_source: ``"auto"``, ``"local"``, or ``"huggingface"``
            embedding_dim: Expected output dimension (512 for IR-50)
            detection_size: RetinaFace detection resolution as (width, height)
            facial_area_threshold: Minimum face area as fraction of image area
            device: PyTorch device string (``"cpu"``, ``"cuda:0"``, ``"mps"``)
        """
        self._embedding_dim = embedding_dim
        self._detection_size = detection_size
        self._facial_area_threshold = facial_area_threshold
        self._backbone: AdaFaceIR50 | None = None
        self._detector: FaceAnalysis | None = None

        config = ModelServiceConfig(
            model_name="adaface_embedding",
            huggingface_repo_id=huggingface_repo_id,
            huggingface_model_file=huggingface_model_file,
            local_model_path=local_model_path,
            model_source=model_source,
            device=device,
        )
        super().__init__(config)

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _load_detector(self) -> FaceAnalysis:
        """Lazy-load InsightFace detector (detection model only, no recognition)."""
        if self._detector is not None:
            return self._detector

        ctx_id = 0 if self.config.device != "cpu" else -1
        app = FaceAnalysis(name="buffalo_l")
        app.prepare(ctx_id=ctx_id, det_size=self._detection_size)
        # Keep only the face detector; AdaFace handles recognition
        app.models = {k: v for k, v in app.models.items() if k == "detection"}

        self._detector = app
        return self._detector

    def load_model(self) -> AdaFaceIR50:
        """Load AdaFace IR-50 backbone from checkpoint.

        Supports two checkpoint formats:
        - Training checkpoint: ``{"state_dict": {"model.<key>": tensor, ...}}``
        - Plain state dict: ``{"<key>": tensor, ...}``

        Returns:
            AdaFaceIR50: Backbone in eval mode on the configured device
        """
        if self._backbone is not None:
            return self._backbone

        model_bytes = self._download_model()
        buffer = io.BytesIO(model_bytes)
        checkpoint = torch.load(buffer, map_location=self.config.device, weights_only=False)

        # Unwrap training checkpoint wrapper if present
        if "state_dict" in checkpoint:
            raw = checkpoint["state_dict"]
            # Strip "model." prefix added by AdaFace training harness
            state_dict = {
                (k[len("model.") :] if k.startswith("model.") else k): v for k, v in raw.items()
            }
        else:
            state_dict = checkpoint

        backbone = AdaFaceIR50()
        backbone.load_state_dict(state_dict, strict=True)
        backbone.to(self.config.device)
        backbone.eval()

        self._backbone = backbone
        return self._backbone

    # ------------------------------------------------------------------
    # Inference pipeline
    # ------------------------------------------------------------------

    def preprocess(self, image_rgb: np.ndarray) -> dict:
        """Detect faces and align the most central qualifying face to 112×112.

        Args:
            image_rgb: Input image in RGB format, shape (H, W, 3), dtype uint8

        Returns:
            dict with keys:
                ``"tensor"``: (1, 3, 112, 112) float32 tensor ready for AdaFace,
                              or ``None`` if no face was detected.
                ``"face_detected"``: bool
        """
        detector = self._load_detector()
        bboxes, kpss = detector.det_model.detect(image_rgb, max_num=0, metric="default")

        if len(bboxes) == 0 or kpss is None:
            return {"tensor": None, "face_detected": False}

        h, w = image_rgb.shape[:2]
        img_area = h * w
        img_center = np.array([w / 2.0, h / 2.0])

        # Filter by minimum face area, fall back to all detections if none qualify
        qualified = [
            i
            for i, box in enumerate(bboxes)
            if ((box[2] - box[0]) * (box[3] - box[1])) / img_area > self._facial_area_threshold
        ]
        candidates = qualified if qualified else list(range(len(bboxes)))

        # Pick the face closest to the image center
        selected = min(
            candidates,
            key=lambda i: np.linalg.norm(
                np.array(
                    [
                        (bboxes[i][0] + bboxes[i][2]) / 2.0,
                        (bboxes[i][1] + bboxes[i][3]) / 2.0,
                    ]
                )
                - img_center
            ),
        )

        # Align to canonical 112×112 crop using 5-point landmarks
        aligned_rgb = norm_crop(image_rgb, kpss[selected], image_size=_ALIGNED_SIZE)

        # AdaFace normalization: pixel ∈ [0,255] → [-1, 1]
        normalized = (aligned_rgb.astype(np.float32) / 255.0 - _ADAFACE_MEAN) / _ADAFACE_STD
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0)  # (1,3,112,112)

        return {"tensor": tensor, "face_detected": True}

    def predict(self, preprocessed_data: dict) -> tuple[np.ndarray, bool]:
        """Run AdaFace forward pass and return the embedding.

        Args:
            preprocessed_data: Output dict from ``preprocess()``

        Returns:
            tuple[np.ndarray, bool]: (embedding of shape (512,), face_detected)
        """
        if not preprocessed_data["face_detected"]:
            return np.zeros(self._embedding_dim, dtype=np.float32), False

        backbone = self.load_model()
        tensor = preprocessed_data["tensor"].to(self.config.device)

        with torch.no_grad():
            embedding = backbone(tensor)  # (1, 512), already L2-normalized

        return embedding.squeeze(0).cpu().numpy().astype(np.float32), True

    def postprocess(self, output: tuple[np.ndarray, bool]) -> FaceEmbeddingResult:
        """Wrap embedding in FaceEmbeddingResult schema.

        Args:
            output: (embedding, face_detected) from ``predict()``

        Returns:
            FaceEmbeddingResult: L2-normalized embedding with metadata
        """
        embedding, face_detected = output

        # Defensive re-normalization (backbone already normalizes, but be safe)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return FaceEmbeddingResult(
            face_detected=face_detected,
            embedding=embedding.tolist(),
            embedding_dim=len(embedding),
        )

    def embed(self, image_rgb: np.ndarray) -> FaceEmbeddingResult:
        """Generate AdaFace embedding from an RGB image.

        Convenience alias for ``forward()``. Drop-in replacement for
        ``FaceEmbeddingService.embed()``.

        Args:
            image_rgb: Input image in RGB format, shape (H, W, 3), dtype uint8

        Returns:
            FaceEmbeddingResult: 512-d L2-normalized embedding with detection status
        """
        return self.forward(image_rgb)
