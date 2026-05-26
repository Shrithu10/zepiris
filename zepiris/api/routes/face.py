from __future__ import annotations

import base64
import uuid

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, Query, UploadFile

from zepiris.deps import EmbeddingDep, IQADep, MilvusDep, MinioDep, SettingsDep
from zepiris.exceptions import (
    DuplicateFaceIdError,
    EmbeddingDimensionMismatchError,
    EmptyUploadError,
    FaceRecordNotFoundError,
    ImageEncodeError,
    ImageQualityCheckFailedError,
    ImageTooLargeError,
    MilvusOperationError,
)
from zepiris.schemas.face import (
    MAX_IMAGE_SIZE_BYTES,
    MAX_IMAGE_SIZE_MB,
    CRUDResult,
    DeleteResponse,
    SearchMatch,
    SearchResponse,
    SearchStruct,
    UpsertResponse,
)
from zepiris.schemas.ml_inference import (
    BlurDetectionResult,
    ImageQualityAssessmentResult,
    NSFWDetectionResult,
    SpoofDetectionResult,
)

router = APIRouter()


def _validate_image_bytes(raw: bytes) -> None:
    """Design doc: decoded image size < 5MB (we validate raw upload size)."""
    if not raw:
        raise EmptyUploadError()
    if len(raw) > MAX_IMAGE_SIZE_BYTES:
        mb = len(raw) / (1024 * 1024)
        raise ImageTooLargeError(mb=mb, max_mb=MAX_IMAGE_SIZE_MB)


def _decode_upload(raw: bytes) -> np.ndarray | None:
    arr = np.frombuffer(raw, dtype=np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb


async def _run_iqa_and_embed(
    raw: bytes,
    settings: SettingsDep,
    iqa_svc: IQADep,
    embedding_svc: EmbeddingDep,
    minio: MinioDep,
):
    """Store image in MinIO, run IQA (local + ML via base64), then embedding.

    Returns (object_key, ml_struct, vector).
    """
    object_key = minio.put_image(raw, content_type="image/jpeg")
    image_bgr = _decode_upload(raw)
    if image_bgr is None:
        ml_struct = ImageQualityAssessmentResult(
            passed=False,
            nsfw=NSFWDetectionResult(is_safe=True, probability=1.0),
            spoof=SpoofDetectionResult(is_live=True, probability=1.0),
            blur=BlurDetectionResult(is_sharp=False, probability=0.0),
        )
        return object_key, ml_struct, None

    ok, buf = cv2.imencode(".jpg", image_bgr)
    if not ok:
        raise ImageEncodeError()
    image_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

    ml_struct = await iqa_svc.assess(image_bgr, image_b64)

    if not ml_struct.passed:
        return object_key, ml_struct, None

    embed_result = await embedding_svc.embed(image_bgr)

    if not embed_result.face_detected:
        ml_struct = ImageQualityAssessmentResult(
            passed=False,
            nsfw=ml_struct.nsfw,
            spoof=ml_struct.spoof,
            blur=BlurDetectionResult(is_sharp=False, probability=0.0),
        )
        return object_key, ml_struct, None

    vector = embed_result.embedding
    if len(vector) != settings.milvus_embedding_dim:
        raise EmbeddingDimensionMismatchError(
            expected=settings.milvus_embedding_dim,
            got=len(vector),
        )

    return object_key, ml_struct, vector


# ---------------------------------------------------------------------------
# Document-aligned API (Search / Insert / Upsert / Delete)
# ---------------------------------------------------------------------------


@router.post("/search", response_model=SearchResponse)
async def search_face(
    settings: SettingsDep,
    minio: MinioDep,
    iqa_svc: IQADep,
    embedding_svc: EmbeddingDep,
    milvus: MilvusDep,
    id: str = Form(...),
    tenant: str = Form(...),
    file: UploadFile = File(...),
    top_k: int = 5,
    threshold: float | None = None,
) -> SearchResponse:
    """1-to-N vector similarity search with tenant filtering and threshold.

    Multipart: id, tenant, file (image). API encodes to base64 for the ML service.
    Threshold defaults to ZEPIRIS_MILVUS_SEARCH_THRESHOLD from settings.
    """
    request_id = str(uuid.uuid4())
    raw = await file.read()
    _validate_image_bytes(raw)

    object_key, ml_struct, vector = await _run_iqa_and_embed(
        raw, settings, iqa_svc, embedding_svc, minio
    )

    if vector is None:
        return SearchResponse(
            requestId=request_id,
            imageQualityAssessment=ml_struct,
            searchResult=SearchStruct(matches=[]),
        )

    search_threshold = threshold if threshold is not None else settings.milvus_search_threshold
    matches_raw = milvus.search(vector, top_k=top_k, tenant=tenant, threshold=search_threshold)
    search_matches = [SearchMatch(id=m.face_id, score=m.distance) for m in matches_raw]

    return SearchResponse(
        requestId=request_id,
        imageQualityAssessment=ml_struct,
        searchResult=SearchStruct(matches=search_matches),
    )


@router.post("/insert", response_model=UpsertResponse)
async def insert_face(
    settings: SettingsDep,
    minio: MinioDep,
    iqa_svc: IQADep,
    embedding_svc: EmbeddingDep,
    milvus: MilvusDep,
    id: str = Form(...),
    tenant: str = Form(...),
    file: UploadFile = File(...),
) -> UpsertResponse:
    """Register a new face (INSERT). Rejects duplicate face_id."""
    request_id = str(uuid.uuid4())
    raw = await file.read()
    _validate_image_bytes(raw)

    object_key, ml_struct, vector = await _run_iqa_and_embed(
        raw, settings, iqa_svc, embedding_svc, minio
    )

    if vector is None:
        raise ImageQualityCheckFailedError(
            detail={
                "message": "image_quality_check_failed",
                "imageQualityAssessment": ml_struct.model_dump(),
                "object_key": object_key,
            },
        )

    face_id = id if id else str(uuid.uuid4())

    if milvus.exists(face_id):
        raise DuplicateFaceIdError(face_id)

    try:
        milvus.insert(face_id=face_id, tenant=tenant, object_key=object_key, embedding=vector)
        crud_result = CRUDResult(operation="INSERT", status="success")
    except Exception as exc:
        raise MilvusOperationError("insert", exc) from exc

    return UpsertResponse(
        requestId=request_id,
        imageQualityAssessment=ml_struct,
        userOperationResult=crud_result,
    )


@router.post("/upsert", response_model=UpsertResponse)
async def upsert_face(
    settings: SettingsDep,
    minio: MinioDep,
    iqa_svc: IQADep,
    embedding_svc: EmbeddingDep,
    milvus: MilvusDep,
    id: str = Form(...),
    tenant: str = Form(...),
    file: UploadFile = File(...),
) -> UpsertResponse:
    """Insert or update a face (UPSERT)."""
    request_id = str(uuid.uuid4())
    raw = await file.read()
    _validate_image_bytes(raw)

    object_key, ml_struct, vector = await _run_iqa_and_embed(
        raw, settings, iqa_svc, embedding_svc, minio
    )

    if vector is None:
        raise ImageQualityCheckFailedError(
            detail={
                "message": "image_quality_check_failed",
                "imageQualityAssessment": ml_struct.model_dump(),
                "object_key": object_key,
            },
        )

    try:
        milvus.upsert(face_id=id, tenant=tenant, object_key=object_key, embedding=vector)
        crud_result = CRUDResult(operation="UPSERT", status="success")
    except Exception as exc:
        raise MilvusOperationError("upsert", exc) from exc

    return UpsertResponse(
        requestId=request_id,
        imageQualityAssessment=ml_struct,
        userOperationResult=crud_result,
    )


@router.delete("/delete", response_model=DeleteResponse)
async def delete_face(
    milvus: MilvusDep,
    face_id: str = Query(..., alias="id", description="Face id to delete"),
) -> DeleteResponse:
    """Delete a face record by ID (query parameter: id)."""
    request_id = str(uuid.uuid4())

    if not milvus.exists(face_id):
        return DeleteResponse(
            requestId=request_id,
            userOperationResult=CRUDResult(operation="DELETE", status="not_found"),
        )

    try:
        milvus.delete(face_id)
    except Exception as exc:
        raise MilvusOperationError("delete", exc) from exc

    return DeleteResponse(
        requestId=request_id,
        userOperationResult=CRUDResult(operation="DELETE", status="success"),
    )


@router.get("/get/{face_id}")
async def get_face(
    milvus: MilvusDep,
    face_id: str,
) -> dict:
    """Retrieve face metadata by ID."""
    record = milvus.get_by_id(face_id)
    if record is None:
        raise FaceRecordNotFoundError(face_id)
    return record
