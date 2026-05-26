from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from minio import Minio

from zepiris.api.routes import build_api_router
from zepiris.config import get_settings
from zepiris.exception_handlers import register_exception_handlers
from zepiris.services.embedding import MLInferenceEmbeddingService
from zepiris.services.iqa import MLInferenceIQAService
from zepiris.services.milvus_store import MilvusFaceStore
from zepiris.services.minio_storage import MinioStorageService
from zepiris.services.ml_client import MLInferenceClient


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    minio_client = Minio(
        settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
    )
    minio_storage = MinioStorageService(minio_client, settings.minio_bucket)
    minio_storage.ensure_bucket()

    milvus = MilvusFaceStore(
        alias="default",
        host=settings.milvus_host,
        port=settings.milvus_port,
        collection_name=settings.milvus_collection,
        embedding_dim=settings.milvus_embedding_dim,
    )
    milvus.connect()
    milvus.ensure_collection()

    base = settings.ml_inference_service_url.rstrip("/")
    ml_client = MLInferenceClient(base)

    app.state.minio_storage = minio_storage
    app.state.iqa = MLInferenceIQAService(ml_client)
    app.state.embedding = MLInferenceEmbeddingService(ml_client)
    app.state.milvus = milvus

    yield

    await ml_client.aclose()
    milvus.disconnect()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        lifespan=lifespan,
    )
    register_exception_handlers(app)
    app.include_router(build_api_router())
    return app


app = create_app()


def run() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "zepiris.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )
