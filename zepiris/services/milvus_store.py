from __future__ import annotations

from dataclasses import dataclass

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)


@dataclass
class VectorMatch:
    face_id: str
    tenant: str
    object_key: str
    distance: float


class MilvusFaceStore:
    """Milvus 2.x collection: VARCHAR pk, object_key, FLOAT_VECTOR embedding."""

    def __init__(
        self,
        alias: str,
        host: str,
        port: int,
        collection_name: str,
        embedding_dim: int,
    ) -> None:
        self._alias = alias
        self._host = host
        self._port = port
        self._collection_name = collection_name
        self._embedding_dim = embedding_dim
        self._collection: Collection | None = None

    def connect(self) -> None:
        connections.connect(alias=self._alias, host=self._host, port=str(self._port))

    def disconnect(self) -> None:
        connections.disconnect(alias=self._alias)
        self._collection = None

    def ensure_collection(self) -> Collection:
        if self._collection is not None:
            return self._collection

        fields = [
            FieldSchema(
                name="face_id",
                dtype=DataType.VARCHAR,
                max_length=128,
                is_primary=True,
            ),
            FieldSchema(
                name="tenant",
                dtype=DataType.VARCHAR,
                max_length=256,
            ),
            FieldSchema(
                name="object_key",
                dtype=DataType.VARCHAR,
                max_length=512,
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self._embedding_dim,
            ),
        ]
        schema = CollectionSchema(fields, description="ZepIris face embeddings")

        if utility.has_collection(self._collection_name, using=self._alias):
            col = Collection(self._collection_name, using=self._alias)
        else:
            col = Collection(
                name=self._collection_name,
                schema=schema,
                using=self._alias,
            )

        if not col.has_index():
            # HNSW graph index: O(log N) ANN search vs O(N) brute-force FLAT.
            # M=16 (graph connectivity) and efConstruction=200 (build quality)
            # match Zepto's published production configuration.
            col.create_index(
                field_name="embedding",
                index_params={
                    "index_type": "HNSW",
                    "metric_type": "COSINE",
                    "params": {"M": 16, "efConstruction": 200},
                },
            )
        col.load()
        self._collection = col
        return col

    def insert(self, face_id: str, tenant: str, object_key: str, embedding: list[float]) -> None:
        col = self.ensure_collection()
        col.insert([[face_id], [tenant], [object_key], [embedding]])
        col.flush()

    def search(
        self,
        embedding: list[float],
        top_k: int,
        tenant: str | None = None,
        threshold: float | None = None,
    ) -> list[VectorMatch]:
        col = self.ensure_collection()
        if top_k < 1:
            return []

        expr: str | None = None
        if tenant:
            expr = f'tenant == "{tenant}"'

        # ef must be >= top_k for HNSW; larger values improve recall at the
        # cost of latency. max(top_k, 64) keeps single-digit ms at scale.
        results = col.search(
            data=[embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": max(top_k, 64)}},
            limit=top_k,
            expr=expr,
            output_fields=["face_id", "tenant", "object_key"],
        )
        matches: list[VectorMatch] = []
        for hit in results[0]:
            distance = float(hit.get("distance", 0.0))
            if threshold is not None and distance < threshold:
                continue
            matches.append(
                VectorMatch(
                    face_id=str(hit.get("face_id") or ""),
                    tenant=str(hit.get("tenant") or ""),
                    object_key=str(hit.get("object_key") or ""),
                    distance=distance,
                )
            )
        return matches

    def delete(self, face_id: str) -> None:
        """Delete a face record by ID."""
        col = self.ensure_collection()
        expr = f'face_id == "{face_id}"'
        col.delete(expr)
        col.flush()

    def upsert(self, face_id: str, tenant: str, object_key: str, embedding: list[float]) -> None:
        """Atomically insert or replace a face record by primary key."""
        col = self.ensure_collection()
        col.upsert([[face_id], [tenant], [object_key], [embedding]])
        col.flush()

    def get_by_id(self, face_id: str) -> dict | None:
        """Retrieve a face record by ID."""
        col = self.ensure_collection()
        expr = f'face_id == "{face_id}"'
        results = col.query(expr=expr, output_fields=["face_id", "tenant", "object_key"])
        if not results:
            return None
        result = results[0]
        return {
            "face_id": str(result.get("face_id") or ""),
            "tenant": str(result.get("tenant") or ""),
            "object_key": str(result.get("object_key") or ""),
        }

    def exists(self, face_id: str) -> bool:
        """Check if a face ID exists."""
        return self.get_by_id(face_id) is not None
