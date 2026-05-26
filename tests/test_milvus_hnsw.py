"""Tests for MilvusFaceStore HNSW index configuration.

All Milvus SDK calls are mocked — no running Milvus instance required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from zepiris.services.milvus_store import MilvusFaceStore, VectorMatch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store() -> MilvusFaceStore:
    return MilvusFaceStore(
        alias="test",
        host="localhost",
        port=19530,
        collection_name="test_faces",
        embedding_dim=512,
    )


def _mock_collection(has_index: bool = False) -> MagicMock:
    col = MagicMock()
    col.has_index.return_value = has_index
    return col


# ---------------------------------------------------------------------------
# ensure_collection — index creation
# ---------------------------------------------------------------------------


class TestEnsureCollectionHNSW:
    @patch("zepiris.services.milvus_store.Collection")
    @patch("zepiris.services.milvus_store.utility")
    def test_creates_hnsw_index_on_new_collection(self, mock_utility, mock_collection_cls) -> None:
        mock_utility.has_collection.return_value = False
        col = _mock_collection(has_index=False)
        mock_collection_cls.return_value = col

        store = _make_store()
        store.ensure_collection()

        col.create_index.assert_called_once()
        kwargs = col.create_index.call_args
        index_params = kwargs[1]["index_params"] if kwargs[1] else kwargs[0][1]
        assert index_params["index_type"] == "HNSW"

    @patch("zepiris.services.milvus_store.Collection")
    @patch("zepiris.services.milvus_store.utility")
    def test_hnsw_index_uses_cosine_metric(self, mock_utility, mock_collection_cls) -> None:
        mock_utility.has_collection.return_value = False
        col = _mock_collection(has_index=False)
        mock_collection_cls.return_value = col

        store = _make_store()
        store.ensure_collection()

        kwargs = col.create_index.call_args
        index_params = kwargs[1]["index_params"] if kwargs[1] else kwargs[0][1]
        assert index_params["metric_type"] == "COSINE"

    @patch("zepiris.services.milvus_store.Collection")
    @patch("zepiris.services.milvus_store.utility")
    def test_hnsw_m_and_ef_construction_params(self, mock_utility, mock_collection_cls) -> None:
        mock_utility.has_collection.return_value = False
        col = _mock_collection(has_index=False)
        mock_collection_cls.return_value = col

        store = _make_store()
        store.ensure_collection()

        kwargs = col.create_index.call_args
        index_params = kwargs[1]["index_params"] if kwargs[1] else kwargs[0][1]
        params = index_params["params"]
        assert "M" in params
        assert "efConstruction" in params
        assert params["M"] >= 8, "M should be at least 8 for reasonable graph connectivity"
        assert params["efConstruction"] >= 100, "efConstruction should be at least 100"

    @patch("zepiris.services.milvus_store.Collection")
    @patch("zepiris.services.milvus_store.utility")
    def test_no_index_created_when_index_exists(self, mock_utility, mock_collection_cls) -> None:
        mock_utility.has_collection.return_value = True
        col = _mock_collection(has_index=True)
        mock_collection_cls.return_value = col

        store = _make_store()
        store.ensure_collection()

        col.create_index.assert_not_called()

    @patch("zepiris.services.milvus_store.Collection")
    @patch("zepiris.services.milvus_store.utility")
    def test_collection_is_loaded_after_index(self, mock_utility, mock_collection_cls) -> None:
        mock_utility.has_collection.return_value = False
        col = _mock_collection(has_index=False)
        mock_collection_cls.return_value = col

        store = _make_store()
        store.ensure_collection()

        col.load.assert_called_once()

    @patch("zepiris.services.milvus_store.Collection")
    @patch("zepiris.services.milvus_store.utility")
    def test_returns_cached_collection_on_second_call(
        self, mock_utility, mock_collection_cls
    ) -> None:
        mock_utility.has_collection.return_value = False
        col = _mock_collection(has_index=False)
        mock_collection_cls.return_value = col

        store = _make_store()
        first = store.ensure_collection()
        second = store.ensure_collection()

        assert first is second
        # Collection constructor and create_index called only once
        assert mock_collection_cls.call_count == 1
        assert col.create_index.call_count == 1


# ---------------------------------------------------------------------------
# search — ef parameter
# ---------------------------------------------------------------------------


class TestSearchHNSWParams:
    def _setup_store_with_col(self) -> tuple[MilvusFaceStore, MagicMock]:
        store = _make_store()
        col = MagicMock()
        hit = MagicMock()
        hit.get.side_effect = lambda k, *a: {
            "face_id": "f1",
            "tenant": "t1",
            "object_key": "k1",
            "distance": 0.9,
        }.get(k)
        col.search.return_value = [[hit]]
        store._collection = col
        return store, col

    def test_search_passes_ef_param(self) -> None:
        store, col = self._setup_store_with_col()
        store.search([0.1] * 512, top_k=5)

        search_call = col.search.call_args
        param = search_call[1]["param"] if search_call[1] else search_call[0][2]
        assert "ef" in param["params"]

    def test_search_ef_at_least_top_k(self) -> None:
        store, col = self._setup_store_with_col()
        store.search([0.1] * 512, top_k=10)

        param = col.search.call_args[1]["param"]
        assert param["params"]["ef"] >= 10

    def test_search_ef_minimum_is_64(self) -> None:
        """ef must be >= 64 even when top_k is small, for recall quality."""
        store, col = self._setup_store_with_col()
        store.search([0.1] * 512, top_k=1)

        param = col.search.call_args[1]["param"]
        assert param["params"]["ef"] >= 64

    def test_search_ef_scales_with_large_top_k(self) -> None:
        store, col = self._setup_store_with_col()
        col.search.return_value = [[]]
        store.search([0.1] * 512, top_k=200)

        param = col.search.call_args[1]["param"]
        assert param["params"]["ef"] >= 200

    def test_search_metric_type_remains_cosine(self) -> None:
        store, col = self._setup_store_with_col()
        store.search([0.1] * 512, top_k=5)

        param = col.search.call_args[1]["param"]
        assert param["metric_type"] == "COSINE"

    def test_search_returns_vector_matches(self) -> None:
        store, col = self._setup_store_with_col()
        results = store.search([0.1] * 512, top_k=5)
        assert len(results) == 1
        assert isinstance(results[0], VectorMatch)
        assert results[0].face_id == "f1"

    def test_search_empty_top_k_returns_early(self) -> None:
        store, col = self._setup_store_with_col()
        results = store.search([0.1] * 512, top_k=0)
        col.search.assert_not_called()
        assert results == []
