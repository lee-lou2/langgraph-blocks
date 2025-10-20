import enum

from pymilvus import (
    AnnSearchRequest,
    WeightedRanker,
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    RRFRanker,
)


class MilvusRerankType(enum.Enum):
    """밀버스 리랭크 타입"""

    WEIGHTED_RANKER = "weighted_ranker"
    RRF_RANKER = "rrf_ranker"


class Milvus:
    """
    밀버스 라이브러리

    Args:
        uri: 밀버스 접속을 위한 URI
        collection_name: 연결이 필요한 컬렉션
    """

    DEFAULT_MILVUS_URI = "./milvus.db"
    DEFAULT_COLLECTION_NAME = "milvus"

    def __init__(self, uri: str = None, collection_name: str = None):
        self.uri = self._get_uri(uri)
        self.collection_name = self._get_collection_name(collection_name)
        self.collection = None
        self._init_collection()

    def _get_uri(self, uri: str) -> str:
        return uri if uri else self.DEFAULT_MILVUS_URI

    def _get_collection_name(self, collection_name: str) -> str:
        return collection_name if collection_name else self.DEFAULT_COLLECTION_NAME

    def _init_collection(self):
        connections.connect(uri=self.uri)

        # 컬렉션 존재 여부 확인
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            self.collection.load()
            return

        collection_schema = CollectionSchema(
            [
                FieldSchema(
                    name="pk",
                    dtype=DataType.VARCHAR,
                    is_primary=True,
                    auto_id=True,
                    max_length=100,
                ),
                FieldSchema(
                    name="text",
                    dtype=DataType.VARCHAR,
                    max_length=512,
                ),
                FieldSchema(
                    name="sparse_vector",
                    dtype=DataType.SPARSE_FLOAT_VECTOR,
                ),
                FieldSchema(
                    name="dense_vector",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=128,
                ),
            ]
        )
        collection = Collection(
            self.collection_name, collection_schema, consistency_level="Bounded"
        )
        collection.create_index(
            "sparse_vector",
            {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"},
        )
        collection.create_index(
            "dense_vector",
            {"index_type": "AUTOINDEX", "metric_type": "IP"},
        )
        collection.load()
        self.collection = collection

    def dense_search(self, query_dense_embedding, params: dict = None, limit=10):
        res = self.collection.search(
            [query_dense_embedding],
            anns_field="dense_vector",
            limit=limit,
            output_fields=["text"],
            param={"metric_type": "IP", "params": {} if params is None else params},
        )[0]
        return [hit.get("text") for hit in res]

    def sparse_search(self, query_sparse_embedding, params: dict = None, limit=10):
        res = self.collection.search(
            [query_sparse_embedding],
            anns_field="sparse_vector",
            limit=limit,
            output_fields=["text"],
            param={
                "metric_type": "IP",
                "params": {} if params is None else params,
            },
        )[0]
        return [hit.get("text") for hit in res]

    def hybrid_search(
        self,
        query_dense_embedding,
        query_sparse_embedding,
        dense_params=None,
        sparse_params=None,
        ranker_type=MilvusRerankType.RRF_RANKER,
        sparse_weight=1.0,  # MilvusRerankType.WEIGHTED_RANKER 에서만 사용
        dense_weight=1.0,  # MilvusRerankType.WEIGHTED_RANKER 에서만 사용
        limit=10,
    ):
        dense_req = AnnSearchRequest(
            [query_dense_embedding],
            "dense_vector",
            {
                "metric_type": "IP",
                "params": {} if dense_params is None else dense_params,
            },
            limit=limit,
        )
        sparse_req = AnnSearchRequest(
            [query_sparse_embedding],
            "sparse_vector",
            {
                "metric_type": "IP",
                "params": {} if sparse_params is None else sparse_params,
            },
            limit=limit,
        )
        rerank = RRFRanker()
        if ranker_type == MilvusRerankType.WEIGHTED_RANKER:
            rerank = WeightedRanker(sparse_weight, dense_weight)
        res = self.collection.hybrid_search(
            [sparse_req, dense_req],
            rerank=rerank,
            limit=limit,
            output_fields=["text"],
        )[0]
        return [hit.get("text") for hit in res]
