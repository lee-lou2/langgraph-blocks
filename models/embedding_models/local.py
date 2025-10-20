import enum

from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")


class HuggingfaceEmbeddingModel(enum.StrEnum):
    """허깅페이스 임베딩 모델"""

    QWEN3_0_6B = "Qwen/Qwen3-Embedding-0.6B"
    QWEN3_4B = "Qwen/Qwen3-Embedding-4B"
    QWEN3_8B = "Qwen/Qwen3-Embedding-8B"


class LocalEmbedding(HuggingFaceEmbeddings):
    """
    온디바이스 임베딩

    Args:
        embedding_model: 사용할 임베딩 모델 (기본: QWEN3_0_6B)
    """

    def __init__(
        self,
        embedding_model: HuggingfaceEmbeddingModel = HuggingfaceEmbeddingModel.QWEN3_0_6B,
        **kwargs
    ):
        super().__init__(
            model_name=embedding_model,
            **kwargs,
        )
