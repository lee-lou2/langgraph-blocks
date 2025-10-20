import enum

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class HuggingfaceRerankModel(enum.StrEnum):
    """허깅페이스 리랭크 모델"""

    QWEN3_0_6B = "Qwen/Qwen3-Reranker-0.6B"
    QWEN3_4B = "Qwen/Qwen3-Reranker-4B"
    QWEN3_8B = "Qwen/Qwen3-Reranker-8B"


class LocalReranking:
    """
    로컬 리랭킹 모델

    Args:
        rerank_model: 사용할 리랭크 모델 (기본: QWEN3_0_6B)
    """

    INSTRUCT = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )
    MAX_LEN = 8192
    PREFIX = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(
        self, rerank_model: HuggingfaceRerankModel = HuggingfaceRerankModel.QWEN3_0_6B
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(
            rerank_model,
            padding_side="left",
        )
        self.model = (
            AutoModelForCausalLM.from_pretrained(rerank_model).to(self.device).eval()
        )

    @property
    def tid_no(self):
        """'no' 토큰의 ID"""
        return self.tokenizer.convert_tokens_to_ids("no")

    @property
    def tid_yes(self):
        """'yes' 토큰의 ID"""
        return self.tokenizer.convert_tokens_to_ids("yes")

    @property
    def prefix_ids(self):
        """프롬프트 접두사 토큰 ID 리스트"""
        return self.tokenizer.encode(self.PREFIX, add_special_tokens=False)

    @property
    def suffix_ids(self):
        """프롬프트 접미사 토큰 ID 리스트"""
        return self.tokenizer.encode(self.SUFFIX, add_special_tokens=False)

    def scores(
        self,
        query: str,
        docs: list[str],
        top_k: int | None = None,
        batch_size: int = 16,
    ):
        """
        리랭킹 스코어 계산
        1. query와 docs를 받아 각 문서의 관련도 점수(0~1, P('yes'))를 계산
        2. (doc, score) 리스트를 score 내림차순으로 반환
        3. top_k가 주어지면 상위 k개만 반환

        Args:
            query: 검색 쿼리
            docs: 문서 리스트
            top_k: 상위 k개 결과만 반환 (기본: None, 전체 반환
            batch_size: 배치 크기 (기본: 16)

        Returns:
            list[tuple[str, float]]: (문서, 관련도 점수) 리스트
        """
        results = []
        for i in range(0, len(docs), batch_size):
            chunk = docs[i : i + batch_size]
            # Instruct/Query/Document 한 번에 문자열로 구성
            pairs = [
                f"<Instruct>: {self.INSTRUCT}\n<Query>: {query}\n<Document>: {d}"
                for d in chunk
            ]
            enc = self.tokenizer(
                pairs,
                padding=False,
                truncation="longest_first",
                return_attention_mask=False,
                max_length=self.MAX_LEN - len(self.prefix_ids) - len(self.suffix_ids),
            )
            # prefix/suffix 삽입
            for j, ids in enumerate(enc["input_ids"]):
                enc["input_ids"][j] = self.prefix_ids + ids + self.suffix_ids

            # 배치 패딩 후 텐서화
            enc = self.tokenizer.pad(
                enc, padding=True, return_tensors="pt", max_length=self.MAX_LEN
            )
            for k in enc:
                enc[k] = enc[k].to(self.device)

            with torch.no_grad():
                logits = self.model(**enc).logits[:, -1, :]  # 마지막 토큰 로짓
                scores = torch.stack(
                    [logits[:, self.tid_no], logits[:, self.tid_yes]], dim=1
                )
                probs_yes = torch.softmax(scores, dim=1)[:, 1].tolist()

            results.extend(zip(chunk, probs_yes))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k] if top_k else results
