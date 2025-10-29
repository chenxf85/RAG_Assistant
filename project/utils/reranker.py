from __future__ import annotations

import operator
from typing import Optional, Sequence

from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from pydantic import ConfigDict

from langchain.retrievers.document_compressors.cross_encoder import BaseCrossEncoder
from pydantic import Field


from langchain_core.language_models import BaseLanguageModel
from llm.model_to_llm import model_to_llm
class LLMReranker2(BaseDocumentCompressor):
    """
    Document compressor that uses an LLM to rerank documents.
    """

    llm: BaseLanguageModel
    top_n: int = 3
    score_threshold: float = 0.0
    scores: list[tuple[str, float]] = Field(default_factory=list)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _build_prompt(self, query: str, docs: Sequence[Document]) -> str:
        docs_text = "\n".join(
            [f"[{i+1}] {doc.page_content.strip()}" for i, doc in enumerate(docs)]
        )
        #如果还想实现上下文压缩，其实这里可以再构建一个上下文压缩的LLM compressor，对重排后文档压缩
        #或者直接一步LLM调用实现；
        prompt = f"""
        你是一个信息检索专家。请根据以下用户问题，对提供的文档进行相关性评分（0~1之间）。
        【用户问题】：
        {query}\n
         【候选文档】(每个文档前面有编号比如[1],[2]...)：{docs_text}\n
       【打分标准】：
- **1.0**：文档直接回答了问题，包含关键概念、方法、定义或结论。
- **0.8-0.9**：文档与问题高度相关，虽然不是直接答案，但有核心背景或推理线索。
- **0.5-0.7**：文档部分相关，只涉及问题的一部分或相关主题。
- **0.2-0.4**：文档仅有少量相似词汇或远程关联，无法帮助回答问题。
- **0.0-0.1**：完全无关或与主题相反。\n
【输出格式】：
编号,分数
例如：
1,0.9
2,0.2
仅输出这些编号和分数，不要额外说明。
"""
        return prompt.strip()

    def _parse_scores(self, text: str) -> list[tuple[int, float]]:
        scores = []
        for line in text.strip().splitlines():
            try:
                idx, score = line.strip().split(",")
                scores.append((int(idx) - 1, float(score)))
            except Exception:
                continue
        return scores

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Rerank documents using an LLM.
        """
        if not documents:
            return []
        # self.llm = model_to_llm(
        #     model_type="ZHIPUAI",
        #     model="glm-4.5-flash",
        #     temperature=0
        # )  #test
        # 构造prompt
        prompt = self._build_prompt(query, documents)

        # 调用LLM
        response = self.llm.invoke(prompt)
        text = getattr(response, "content", response)

        # 解析分数
        parsed_scores = self._parse_scores(text)

        # 按分数排序
        docs_with_scores = [
            (documents[idx].metadata.get("source", f"doc_{idx}"), documents[idx], score)
            for idx, score in parsed_scores
            if 0 <= idx < len(documents)
        ]

        result = sorted(docs_with_scores, key=operator.itemgetter(2), reverse=True)
        self.scores = [(name, score) for name, _, score in result[: self.top_n] if score > self.score_threshold]

        # 取 top_n
        reranked_docs = [doc for _, doc, score in result[: self.top_n] if score > self.score_threshold]
        return reranked_docs

class CrossEncoderReranker2(BaseDocumentCompressor):
    """Document compressor that uses CrossEncoder for reranking."""

    model: BaseCrossEncoder
    """CrossEncoder model to use for scoring similarity
      between the query and documents."""
    top_n: int = 3
    """Number of documents to return."""
    scores: list[tuple[str,float]] = Field(default_factory=list)
    score_threshold: float = Field(default_factory=float)
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compress_documents(
            self,
            documents: Sequence[Document],
            query: str,
            callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Rerank documents using CrossEncoder.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """
        text_pairs = []
        file_names = []
        for doc in documents:
            text_pairs.append((query, doc.page_content))
            file_names.append(doc.metadata["source"])
        scores = self.model.score(text_pairs)
        docs_with_scores = list(zip(file_names,documents, scores))
        result = sorted(docs_with_scores, key=operator.itemgetter(2), reverse=True)
        self.scores = [(name,score)  for name,doc,score in result[:self.top_n] if score>self.score_threshold]
       # print(result)
        #self.scores = [(name, score) for name, doc, score in result[:self.top_n]]
        return [doc for _, doc, score in result[: self.top_n] if score > self.score_threshold ]
