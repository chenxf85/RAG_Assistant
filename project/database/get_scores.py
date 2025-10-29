import numpy as np


def get_scores(question, vectordb,**kwargs):
    search_args = {
        "k": kwargs["top_k_query"],
        "score_threshold": kwargs["score_threshold"]
    }
    if kwargs.get("rag_files"):
        rag_filter = {"source": {"$in": kwargs["rag_files"]}}
        search_args["filter"] = rag_filter

    if kwargs["search_type"] == "similarity_score_threshold":

        docs_with_scores = vectordb.similarity_search_with_relevance_scores(
            question, **search_args
        )

        scores_with_docs = [(doc.metadata["source"], score) for doc, score in
                            docs_with_scores]  # score（-1，1)转化为（0，1）

    elif kwargs["search_type"] == "mmr":
        search_args["fetch_k"] = kwargs["fetch_k"]
        search_args["lambda_mult"] = kwargs["lambda_mult"]
        scores_with_docs = MMR(question, **search_args)
    else:
        scores_with_docs = ""

    return scores_with_docs
def cosine_similarity( vec1, vec2):
    """计算两个向量的余弦相似度"""
    vec1, vec2 = np.array(vec1), np.array(vec2)
    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return float(np.dot(vec1, vec2) / denom) if denom != 0 else 0.0


def MMR( question: str,vectordb, **kwargs) -> list[tuple[str, list[float]]]:
    """
    MMR
    算法，返回被选中的文档的
    source
    和对应得分。
    """

    lambda_mult = kwargs.get("lambda_mult", 0.5)
    k = kwargs.get("k", 5)
    fetch_k = kwargs.get("fetch_k", 20)

    # 第一步：预选 fetch_k 个最相似文档
    docs_with_scores = vectordb.similarity_search_with_relevance_scores(
        question, k=fetch_k
    )
    ids = [doc.id for doc, socre in docs_with_scores]
    #      print("\n\n".join([doc.page_content for doc, socre in docs_with_scores]))
    topk_embeddings = vectordb.get(ids, include=["embeddings"])["embeddings"]  # 得分从高到低排序的embedding

    # 第二步：在 topk_embeddings 上执行 MMR
    selected = []  # 选中的文档索引,初始化选中最相似的文档
    remaining = list(range(len(topk_embeddings)))  # 剩余未选择的文档索引
    mmr_scores = []  # 存储 MMR 得分
    query_embedding = vectordb._embedding_function.embed_query(question)

    for _ in range(min(k, len(topk_embeddings))):
        best_score = -np.inf
        best_idx = -1

        # 遍历剩余的候选文档
        for idx in remaining:
            candidate = topk_embeddings[idx]
            sim_query = cosine_similarity(query_embedding, candidate)
            #     print(f"{idx}:{sim_query}")
            sim_div = max(
                cosine_similarity(candidate, topk_embeddings[sel])
                for sel in selected
            ) if selected else 0.0

            score = lambda_mult * sim_query - (1 - lambda_mult) * sim_div

            if score > best_score:
                best_score = score
                best_idx = idx

        # 获取对应文档的 source 和得分
        best_doc = docs_with_scores[best_idx][0]  # 文档对象
        mmr_scores.append((best_doc.metadata["source"], best_score))
        selected.append(best_idx)  # 将当前选中的索引加入选中列表
        remaining.remove(best_idx)  # 从剩余候选中移除

    return mmr_scores
