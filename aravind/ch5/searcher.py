from langchain_core.documents import Document
from pymilvus import MilvusClient

from aravind.ch5.embedder import get_embeddings


def search_for(search_text: str, vec_store: MilvusClient, collection_name: str, metric_type: str, limit: int = 5) -> \
        list[list[dict]]:
    query_doc = Document(page_content=search_text, metadata={"source": "user-query"})
    srch_vec = get_embeddings(query_doc)
    res: list[list[dict]] = vec_store.search(collection_name=collection_name,
                                             data=[srch_vec.vector],
                                             limit=limit,
                                             output_fields=["source", "chunk"],
                                             search_params={"metric_type": metric_type, "params": {}})

    return res


def count_result_set(result_sets):
    count = 0
    for result_set in result_sets:
        count += len(result_set)
    return count


# update below code to return only top k chunks
def get_top_k(result_sets, top_k=5) -> list[str]:
    chunks = []
    for result_set in result_sets:
        for result in result_set:
            chunks.append(result['entity'])
            if len(chunks) >= top_k:
                break
    return chunks
