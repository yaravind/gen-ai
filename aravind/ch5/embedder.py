import logging

from langchain_community.embeddings import HuggingFaceEmbeddings
# TODO from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import TypedDict

LOG = logging.getLogger('embedder')


class EmbeddingResult():
    def __init__(self, vector: list[float], source: dict, chunk: str):
        self._vector = vector
        self._source = source
        self._chunk = chunk

    @property
    def vector(self) -> list[float]:
        return self._vector

    @property
    def source(self) -> dict:
        return self._source

    @property
    def chunk(self) -> str:
        return self._chunk

    def __str__(self) -> str:
        return f"EmbeddingResult(source={self._source}, len(vector)={len(self._vector)}, len(chunk)='{len(self._chunk)}')"


model_name = "all-MiniLM-L6-v2"
embedding_model = HuggingFaceEmbeddings()


def get_embeddings(chunk: Document) -> EmbeddingResult:
    embedding_vec = embedding_model.embed_documents([chunk.page_content])
    LOG.debug(f"Chunk: {chunk.metadata}, Vector count: {len(embedding_vec)}")
    return EmbeddingResult(embedding_vec[0], chunk.metadata['source'], chunk.page_content)


# def get_embeddings(chunk: Document) -> list[float]:
#     model_name = "all-MiniLM-L6-v2"
#     # model_kwargs = {'device': 'gpu'}
#     # encode_kwargs = {'normalize_embeddings': False}
#     embedding_model = HuggingFaceEmbeddings()
#
#     embedding_vec = embedding_model.embed_documents([chunk.page_content])
#     return {'vector': embedding_vec[0], 'metadata': chunk.metadata, 'page_content': chunk.page_content}


def get_embeddings_for_all_chunks(chunks: list[Document]) -> list[list[float]]:
    LOG.debug(f"Embedding vector dimension: {embedding_model.client.get_sentence_embedding_dimension()}")
    LOG.debug(f"Embedding model info: {embedding_model.model_dump()}")
    embedding_vectors = embedding_model.embed_documents([chunk.page_content for chunk in chunks])
    LOG.debug(f"Chunk count: {len(chunks)}, Vector count: {len(embedding_vectors)}")

    return embedding_vectors
