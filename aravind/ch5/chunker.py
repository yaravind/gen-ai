import logging

from langchain_core.documents import Document
from langchain_text_splitters.markdown import MarkdownTextSplitter

LOG = logging.getLogger('chunker')


# Chunk the documents
def get_chunks(document: Document) -> list[Document]:
    splitter = MarkdownTextSplitter()
    chunks = splitter.split_documents([document])
    return chunks


def get_chunks_for_all_docs(docs: list[Document]) -> list[Document]:
    all_chunks = []
    for doc in docs:
        chunks = get_chunks(doc)
        LOG.debug(f"Doc: {doc.metadata['source']}, Chunk count: {len(chunks)}")
        all_chunks.extend(chunks)
    return all_chunks


def print_chunks(chunks: list[Document]):
    print(f"Chunk count: {len(chunks)}")
    for chunk in chunks:
        print(f"Metadata: {chunk.metadata}")
        print(chunk.page_content)
