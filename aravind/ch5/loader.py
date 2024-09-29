from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredMarkdownLoader, DirectoryLoader
from config import set_environment
import nltk
import os
from genai_utils import download_nltk_resource

set_environment()


def install_nltk():
    """
    wordnet, punkt, averaged_perceptron_tagger are needed for the UnstructuredMarkdownLoader.
    This is essential for sentence tokenization in NLTK.
    """
    # Check if 'wordnet' and 'punkt' are already downloaded
    download_nltk_resource("corpora", "wordnet")
    download_nltk_resource("tokenizers", "punkt")
    download_nltk_resource("corpora", "averaged_perceptron_tagger")


def get_docs(dir_path: str, doc_type="md", show_progress=False, **loader_kwargs) -> list[Document]:
    loader = DirectoryLoader(dir_path, glob=f"**/*.{doc_type}",
                             loader_cls=UnstructuredMarkdownLoader, loader_kwargs=loader_kwargs,
                             show_progress=show_progress)
    docs = loader.load()
    return docs


def print_doc(doc: Document):
    print(doc.type)
    print(doc.metadata)
    print(doc.page_content)


def get_docs_for_all_paths(paths: list[str]) -> list[Document]:
    all_docs = []
    for path in paths:
        docs = get_docs(path)
        print(f"Path: {path}, Markdown file count: {len(docs)}")
        all_docs.extend(docs)
    return all_docs
