"""Vector store and embedding setup for MultiDocChat."""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Iterable, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document


EMBEDDING_PROVIDER = "local"
LOCAL_PERSIST_ROOT = "chroma_db_local"
DEFAULT_COLLECTION_NAME = "multidocchat"

_VECTOR_STORE: Optional[Chroma] = None


def get_embedding_function(provider="local"):
    """Return the configured embedding function.

    Keep embedding provider selection centralized here so the project can
    switch from local embeddings to OpenAI embeddings later without touching
    retrieval code.
    """
    if provider == "local":
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    elif provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model="text-embedding-3-small")

    raise ValueError(f"Unsupported embedding provider: {provider}")


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("._-")
    return slug or "session"


def get_persist_directory(
    *,
    provider: str = EMBEDDING_PROVIDER,
    session_id: Optional[str] = None,
    base_dir: str | Path = ".",
) -> str:
    """Return a provider-specific Chroma persistence directory.

    The provider is baked into the path so local MiniLM vectors never share a
    physical collection with future OpenAI vectors.
    """

    root_name = (
        LOCAL_PERSIST_ROOT
        if provider == EMBEDDING_PROVIDER
        else f"chroma_db_{provider}"
    )
    root = Path(base_dir) / root_name
    if session_id:
        root = root / _slug(session_id)
    return str(root)


def make_temp_persist_directory(provider: str = EMBEDDING_PROVIDER) -> str:
    """Create a fresh provider-specific temporary Chroma directory."""

    return tempfile.mkdtemp(prefix=f"chroma_db_{provider}_")


def _clean_metadata_value(value):
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _prepare_documents(docs: Iterable[Document]) -> list[Document]:
    prepared_docs: list[Document] = []

    for doc in docs:
        prepared_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={
                    key: _clean_metadata_value(value)
                    for key, value in doc.metadata.items()
                },
            )
        )

    return prepared_docs


def create_vector_store(
    docs: Iterable[Document],
    *,
    persist_directory: Optional[str] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    reset: bool = True,
    provider: str = EMBEDDING_PROVIDER,
) -> Chroma:
    """Embed documents into a Chroma collection and keep it as the active store."""

    if provider != EMBEDDING_PROVIDER:
        raise ValueError(
            "Phase 2 through Phase 8 must use local embeddings only. "
            f"Received provider={provider!r}."
        )

    prepared_docs = _prepare_documents(docs)
    if not prepared_docs:
        raise ValueError("Cannot create a vector store without documents.")

    if persist_directory is None:
        persist_directory = make_temp_persist_directory(provider)
    elif reset and Path(persist_directory).exists():
        shutil.rmtree(persist_directory)

    embedding_fn = get_embedding_function(provider)

    global _VECTOR_STORE
    _VECTOR_STORE = Chroma.from_documents(
        prepared_docs,
        embedding=embedding_fn,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    return _VECTOR_STORE


def load_vector_store(
    *,
    persist_directory: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    provider: str = EMBEDDING_PROVIDER,
) -> Chroma:
    """Load an existing Chroma collection and keep it as the active store."""

    if provider != EMBEDDING_PROVIDER:
        raise ValueError(
            "Phase 2 through Phase 8 must use local embeddings only. "
            f"Received provider={provider!r}."
        )

    global _VECTOR_STORE
    _VECTOR_STORE = Chroma(
        collection_name=collection_name,
        embedding_function=get_embedding_function(provider),
        persist_directory=persist_directory,
    )
    return _VECTOR_STORE


def get_vector_store() -> Chroma:
    """Return the active Chroma vector store."""

    if _VECTOR_STORE is None:
        raise RuntimeError("Vector store has not been initialized yet.")
    return _VECTOR_STORE


def get_retriever(k: int = 4):
    """Return the active vector store retriever for downstream chains."""

    return get_vector_store().as_retriever(search_kwargs={"k": k})


def similarity_search(query: str, *, k: int = 4) -> list[Document]:
    """Run a manual similarity search against the active vector store."""

    return get_vector_store().similarity_search(query, k=k)


def get_retriever_per_source(k_per_source: int = 3) -> Callable[[str], list[Document]]:
    """Retrieve top chunks independently from each source file, then merge.

    Querying every source separately prevents a file with many chunks from
    crowding smaller uploaded files out of a single global top-k result.
    """

    vector_store = get_vector_store()
    all_metadatas = vector_store.get()["metadatas"]
    source_files = sorted(
        {
            metadata.get("source_file")
            for metadata in all_metadatas
            if metadata and metadata.get("source_file")
        }
    )

    def retrieve(query: str) -> list[Document]:
        merged: list[Document] = []
        for source in source_files:
            merged.extend(
                vector_store.similarity_search(
                    query,
                    k=k_per_source,
                    filter={"source_file": source},
                )
            )
        return merged

    return retrieve


if __name__ == "__main__":
    import argparse

    from ingestion.loaders import load_and_chunk

    parser = argparse.ArgumentParser(
        description="Embed files into local Chroma and run a similarity search."
    )
    parser.add_argument("query", help="Similarity search query.")
    parser.add_argument("files", nargs="+", help="Files to embed and search.")
    parser.add_argument("--k", type=int, default=4, help="Number of chunks to return.")
    parser.add_argument(
        "--session-id",
        help="Optional persistent session ID under chroma_db_local/.",
    )
    args = parser.parse_args()

    chunks: list[Document] = []
    for file_path in args.files:
        chunks.extend(load_and_chunk(file_path))

    persist_dir = (
        get_persist_directory(session_id=args.session_id)
        if args.session_id
        else make_temp_persist_directory()
    )
    create_vector_store(chunks, persist_directory=persist_dir)

    for result in similarity_search(args.query, k=args.k):
        preview = " ".join(result.page_content.split())[:240]
        print(f"\nmetadata={result.metadata}")
        print(f"preview={preview}")
