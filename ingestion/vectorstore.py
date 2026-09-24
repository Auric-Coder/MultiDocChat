"""Vector store and embedding setup for MultiDocChat."""

from __future__ import annotations

import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Optional, TypeVar

from langchain_chroma import Chroma
from langchain_core.documents import Document

from ingestion.keyword_search import KeywordIndex


EMBEDDING_PROVIDER = "local"
LOCAL_PERSIST_ROOT = "chroma_db_local"
DEFAULT_COLLECTION_NAME = "multidocchat"
DEFAULT_SEARCH_STRATEGY = "hybrid"
DEFAULT_HYBRID_ALPHA = 0.7
# Chroma returns L2 distance from ``similarity_search_with_score``: lower is
# more relevant.  Keep matches relative to the best distance instead of using
# an embedding-model-specific absolute cutoff.
RELATIVE_DISTANCE_MARGIN = 1.5
RETRIEVAL_DISTANCE_METADATA_KEY = "retrieval_distance"

T = TypeVar("T")

_VECTOR_STORE: Optional[Chroma] = None
_KEYWORD_INDEX: Optional[KeywordIndex] = None


def get_embedding_function(provider="local"):
    """Return the configured embedding function.

    Keep embedding provider selection centralized here so the project can
    switch from local embeddings to OpenAI embeddings later without touching
    retrieval code.
    """
    if provider == "local":
        from langchain_huggingface import HuggingFaceEmbeddings

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

    global _VECTOR_STORE, _KEYWORD_INDEX
    _VECTOR_STORE = Chroma.from_documents(
        prepared_docs,
        embedding=embedding_fn,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    _KEYWORD_INDEX = KeywordIndex(prepared_docs)
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


def get_keyword_index() -> Optional[KeywordIndex]:
    """Return the active BM25 keyword index, or None if not built."""

    global _KEYWORD_INDEX
    if _KEYWORD_INDEX is None and _VECTOR_STORE is not None:
        try:
            res = _VECTOR_STORE.get()
            if res and res.get("documents"):
                docs = [
                    Document(page_content=text, metadata=meta or {})
                    for text, meta in zip(res["documents"], res.get("metadatas") or [])
                ]
                if docs:
                    _KEYWORD_INDEX = KeywordIndex(docs)
        except Exception:
            pass
    return _KEYWORD_INDEX


def get_retriever(k: int = 8):
    """Return the active vector store retriever for downstream chains."""

    return get_vector_store().as_retriever(search_kwargs={"k": k})


def similarity_search(query: str, *, k: int = 4) -> list[Document]:
    """Run a manual similarity search against the active vector store."""

    return get_vector_store().similarity_search(query, k=k)


def filter_by_relative_relevance(
    results: list[tuple[T, float]],
    *,
    margin: float = RELATIVE_DISTANCE_MARGIN,
) -> list[tuple[T, float]]:
    """Keep L2-distance matches within ``margin`` of the query's best match."""

    if not results:
        return results
    if margin < 1:
        raise ValueError("margin must be at least 1")

    best_distance = min(score for _, score in results)
    return [
        (item, score)
        for item, score in results
        if score <= best_distance * margin
    ]


def get_retriever_per_source(
    k_per_source: int = 5,
    *,
    relative_distance_margin: float = RELATIVE_DISTANCE_MARGIN,
) -> Callable[[str], list[Document]]:
    """Retrieve sufficiently relevant top chunks independently per source.

    Querying every source separately prevents a file with many chunks from
    crowding smaller uploaded files out of a single global top-k result. Weak
    chunks are then filtered against the best L2 distance for the whole query,
    so an unrelated file does not contribute merely because it has a top-k hit.
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
        scored_results: list[tuple[Document, float]] = []
        for source in source_files:
            if hasattr(vector_store, "similarity_search_with_score"):
                scored_results.extend(vector_store.similarity_search_with_score(
                    query, k=k_per_source, filter={"source_file": source}
                ))
            else:
                # Compatibility for older stores and lightweight test doubles.
                merged.extend(vector_store.similarity_search(
                    query, k=k_per_source, filter={"source_file": source}
                ))

        for document, distance in filter_by_relative_relevance(
            scored_results, margin=relative_distance_margin
        ):
            metadata = dict(document.metadata)
            metadata[RETRIEVAL_DISTANCE_METADATA_KEY] = float(distance)
            merged.append(Document(page_content=document.page_content, metadata=metadata))
        return merged

    return retrieve


def get_hybrid_retriever_per_source(
    k_per_source: int = 5,
    *,
    alpha: float = 0.7,
    strategy: str = "hybrid",
    relative_distance_margin: float = RELATIVE_DISTANCE_MARGIN,
) -> Callable[[str], list[Document]]:
    """Retrieve sufficiently relevant top chunks using hybrid search per source."""

    if strategy == "semantic":
        return get_retriever_per_source(
            k_per_source=k_per_source,
            relative_distance_margin=relative_distance_margin,
        )

    vector_store = get_vector_store()
    global _KEYWORD_INDEX
    if _KEYWORD_INDEX is None:
        result = vector_store.get()
        docs = [
            Document(page_content=txt, metadata=meta)
            for txt, meta in zip(result["documents"], result["metadatas"])
        ]
        _KEYWORD_INDEX = KeywordIndex(docs)
        
    def retrieve(query: str) -> list[Document]:
        semantic_results: list[Document] = []
        keyword_results: list[Document] = []
        
        if strategy in ("hybrid", "semantic"):
            semantic_results = get_retriever_per_source(
                k_per_source=k_per_source, 
                relative_distance_margin=relative_distance_margin
            )(query)
            
        if strategy in ("hybrid", "keyword"):
            # Use negative BM25 score as distance so lower=better
            scored_results = _KEYWORD_INDEX.search_per_source(query, k_per_source=k_per_source)
            for doc, score in filter_by_relative_relevance([
                (d, -s) for d, s in scored_results
            ], margin=relative_distance_margin):
                metadata = dict(doc.metadata)
                metadata[RETRIEVAL_DISTANCE_METADATA_KEY] = float(score)
                keyword_results.append(Document(page_content=doc.page_content, metadata=metadata))

        if strategy == "semantic":
            return semantic_results
        if strategy == "keyword":
            return keyword_results
            
        # Hybrid Fusion
        sem_map = {d.metadata.get("chunk_id"): d for d in semantic_results if d.metadata.get("chunk_id")}
        kw_map = {d.metadata.get("chunk_id"): d for d in keyword_results if d.metadata.get("chunk_id")}
        
        all_ids = set(sem_map.keys()) | set(kw_map.keys())
        
        # Get raw scores. For semantic, it's L2 (lower better), we invert for similarity.
        # For keyword, we already set distance to -BM25 score (lower better). 
        # Let's map back to similarity.
        sem_sims = [-d.metadata[RETRIEVAL_DISTANCE_METADATA_KEY] for d in semantic_results]
        kw_sims = [-d.metadata[RETRIEVAL_DISTANCE_METADATA_KEY] for d in keyword_results]
        
        def min_max(val, sims):
            if not sims: return 0.0
            min_s, max_s = min(sims), max(sims)
            if max_s == min_s: return 0.5
            return (val - min_s) / (max_s - min_s)
            
        fused = []
        for cid in all_ids:
            s_doc = sem_map.get(cid)
            k_doc = kw_map.get(cid)
            base_doc = s_doc or k_doc
            if not base_doc: continue
            
            s_sim = -s_doc.metadata[RETRIEVAL_DISTANCE_METADATA_KEY] if s_doc else (min(sem_sims) if sem_sims else 0.0)
            k_sim = -k_doc.metadata[RETRIEVAL_DISTANCE_METADATA_KEY] if k_doc else (min(kw_sims) if kw_sims else 0.0)
            
            score = alpha * min_max(s_sim, sem_sims) + (1 - alpha) * min_max(k_sim, kw_sims)
            
            # Map [0,1] score back to a distance where 0 is best
            dist = 1.0 - score
            meta = dict(base_doc.metadata)
            meta[RETRIEVAL_DISTANCE_METADATA_KEY] = dist
            fused.append((Document(page_content=base_doc.page_content, metadata=meta), dist))
            
        # Sort and filter by relative relevance again to ensure top quality
        fused.sort(key=lambda x: x[1])
        
        # Group by source to limit to top k_per_source? The prompt says "returns merged results filtered by relative relevance".
        merged = []
        for doc, dist in filter_by_relative_relevance(fused, margin=relative_distance_margin):
            merged.append(doc)
            
        return merged
        
    return retrieve


def _normalize_scores(
    results: list[tuple[Document, float]],
    *,
    higher_is_better: bool,
) -> list[tuple[Document, float]]:
    """Min-max normalize scores to [0, 1] where 1 is best.

    For L2 distances (lower is better) the scale is inverted so the best
    match maps to 1.  For BM25 scores (higher is better) the scale is
    kept as-is.
    """

    if not results:
        return results
    scores = [s for _, s in results]
    lo, hi = min(scores), max(scores)
    spread = hi - lo if hi != lo else 1.0
    if higher_is_better:
        return [(doc, (s - lo) / spread) for doc, s in results]
    return [(doc, 1.0 - (s - lo) / spread) for doc, s in results]


def get_hybrid_retriever_per_source(
    k_per_source: int = 5,
    *,
    alpha: float = DEFAULT_HYBRID_ALPHA,
    strategy: str = DEFAULT_SEARCH_STRATEGY,
    relative_distance_margin: float = RELATIVE_DISTANCE_MARGIN,
) -> Callable[[str], list[Document]]:
    """Build a retriever that supports semantic, keyword, or hybrid search.

    ``strategy`` controls which search signals are used:

    * ``"semantic"`` — ChromaDB vector similarity only (original behaviour).
    * ``"keyword"``  — BM25 keyword scoring only.
    * ``"hybrid"``   — Both signals fused with ``alpha`` weighting:
      ``score = alpha * semantic_norm + (1 - alpha) * keyword_norm``.

    Results are deduplicated by ``chunk_id`` and filtered by relative
    relevance so weak cross-file matches do not leak through.
    """

    keyword_index = get_keyword_index()
    if strategy == "semantic" or keyword_index is None:
        return get_retriever_per_source(
            k_per_source=k_per_source,
            relative_distance_margin=relative_distance_margin,
        )

    if strategy == "keyword":
        def _keyword_retrieve(query: str) -> list[Document]:
            bm25_results = keyword_index.search_per_source(
                query, k_per_source=k_per_source,
            )
            merged: list[Document] = []
            for doc, bm25_score in bm25_results:
                metadata = dict(doc.metadata)
                # Invert BM25 so lower = better, matching L2 convention.
                metadata[RETRIEVAL_DISTANCE_METADATA_KEY] = -bm25_score
                merged.append(
                    Document(page_content=doc.page_content, metadata=metadata)
                )
            return merged

        return _keyword_retrieve

    # --- hybrid ---
    vector_store = get_vector_store()
    all_metadatas = vector_store.get()["metadatas"]
    source_files = sorted(
        {
            m.get("source_file")
            for m in all_metadatas
            if m and m.get("source_file")
        }
    )

    def _hybrid_retrieve(query: str) -> list[Document]:
        # 1. Collect semantic scores (L2 distance, lower is better).
        semantic_raw: list[tuple[Document, float]] = []
        for source in source_files:
            if hasattr(vector_store, "similarity_search_with_score"):
                semantic_raw.extend(
                    vector_store.similarity_search_with_score(
                        query, k=k_per_source, filter={"source_file": source},
                    )
                )
            else:
                for doc in vector_store.similarity_search(
                    query, k=k_per_source, filter={"source_file": source},
                ):
                    semantic_raw.append((doc, 0.0))

        # 2. Collect BM25 scores (higher is better).
        bm25_raw = keyword_index.search_per_source(
            query, k_per_source=k_per_source,
        )

        # 3. Normalize both to [0, 1] where 1 = best.
        sem_norm = _normalize_scores(semantic_raw, higher_is_better=False)
        bm25_norm = _normalize_scores(bm25_raw, higher_is_better=True)

        # 4. Build lookup by chunk_id, fuse, and deduplicate.
        fused: dict[str, tuple[Document, float]] = {}

        for doc, norm_score in sem_norm:
            cid = doc.metadata.get("chunk_id", id(doc))
            existing_score = fused[cid][1] if cid in fused else 0.0
            fused[cid] = (doc, existing_score + alpha * norm_score)

        for doc, norm_score in bm25_norm:
            cid = doc.metadata.get("chunk_id", id(doc))
            if cid in fused:
                fused[cid] = (fused[cid][0], fused[cid][1] + (1 - alpha) * norm_score)
            else:
                fused[cid] = (doc, (1 - alpha) * norm_score)

        # 5. Sort by fused score descending and convert to Documents.
        ranked = sorted(fused.values(), key=lambda x: x[1], reverse=True)

        merged: list[Document] = []
        for doc, fused_score in ranked:
            metadata = dict(doc.metadata)
            # Store inverted fused score so lower = better (L2 convention).
            metadata[RETRIEVAL_DISTANCE_METADATA_KEY] = -fused_score
            merged.append(
                Document(page_content=doc.page_content, metadata=metadata)
            )
        return merged

    return _hybrid_retrieve


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
