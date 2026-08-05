"""RAGAS-inspired evaluation metrics engine for MultiDocChat.

Provides automated quantitative scoring for:
1. Faithfulness — Groundedness of answer in retrieved context chunks.
2. Answer Relevancy — Direct relevance of response to the user query.
3. Context Precision — Proportion of retrieved chunks relevant to query.
4. Context Recall — Proportion of ground truth facts captured in retrieval.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Sequence
from langchain_core.documents import Document


def _tokenize_clean(text: str) -> set[str]:
    stop_words = {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if",
        "in", "into", "is", "it", "no", "not", "of", "on", "or", "such",
        "that", "the", "their", "then", "there", "these", "they", "this",
        "to", "was", "will", "with", "what", "how", "where", "who", "which"
    }
    words = re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", text.lower())
    return {w for w in words if w not in stop_words}


def evaluate_faithfulness(answer: str, retrieved_chunks: Sequence[Document | Any]) -> float:
    """Measure proportion of non-trivial answer terms found in retrieved context."""
    if not answer or not retrieved_chunks:
        return 0.0

    if "could not find" in answer.lower():
        return 1.0  # Honest fallback is 100% faithful

    answer_tokens = _tokenize_clean(answer)
    if not answer_tokens:
        return 1.0

    context_text = " ".join(
        getattr(c, "page_content", getattr(c, "excerpt", str(c))) for c in retrieved_chunks
    ).lower()
    context_tokens = _tokenize_clean(context_text)

    overlap = answer_tokens.intersection(context_tokens)
    return round(len(overlap) / len(answer_tokens), 2)


def evaluate_answer_relevancy(question: str, answer: str) -> float:
    """Measure keyword/concept overlap between question and answer."""
    q_tokens = _tokenize_clean(question)
    a_tokens = _tokenize_clean(answer)

    if not q_tokens or not a_tokens:
        return 0.0

    overlap = q_tokens.intersection(a_tokens)
    return round(len(overlap) / len(q_tokens), 2)


def evaluate_context_precision(question: str, retrieved_chunks: Sequence[Document | Any]) -> float:
    """Measure fraction of retrieved chunks that contain key terms from question."""
    q_tokens = _tokenize_clean(question)
    if not q_tokens or not retrieved_chunks:
        return 0.0

    relevant_count = 0
    for chunk in retrieved_chunks:
        text = getattr(chunk, "page_content", getattr(chunk, "excerpt", str(chunk)))
        chunk_tokens = _tokenize_clean(text)
        if q_tokens.intersection(chunk_tokens):
            relevant_count += 1

    return round(relevant_count / len(retrieved_chunks), 2)


def evaluate_context_recall(expected: str, retrieved_chunks: Sequence[Document | Any]) -> float:
    """Measure fraction of expected ground-truth terms present in retrieved context."""
    if not expected or not retrieved_chunks:
        return 1.0  # No ground truth specified

    exp_tokens = _tokenize_clean(expected)
    if not exp_tokens:
        return 1.0

    context_text = " ".join(
        getattr(c, "page_content", getattr(c, "excerpt", str(c))) for c in retrieved_chunks
    ).lower()
    context_tokens = _tokenize_clean(context_text)

    overlap = exp_tokens.intersection(context_tokens)
    return round(len(overlap) / len(exp_tokens), 2)


def run_rag_evaluation(
    question: str,
    answer: str,
    retrieved_sources: Sequence[Any],
    expected: str = "",
) -> Dict[str, float]:
    """Calculate all 4 RAG metrics and return overall composite RAG score."""
    faithfulness = evaluate_faithfulness(answer, retrieved_sources)
    relevancy = evaluate_answer_relevancy(question, answer)
    precision = evaluate_context_precision(question, retrieved_sources)
    recall = evaluate_context_recall(expected, retrieved_sources)

    rag_score = round(
        0.35 * faithfulness + 0.25 * relevancy + 0.25 * precision + 0.15 * recall, 2
    )

    return {
        "faithfulness": faithfulness,
        "relevancy": relevancy,
        "precision": precision,
        "recall": recall,
        "rag_score": rag_score,
    }
