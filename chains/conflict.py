"""Lightweight, source-aware conflict detection for retrieved excerpts."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Protocol

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate


class SourceLike(Protocol):
    """The small portion of a QA source record needed for conflict checks."""

    source_file: str
    location: str
    excerpt: str


@dataclass(frozen=True)
class ConflictResult:
    """The outcome of a gated secondary source-comparison call."""

    checked: bool
    has_conflict: bool
    summary: str = ""
    sources: list[SourceLike] | None = None

    def source_positions(self) -> list[str]:
        """Fallback positions for the UI if the model omits its requested bullets."""

        grouped = group_by_source(self.sources or [])
        return [
            f"- **{filename}**: {sources[0].excerpt[:900]}"
            for filename, sources in grouped.items()
            if sources
        ]


CONFLICT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Compare the source excerpts for the user's question. Use only the "
            "excerpts. Start with exactly `CONFLICT: YES` if they make incompatible "
            "claims, otherwise start with exactly `CONFLICT: NO`. If there is a "
            "conflict, give one concise bullet for each position and name its source "
            "file. Do not treat complementary details as a conflict.",
        ),
        (
            "human",
            "Question:\n{question}\n\nSource excerpts:\n{excerpts}",
        ),
    ]
)

_WORD_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9'-]{2,}")
_NUMBER_PATTERN = re.compile(r"(?<![\w.])(?:\$|€|£)?\d+(?:\.\d+)?%?")
_NEGATION_PATTERN = re.compile(
    r"\b(?:no|not|never|prohibit(?:ed|s|ing)?|forbid(?:den|s)?|cannot|can't)\b",
    re.IGNORECASE,
)
_STOP_WORDS = {
    "and", "are", "but", "for", "from", "have", "must", "not", "that",
    "the", "this", "with", "will", "your", "you", "may", "shall", "than",
}


def _message_content(response: BaseMessage | str) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response.content, str):
        return response.content
    if isinstance(response.content, list):
        return "\n".join(
            item if isinstance(item, str) else item.get("text", "")
            for item in response.content
            if isinstance(item, str) or isinstance(item, dict)
        )
    return str(response.content)


def group_by_source(sources: Iterable[SourceLike]) -> dict[str, list[SourceLike]]:
    """Group retrieved excerpts by their original uploaded filename."""

    grouped: dict[str, list[SourceLike]] = defaultdict(list)
    for source in sources:
        grouped[source.source_file].append(source)
    return dict(grouped)


def _terms(text: str) -> set[str]:
    return {
        match.group(0).lower()
        for match in _WORD_PATTERN.finditer(text)
        if match.group(0).lower() not in _STOP_WORDS
    }


def _numeric_contexts(text: str) -> dict[str, set[str]]:
    """Map a meaningful number-bearing sentence shape to its numeric values."""

    contexts: dict[str, set[str]] = defaultdict(set)
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        numbers = _NUMBER_PATTERN.findall(sentence)
        terms = _terms(sentence)
        # Ignore isolated version numbers and dates such as "Policy 2026".
        if not numbers or len(terms) < 3:
            continue
        shape = _NUMBER_PATTERN.sub("<number>", sentence.lower())
        contexts[shape].update(numbers)
    return dict(contexts)


def _has_disagreement_signal(grouped: dict[str, list[SourceLike]]) -> bool:
    """Return a conservative, local signal before spending an LLM call.

    We only compare the first two retrieved excerpts per file: those are the most
    relevant evidence and keep the secondary prompt small. A signal requires a
    shared topic plus either different explicit numeric values or opposing
    negation language.
    """

    documents = [
        " ".join(source.excerpt for source in excerpts[:2])
        for excerpts in grouped.values()
    ]
    for index, left in enumerate(documents):
        left_terms = _terms(left)
        left_contexts = _numeric_contexts(left)
        left_negated = bool(_NEGATION_PATTERN.search(left))
        for right in documents[index + 1 :]:
            # Require a shared topic, not merely two unrelated retrieved files.
            if len(left_terms & _terms(right)) < 2:
                continue
            right_contexts = _numeric_contexts(right)
            for shape in left_contexts.keys() & right_contexts.keys():
                if left_contexts[shape] != right_contexts[shape]:
                    return True
            if left_negated != bool(_NEGATION_PATTERN.search(right)):
                return True
    return False


def should_check_conflict(sources: Iterable[SourceLike]) -> bool:
    """Whether multiple retrieved sources have a cheap disagreement signal."""

    grouped = group_by_source(sources)
    return len(grouped) >= 2 and _has_disagreement_signal(grouped)


def _format_excerpts(grouped: dict[str, list[SourceLike]]) -> str:
    parts: list[str] = []
    for filename, sources in grouped.items():
        excerpts = "\n".join(
            f"- ({source.location}) {source.excerpt}" for source in sources[:2]
        )
        parts.append(f"Source file: {filename}\n{excerpts}")
    return "\n\n".join(parts)


def detect_conflict(question: str, sources: list[SourceLike]) -> ConflictResult:
    """Compare suspicious multi-file retrieval results using the shared chat LLM.

    The import is intentionally local to avoid a module cycle while ensuring this
    secondary call uses the same default provider as retrieval QA.
    """

    grouped = group_by_source(sources)
    if len(grouped) < 2 or not _has_disagreement_signal(grouped):
        return ConflictResult(checked=False, has_conflict=False, sources=sources)

    from chains.qa_chain import get_chat_llm

    messages = CONFLICT_PROMPT.format_messages(
        question=question,
        excerpts=_format_excerpts(grouped),
    )
    response = get_chat_llm(provider="nvidia").invoke(messages)
    summary = _message_content(response).strip()
    has_conflict = not summary.upper().startswith("CONFLICT: NO")
    if summary.upper().startswith("CONFLICT: YES"):
        summary = summary.split("\n", 1)[1].strip() if "\n" in summary else ""

    return ConflictResult(
        checked=True,
        has_conflict=has_conflict,
        summary=summary,
        sources=sources,
    )
