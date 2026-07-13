"""Retrieval QA helpers with inline source attribution."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from ingestion.vectorstore import RETRIEVAL_DISTANCE_METADATA_KEY, get_retriever_per_source

if TYPE_CHECKING:
    from chains.conflict import ConflictResult


DEFAULT_CHAT_PROVIDER = "nvidia"
DEFAULT_NVIDIA_MODEL = "meta/llama-3.1-8b-instruct"
NVIDIA_REQUEST_TIMEOUT_SECONDS = 120
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
EM_DASH = "\N{EM DASH}"
CITATION_FORMAT = f"[filename {EM_DASH} section/page]"
SOURCE_SEPARATOR = f" {EM_DASH} "
CITATION_PATTERN = re.compile(
    # Accept the canonical ``[file — location]`` and common model near-misses:
    # no separator spaces, en dashes, and parentheses instead of square brackets.
    r"[\[(]\s*([^\[\]()]+?)\s*(?:"
    + re.escape(EM_DASH)
    + r"|–|-)\s*([^\[\]()]+?)\s*[\])]"
)
ENUMERATION_QUERY_PATTERN = re.compile(
    r"\b(?:list|all|every|summary|summarize)\b", re.IGNORECASE
)
ENUMERATION_K_PER_SOURCE = 20
SUMMARY_MAX_TOKENS = 700
NVIDIA_FREQUENCY_PENALTY = 0.5


@dataclass(frozen=True)
class SourceExcerpt:
    """A retrieved excerpt labeled for citation and UI display."""

    source_file: str
    location: str
    chunk_id: str
    excerpt: str
    retrieval_distance: float | None = None

    @property
    def citation(self) -> str:
        return f"{self.source_file}{SOURCE_SEPARATOR}{self.location}"


@dataclass(frozen=True)
class QAResult:
    """Answer text plus the cited excerpts that support it."""

    answer: str
    sources: list[SourceExcerpt]
    retrieved_sources: list[SourceExcerpt]
    conflict: ConflictResult | None = None


QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions using only the provided document excerpts.\n"
            "Only answer using the provided excerpts. For every factual statement, "
            f"cite the source in the format {CITATION_FORMAT}. If the excerpts "
            "don't contain the answer, say so.\n"
            "CITATIONS ARE REQUIRED. After every factual sentence, copy the exact "
            "value shown after `Source:` in its supporting excerpt and put it in "
            f"square brackets exactly like this: {CITATION_FORMAT}. Keep the "
            "filename and location verbatim; do not omit brackets, replace the "
            "dash, cite an excerpt number, or invent a source.\n"
            "Do not cite sources that are not present in the excerpts. Keep the "
            "answer concise and directly relevant to the question.\n"
            "For a request to list or summarize, produce one concise, non-repeating "
            "bullet list. Do not restate the list, conclusion, or citations.\n"
            "When multiple excerpts come from the same source file, treat them as "
            "one source and synthesize their content together — do not list the "
            "same file multiple times as if each chunk were a separate source. "
            "Group your citation by unique file, not by individual chunk.",
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Excerpts:\n{excerpts}\n\n"
            "Answer with inline citations:",
        ),
    ]
)

QUESTION_CONDENSING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user's follow-up into a concise, standalone document "
            "search query. Resolve references such as pronouns and ordinal terms "
            "from the conversation history. Do not answer the question, add facts, "
            "or include citations. Return only the standalone query.",
        ),
        (
            "human",
            "Conversation history:\n{chat_history}\n\n"
            "Follow-up question:\n{question}\n\n"
            "Standalone query:",
        ),
    ]
)


def _clean_text(value: object, fallback: str = "") -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text or fallback


def _location_from_metadata(metadata: dict) -> str:
    if metadata.get("page_number"):
        return f"page {metadata['page_number']}"

    section_heading = _clean_text(metadata.get("section_heading"))
    if section_heading:
        return section_heading

    chunk_id = _clean_text(metadata.get("chunk_id"))
    if chunk_id:
        return f"chunk {chunk_id}"

    return "unknown section"


def _source_from_document(doc: Document) -> SourceExcerpt:
    metadata = doc.metadata or {}
    source_file = _clean_text(
        metadata.get("source_file") or metadata.get("source"),
        fallback="unknown file",
    )
    return SourceExcerpt(
        source_file=source_file,
        location=_location_from_metadata(metadata),
        chunk_id=_clean_text(metadata.get("chunk_id"), fallback="unknown chunk"),
        excerpt=_clean_text(doc.page_content),
        retrieval_distance=metadata.get(RETRIEVAL_DISTANCE_METADATA_KEY),
    )


def _format_excerpts(sources: Iterable[SourceExcerpt]) -> str:
    formatted: list[str] = []
    for index, source in enumerate(sources, start=1):
        formatted.append(
            f"[Excerpt {index}]\n"
            f"Source: {source.citation}\n"
            f"Chunk ID: {source.chunk_id}\n"
            f"Text: {source.excerpt}"
        )
    return "\n\n".join(formatted)


def _message_content(response: BaseMessage | str) -> str:
    if isinstance(response, str):
        return response

    content = response.content
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)

    return str(content)


def extract_citations(answer: str) -> list[str]:
    """Return unique inline citations in the order they appear."""

    citations: list[str] = []
    seen: set[str] = set()

    for match in CITATION_PATTERN.finditer(answer):
        citation = f"{match.group(1).strip()}{SOURCE_SEPARATOR}{match.group(2).strip()}"
        if citation not in seen:
            citations.append(citation)
            seen.add(citation)

    return citations


def _match_cited_sources(
    answer: str,
    retrieved_sources: list[SourceExcerpt],
) -> list[SourceExcerpt]:
    by_citation = {source.citation: source for source in retrieved_sources}
    return [
        by_citation[citation]
        for citation in extract_citations(answer)
        if citation in by_citation
    ]


def get_chat_llm(
    provider: str = DEFAULT_CHAT_PROVIDER,
    *,
    max_tokens: int | None = None,
) -> BaseChatModel:
    """Return the configured chat model used for retrieval QA.

    Chat/generation model selection is independent from the embedding provider.
    Keeping it centralized makes the Phase 9 OpenAI swap a small configuration
    change instead of a retrieval-chain rewrite.
    """

    load_dotenv()

    if provider == "nvidia":
        if not os.getenv("NVIDIA_API_KEY"):
            raise RuntimeError(
                "NVIDIA_API_KEY is not set. Add it to your environment or local .env "
                "file."
            )

        from langchain_nvidia_ai_endpoints import ChatNVIDIA

        nvidia_options = {
            "model": DEFAULT_NVIDIA_MODEL,
            "temperature": 0,
            "timeout": NVIDIA_REQUEST_TIMEOUT_SECONDS,
            # NVIDIA's OpenAI-compatible endpoint supports this direct
            # repetition deterrent for Llama models.
            "frequency_penalty": NVIDIA_FREQUENCY_PENALTY,
        }
        if max_tokens is not None:
            nvidia_options["max_tokens"] = max_tokens
        return ChatNVIDIA(
            **nvidia_options,
        )

    if provider == "gemini":
        if not os.getenv("GOOGLE_API_KEY"):
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Add it to your environment or local .env "
                "file. You can create a free key in Google AI Studio."
            )

        from langchain_google_genai import ChatGoogleGenerativeAI

        kwargs = {"model": DEFAULT_GEMINI_MODEL, "temperature": 0}
        if max_tokens is not None:
            kwargs["max_output_tokens"] = max_tokens
        return ChatGoogleGenerativeAI(**kwargs)

    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your environment or local .env "
                "file."
            )

        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=max_tokens)

    raise ValueError(f"Unsupported chat LLM provider: {provider}")


def get_default_llm(*, max_tokens: int | None = None) -> BaseChatModel:
    """Return the default chat model used for retrieval QA."""

    return get_chat_llm(max_tokens=max_tokens)


def condense_question(question: str, chat_history: str) -> str:
    """Resolve a follow-up into a standalone query before document retrieval.

    This deliberately uses the shared NVIDIA factory rather than constructing a
    separate chat client. With no prior turns there is nothing to resolve, so no
    extra model call is made.
    """

    if not chat_history.strip():
        return question

    messages = QUESTION_CONDENSING_PROMPT.format_messages(
        question=question,
        chat_history=chat_history,
    )
    response = get_chat_llm().invoke(messages)
    return _message_content(response).strip() or question


def _k_per_source_for_question(question: str, normal_k: int) -> int:
    """Broaden retrieval for questions that ask for document-wide coverage."""

    if ENUMERATION_QUERY_PATTERN.search(question):
        return max(normal_k, ENUMERATION_K_PER_SOURCE)
    return normal_k


def _max_tokens_for_question(question: str) -> int | None:
    """Bound long list/summary generations, where small models can loop."""

    return SUMMARY_MAX_TOKENS if ENUMERATION_QUERY_PATTERN.search(question) else None


def answer_question(
    question: str,
    *,
    k: int = 8,
    llm: Optional[BaseChatModel] = None,
    chat_history: str = "",
) -> QAResult:
    """Retrieve top chunks per source and answer with inline citations.

    ``k`` is the limit for each uploaded source file, not a global limit.
    """

    standalone_question = condense_question(question, chat_history)
    k_per_source = _k_per_source_for_question(standalone_question, k)
    retrieve = get_retriever_per_source(k_per_source=k_per_source)
    docs = retrieve(standalone_question)
    retrieved_sources = [_source_from_document(doc) for doc in docs]
    if not retrieved_sources:
        return QAResult(
            answer="I could not find relevant excerpts for that question.",
            sources=[],
            retrieved_sources=[],
        )

    chain = QA_PROMPT | (
        llm or get_default_llm(max_tokens=_max_tokens_for_question(standalone_question))
    )
    response = chain.invoke(
        {
            "question": question,
            "excerpts": _format_excerpts(retrieved_sources),
        }
    )
    answer = _message_content(response).strip()

    from chains.conflict import detect_conflict

    return QAResult(
        answer=answer,
        sources=_match_cited_sources(answer, retrieved_sources),
        retrieved_sources=retrieved_sources,
        conflict=detect_conflict(standalone_question, retrieved_sources),
    )
