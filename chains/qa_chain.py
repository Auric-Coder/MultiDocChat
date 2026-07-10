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

from ingestion.vectorstore import get_retriever_per_source

if TYPE_CHECKING:
    from chains.conflict import ConflictResult


DEFAULT_CHAT_PROVIDER = "nvidia"
DEFAULT_NVIDIA_MODEL = "meta/llama-3.3-70b-instruct"
NVIDIA_REQUEST_TIMEOUT_SECONDS = 120
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
EM_DASH = "\N{EM DASH}"
CITATION_FORMAT = f"[filename {EM_DASH} section/page]"
SOURCE_SEPARATOR = f" {EM_DASH} "
CITATION_PATTERN = re.compile(
    r"\[([^\[\]]+?)\s+(?:" + re.escape(EM_DASH) + r"|-)\s+([^\[\]]+?)\]"
)


@dataclass(frozen=True)
class SourceExcerpt:
    """A retrieved excerpt labeled for citation and UI display."""

    source_file: str
    location: str
    chunk_id: str
    excerpt: str

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
            "Do not cite sources that are not present in the excerpts. Keep the "
            "answer concise and directly relevant to the question.",
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Excerpts:\n{excerpts}\n\n"
            "Answer with inline citations:",
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


def get_chat_llm(provider: str = DEFAULT_CHAT_PROVIDER) -> BaseChatModel:
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

        return ChatNVIDIA(
            model=DEFAULT_NVIDIA_MODEL,
            temperature=0,
            timeout=NVIDIA_REQUEST_TIMEOUT_SECONDS,
        )

    if provider == "gemini":
        if not os.getenv("GOOGLE_API_KEY"):
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Add it to your environment or local .env "
                "file. You can create a free key in Google AI Studio."
            )

        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=DEFAULT_GEMINI_MODEL, temperature=0)

    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your environment or local .env "
                "file."
            )

        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model="gpt-4o-mini", temperature=0)

    raise ValueError(f"Unsupported chat LLM provider: {provider}")


def get_default_llm() -> BaseChatModel:
    """Return the default chat model used for retrieval QA."""

    return get_chat_llm(DEFAULT_CHAT_PROVIDER)


def answer_question(
    question: str,
    *,
    k: int = 4,
    llm: Optional[BaseChatModel] = None,
) -> QAResult:
    """Retrieve top chunks per source and answer with inline citations.

    ``k`` is the limit for each uploaded source file, not a global limit.
    """

    retrieve = get_retriever_per_source(k_per_source=k)
    docs = retrieve(question)
    retrieved_sources = [_source_from_document(doc) for doc in docs]
    if not retrieved_sources:
        return QAResult(
            answer="I could not find relevant excerpts for that question.",
            sources=[],
            retrieved_sources=[],
        )

    chain = QA_PROMPT | (llm or get_default_llm())
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
        conflict=detect_conflict(question, retrieved_sources),
    )
