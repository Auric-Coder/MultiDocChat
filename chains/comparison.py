"""Document comparison and diff analysis engine for MultiDocChat."""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from chains.qa_chain import get_chat_llm, _message_content


@dataclass(frozen=True)
class ComparisonResult:
    """Outcome of document comparison analysis."""

    doc1_name: str
    doc2_name: str
    additions_count: int
    deletions_count: int
    diff_lines: List[str]
    executive_summary: str


COMPARISON_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert document and contract reviewer. Compare the two document text "
            "versions and output a structured comparison report.\n"
            "Include:\n"
            "1. **Executive Summary**: Core differences between the versions.\n"
            "2. **Key Changes & Policy Modifications**: Bulleted list of added/removed/updated terms.\n"
            "3. **Financial / Risk Implications**: Impact of these changes.",
        ),
        (
            "human",
            "Document 1 ({doc1_name}):\n{doc1_text}\n\n"
            "Document 2 ({doc2_name}):\n{doc2_text}\n\n"
            "Structured Comparison Report:",
        ),
    ]
)


def compare_documents(
    doc1_text: str,
    doc2_text: str,
    doc1_name: str = "Document A",
    doc2_name: str = "Document B",
    llm: Optional[BaseChatModel] = None,
) -> ComparisonResult:
    """Compare two document texts using line diff and LLM executive analysis."""
    lines1 = [line.strip() for line in doc1_text.splitlines() if line.strip()]
    lines2 = [line.strip() for line in doc2_text.splitlines() if line.strip()]

    diff = list(
        difflib.unified_diff(
            lines1,
            lines2,
            fromfile=doc1_name,
            tofile=doc2_name,
            lineterm="",
        )
    )

    additions = sum(1 for line in diff if line.startswith("+") and not line.startswith("+++"))
    deletions = sum(1 for line in diff if line.startswith("-") and not line.startswith("---"))

    chat_model = llm or get_chat_llm()
    messages = COMPARISON_PROMPT.format_messages(
        doc1_name=doc1_name,
        doc1_text=doc1_text[:3000],  # Bound input length
        doc2_name=doc2_name,
        doc2_text=doc2_text[:3000],
    )
    response = chat_model.invoke(messages)
    summary = _message_content(response).strip()

    return ComparisonResult(
        doc1_name=doc1_name,
        doc2_name=doc2_name,
        additions_count=additions,
        deletions_count=deletions,
        diff_lines=diff,
        executive_summary=summary,
    )
