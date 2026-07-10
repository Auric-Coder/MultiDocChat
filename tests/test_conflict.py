"""Phase 4 regression test: contradictory policy files produce a callout."""

from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from langchain_core.messages import AIMessage

from chains.conflict import detect_conflict
from chains.qa_chain import SourceExcerpt, _source_from_document
from ingestion.loaders import load_and_chunk


class _FakeGemini:
    def invoke(self, _messages):
        return AIMessage(
            content=(
                "CONFLICT: YES\n"
                "- policy_2025.txt: allows $100 per month.\n"
                "- policy_2026.txt: allows $150 per month."
            )
        )


class ConflictDetectionTest(TestCase):
    def test_contradictory_policy_files_trigger_gemini_conflict_call(self):
        root = Path(__file__).resolve().parents[1]
        sources = [
            _source_from_document(document)
            for path in (root / "samples" / "policy_2025.txt", root / "samples" / "policy_2026.txt")
            for document in load_and_chunk(str(path))
        ]

        with patch("chains.qa_chain.get_chat_llm", return_value=_FakeGemini()) as factory:
            result = detect_conflict("What is the monthly internet reimbursement?", sources)

        self.assertTrue(result.checked)
        self.assertTrue(result.has_conflict)
        self.assertEqual(factory.call_args.kwargs, {"provider": "gemini"})
        self.assertIn("policy_2025.txt", result.summary)
        self.assertIn("policy_2026.txt", result.summary)

    def test_matching_multi_file_sources_do_not_spend_a_conflict_call(self):
        sources = [
            SourceExcerpt("one.txt", "chunk 1", "one-1", "The limit is $100 per month."),
            SourceExcerpt("two.txt", "chunk 1", "two-1", "The limit is $100 per month."),
        ]

        with patch("chains.qa_chain.get_chat_llm") as factory:
            result = detect_conflict("What is the monthly limit?", sources)

        self.assertFalse(result.checked)
        factory.assert_not_called()
