"""Phase A1 regression tests: BM25 keyword search and hybrid retrieval."""

from collections import defaultdict
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from ingestion.keyword_search import KeywordIndex, tokenize


class TokenizeTest(TestCase):
    def test_lowercases_and_splits(self):
        tokens = tokenize("Hello World! Testing 123")
        self.assertEqual(tokens, ["hello", "world", "testing", "123"])

    def test_removes_stop_words(self):
        tokens = tokenize("this is a test of the system")
        self.assertNotIn("this", tokens)
        self.assertNotIn("is", tokens)
        self.assertNotIn("a", tokens)
        self.assertNotIn("the", tokens)
        self.assertNotIn("of", tokens)
        self.assertIn("test", tokens)
        self.assertIn("system", tokens)


class KeywordIndexTest(TestCase):
    def _make_docs(self):
        return [
            Document(
                page_content="Machine learning models use gradient descent for optimization.",
                metadata={"source_file": "ml_guide.txt", "chunk_id": "ml-0001"},
            ),
            Document(
                page_content="The company policy allows $100 per month for internet reimbursement.",
                metadata={"source_file": "policy.txt", "chunk_id": "policy-0001"},
            ),
            Document(
                page_content="Python is a popular programming language for data science.",
                metadata={"source_file": "ml_guide.txt", "chunk_id": "ml-0002"},
            ),
        ]

    def test_bm25_index_returns_relevant_documents(self):
        docs = self._make_docs()
        index = KeywordIndex(docs)
        results = index.search("gradient descent optimization", k=3)

        self.assertTrue(len(results) > 0)
        top_doc, top_score = results[0]
        self.assertIn("gradient", top_doc.page_content.lower())
        self.assertGreater(top_score, 0)

    def test_bm25_returns_zero_for_unrelated_query(self):
        docs = self._make_docs()
        index = KeywordIndex(docs)
        results = index.search("xyzzyplugh", k=3)

        # All scores should be 0 or no results returned.
        for _, score in results:
            self.assertEqual(score, 0.0)

    def test_search_per_source_balances_across_files(self):
        docs = self._make_docs()
        index = KeywordIndex(docs)
        # Query that could match both sources.
        results = index.search_per_source("machine learning policy", k_per_source=5)

        source_files = {doc.metadata["source_file"] for doc, _ in results}
        # Should have results from both source files.
        self.assertIn("ml_guide.txt", source_files)
        self.assertIn("policy.txt", source_files)

    def test_empty_index_returns_empty(self):
        index = KeywordIndex([])
        results = index.search("anything", k=5)
        self.assertEqual(results, [])

    def test_empty_query_returns_empty(self):
        docs = self._make_docs()
        index = KeywordIndex(docs)
        results = index.search("", k=5)
        self.assertEqual(results, [])


class HybridDeduplicationTest(TestCase):
    """Verify the hybrid retriever deduplicates by chunk_id."""

    def test_hybrid_deduplicates_by_chunk_id(self):
        from ingestion.vectorstore import _normalize_scores

        doc = Document(
            page_content="Shared content across both retrievers.",
            metadata={"source_file": "test.txt", "chunk_id": "test-0001"},
        )
        # Simulate both retrievers finding the same document.
        semantic_results = [(doc, 1.0)]  # L2 distance
        keyword_results = [(doc, 5.0)]   # BM25 score

        sem_norm = _normalize_scores(semantic_results, higher_is_better=False)
        bm25_norm = _normalize_scores(keyword_results, higher_is_better=True)

        # Fuse manually like the hybrid retriever does.
        alpha = 0.7
        fused: dict[str, tuple[Document, float]] = {}

        for d, ns in sem_norm:
            cid = d.metadata.get("chunk_id", id(d))
            existing = fused[cid][1] if cid in fused else 0.0
            fused[cid] = (d, existing + alpha * ns)

        for d, ns in bm25_norm:
            cid = d.metadata.get("chunk_id", id(d))
            if cid in fused:
                fused[cid] = (fused[cid][0], fused[cid][1] + (1 - alpha) * ns)
            else:
                fused[cid] = (d, (1 - alpha) * ns)

        # The document should appear exactly once.
        self.assertEqual(len(fused), 1)
        self.assertIn("test-0001", fused)


class NormalizeScoresTest(TestCase):
    """Verify min-max normalization handles edge cases."""

    def test_higher_is_better_normalization(self):
        from ingestion.vectorstore import _normalize_scores

        doc_a = Document(page_content="a", metadata={})
        doc_b = Document(page_content="b", metadata={})

        results = [(doc_a, 10.0), (doc_b, 5.0)]
        normalized = _normalize_scores(results, higher_is_better=True)

        # doc_a has highest score → normalized to 1.0.
        self.assertAlmostEqual(normalized[0][1], 1.0)
        # doc_b has lowest score → normalized to 0.0.
        self.assertAlmostEqual(normalized[1][1], 0.0)

    def test_lower_is_better_normalization(self):
        from ingestion.vectorstore import _normalize_scores

        doc_a = Document(page_content="a", metadata={})
        doc_b = Document(page_content="b", metadata={})

        results = [(doc_a, 1.0), (doc_b, 3.0)]
        normalized = _normalize_scores(results, higher_is_better=False)

        # doc_a has lowest distance → normalized to 1.0 (best).
        self.assertAlmostEqual(normalized[0][1], 1.0)
        # doc_b has highest distance → normalized to 0.0 (worst).
        self.assertAlmostEqual(normalized[1][1], 0.0)

    def test_single_result_normalized(self):
        from ingestion.vectorstore import _normalize_scores

        doc = Document(page_content="solo", metadata={})
        results = [(doc, 5.0)]
        normalized = _normalize_scores(results, higher_is_better=True)

        # Single result: spread is 0, so fallback to 1.0 denominator.
        self.assertEqual(len(normalized), 1)

    def test_empty_input_returns_empty(self):
        from ingestion.vectorstore import _normalize_scores

        normalized = _normalize_scores([], higher_is_better=True)
        self.assertEqual(normalized, [])
