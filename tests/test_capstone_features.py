"""Unit tests for Phase C capstone features: RAG metrics, comparison, and export."""

from unittest import TestCase
from unittest.mock import MagicMock
from langchain_core.documents import Document

from eval.metrics import (
    evaluate_faithfulness,
    evaluate_answer_relevancy,
    evaluate_context_precision,
    evaluate_context_recall,
    run_rag_evaluation,
)
from chains.comparison import compare_documents, ComparisonResult
from export.report import export_chat_report_markdown, export_chat_report_html


class RAGMetricsTest(TestCase):
    def test_faithfulness_calculation(self):
        chunks = [Document(page_content="Python is a programming language used for machine learning.")]
        answer = "Python is a programming language."
        score = evaluate_faithfulness(answer, chunks)
        self.assertGreater(score, 0.5)

    def test_honest_fallback_faithfulness(self):
        chunks = [Document(page_content="Some content.")]
        answer = "I could not find relevant excerpts for that question."
        score = evaluate_faithfulness(answer, chunks)
        self.assertEqual(score, 1.0)

    def test_answer_relevancy(self):
        question = "What is Python?"
        answer = "Python is a high-level programming language."
        score = evaluate_answer_relevancy(question, answer)
        self.assertGreater(score, 0.0)

    def test_context_precision(self):
        question = "Python programming"
        chunks = [
            Document(page_content="Python is great for programming."),
            Document(page_content="Unrelated text about cooking recipes."),
        ]
        precision = evaluate_context_precision(question, chunks)
        self.assertEqual(precision, 0.5)

    def test_context_recall(self):
        expected = "Python machine learning"
        chunks = [Document(page_content="Python is widely used in machine learning.")]
        recall = evaluate_context_recall(expected, chunks)
        self.assertEqual(recall, 1.0)

    def test_run_rag_evaluation(self):
        results = run_rag_evaluation(
            question="What is Python?",
            answer="Python is a programming language.",
            retrieved_sources=[Document(page_content="Python is a programming language.")],
            expected="Python language",
        )
        self.assertIn("rag_score", results)
        self.assertGreater(results["rag_score"], 0.5)


class ComparisonEngineTest(TestCase):
    def test_compare_documents(self):
        doc1 = "The home internet reimbursement is $100 per month."
        doc2 = "The home internet reimbursement is $150 per month."

        fake_llm = MagicMock()
        fake_llm.invoke.return_value = MagicMock(content="Reimbursement increased from $100 to $150.")

        res = compare_documents(doc1, doc2, "Policy 2025", "Policy 2026", llm=fake_llm)

        self.assertIsInstance(res, ComparisonResult)
        self.assertEqual(res.doc1_name, "Policy 2025")
        self.assertEqual(res.doc2_name, "Policy 2026")
        self.assertIn("increased", res.executive_summary)


class ReportExportTest(TestCase):
    def test_export_markdown(self):
        turns = [{"question": "What is Python?", "answer": "Python is a language."}]
        details = [{
            "sources": [{"citation": "doc.txt — page 1", "chunk_id": "doc-01"}],
            "confidence_score": 0.9,
            "confidence_label": "High",
        }]
        files = ["doc.txt"]

        md = export_chat_report_markdown(turns, details, files)
        self.assertIn("MultiDocChat", md)
        self.assertIn("What is Python?", md)
        self.assertIn("doc.txt", md)

    def test_export_html(self):
        turns = [{"question": "What is Python?", "answer": "Python is a language."}]
        details = []
        files = []

        html = export_chat_report_html(turns, details, files)
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("MultiDocChat", html)
