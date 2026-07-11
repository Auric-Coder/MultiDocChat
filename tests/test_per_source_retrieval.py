"""Regression coverage for multi-file retrieval balance."""

from unittest import TestCase
from unittest.mock import patch

from langchain_core.documents import Document

from chains.qa_chain import QA_PROMPT, answer_question
from ingestion.vectorstore import get_retriever_per_source


class _FakeVectorStore:
    def __init__(self):
        self.calls = []

    def get(self):
        return {
            "metadatas": [
                {"source_file": "large_document.txt"},
                {"source_file": "resume.txt"},
                {"source_file": "large_document.txt"},
            ]
        }

    def similarity_search(self, query, *, k, filter):
        self.calls.append((query, k, filter))
        source = filter["source_file"]
        return [Document(page_content=source, metadata={"source_file": source})]


class PerSourceRetrievalTest(TestCase):
    def test_retrieves_each_source_independently_in_stable_order(self):
        store = _FakeVectorStore()

        with patch("ingestion.vectorstore.get_vector_store", return_value=store):
            results = get_retriever_per_source(k_per_source=3)("Who is the candidate?")

        self.assertEqual([result.page_content for result in results], ["large_document.txt", "resume.txt"])
        self.assertEqual(
            store.calls,
            [
                ("Who is the candidate?", 3, {"source_file": "large_document.txt"}),
                ("Who is the candidate?", 3, {"source_file": "resume.txt"}),
            ],
        )

    def test_answer_question_uses_per_source_retriever(self):
        resume = Document(
            page_content="Ada is the candidate.",
            metadata={"source_file": "resume.txt", "chunk_id": "resume-1"},
        )
        large = Document(
            page_content="A large unrelated document.",
            metadata={"source_file": "large_document.txt", "chunk_id": "large-1"},
        )

        llm = lambda _messages: "The candidate is named Ada. [resume.txt - chunk resume-1]"
        with patch("chains.qa_chain.get_retriever_per_source", return_value=lambda _query: [large, resume]) as retriever:
            with patch("chains.conflict.detect_conflict", return_value=None):
                result = answer_question("Who is the candidate?", k=3, llm=llm)

        retriever.assert_called_once_with(k_per_source=3)
        self.assertEqual([source.source_file for source in result.retrieved_sources], ["large_document.txt", "resume.txt"])
        self.assertEqual([source.source_file for source in result.sources], ["resume.txt"])

    def test_enumeration_question_broadens_per_source_retrieval(self):
        document = Document(
            page_content="Phase 1 is ingestion.",
            metadata={"source_file": "roadmap.txt", "chunk_id": "roadmap-1"},
        )
        llm = lambda _messages: "Phase 1 is ingestion. [roadmap.txt - chunk roadmap-1]"

        with patch(
            "chains.qa_chain.get_retriever_per_source",
            return_value=lambda _query: [document],
        ) as retriever:
            with patch("chains.conflict.detect_conflict", return_value=None):
                answer_question("List all phases.", k=5, llm=llm)

        retriever.assert_called_once_with(k_per_source=20)

    def test_qa_prompt_instructs_file_level_source_deduplication(self):
        system_message = QA_PROMPT.messages[0].prompt.template

        self.assertIn("treat them as one source", system_message)
        self.assertIn("Group your citation by unique file", system_message)
