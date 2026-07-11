"""Phase 5 regression coverage for windowed follow-up retrieval."""

from unittest import TestCase
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from chains.qa_chain import answer_question
from memory.session import add_conversation_turn, format_chat_history


class _CondensingNvidia:
    def invoke(self, messages):
        prompt = "\n".join(message.content for message in messages)
        if "Follow-up question:" in prompt:
            return AIMessage(content="What is the second qualification listed for Ada?")
        return AIMessage(
            content="Ada's second qualification is a master's degree. "
            "[resume.txt - Education]"
        )


class ConversationMemoryTest(TestCase):
    def test_history_is_windowed_to_the_last_six_completed_turns(self):
        state = {}
        for index in range(7):
            add_conversation_turn(state, f"question {index}", f"answer {index}")

        history = format_chat_history(state["conversation_turns"])
        self.assertNotIn("question 0", history)
        self.assertIn("question 6", history)
        self.assertEqual(len(state["conversation_turns"]), 6)

    def test_third_turn_follow_up_is_condensed_before_retrieval(self):
        document = Document(
            page_content="Ada's second qualification is a master's degree.",
            metadata={
                "source_file": "resume.txt",
                "section_heading": "Education",
                "chunk_id": "resume-education-2",
            },
        )
        history = (
            "User: Tell me about Ada's qualifications.\n"
            "Assistant: Ada has a bachelor's degree and a master's degree. "
            "[resume.txt - Education]\n\n"
            "User: Which qualification came first?\n"
            "Assistant: Her bachelor's degree came first. [resume.txt - Education]"
        )
        nvidia = _CondensingNvidia()

        retrieved_queries = []

        def retrieve(query):
            retrieved_queries.append(query)
            return [document]

        answer_llm = lambda _messages: (
            "Ada's second qualification is a master's degree. "
            "[resume.txt - Education]"
        )

        with patch("chains.qa_chain.get_chat_llm", return_value=nvidia) as factory:
            with patch(
                "chains.qa_chain.get_retriever_per_source",
                return_value=retrieve,
            ) as retriever:
                with patch("chains.conflict.detect_conflict", return_value=None):
                    result = answer_question(
                        "What about the second one?", chat_history=history, llm=answer_llm
                    )

        factory.assert_called_once_with(provider="nvidia")
        retriever.assert_called_once_with(k_per_source=4)
        self.assertEqual(
            retrieved_queries, ["What is the second qualification listed for Ada?"]
        )
        self.assertEqual(result.sources[0].citation, "resume.txt — Education")
        self.assertIn("[resume.txt - Education]", result.answer)
