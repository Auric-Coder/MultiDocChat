from os import environ
from unittest import TestCase
from unittest.mock import patch

from chains.qa_chain import (
    THINKING_OFF_SYSTEM_MESSAGE,
    _message_content,
    _nvidia_ssl_verify,
    _prepend_thinking_off_message,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


class ResponseCleaningTest(TestCase):
    def test_nvidia_ssl_verification_defaults_to_enabled(self):
        with patch.dict(environ, {}, clear=True):
            self.assertTrue(_nvidia_ssl_verify())

    def test_nvidia_ssl_verification_can_use_a_local_override(self):
        with patch.dict(environ, {"NVIDIA_SSL_VERIFY": "false"}, clear=True):
            self.assertFalse(_nvidia_ssl_verify())

    def test_bare_thinking_delimiter_keeps_only_final_answer(self):
        response = AIMessage(
            content="I should compare the excerpts first.</think>\nThe limit is $150."
        )

        self.assertEqual(_message_content(response), "The limit is $150.")

    def test_reasoning_content_block_is_not_rendered(self):
        response = AIMessage(content=[
            {"type": "reasoning", "text": "Private analysis"},
            {"type": "text", "text": "The limit is $150."},
        ])

        self.assertEqual(_message_content(response), "The limit is $150.")

    def test_thinking_off_message_is_prepended_before_existing_system_messages(self):
        messages = [
            SystemMessage(content="Answer only from the excerpts."),
            HumanMessage(content="What is the limit?"),
        ]

        result = _prepend_thinking_off_message(messages)

        self.assertEqual(result[0].content, THINKING_OFF_SYSTEM_MESSAGE)
        self.assertEqual(result[1:], messages)

    def test_thinking_tags_are_removed_from_answer_text(self):
        response = AIMessage(
            content="<think>Private reasoning.</think>\nThe limit is $150."
        )

        self.assertEqual(_message_content(response), "The limit is $150.")
