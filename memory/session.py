"""Windowed conversation helpers for the Streamlit session.

The stored records are plain dictionaries so they can live safely in
``st.session_state`` and be exercised without importing Streamlit in tests.
"""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence


CONVERSATION_STATE_KEY = "conversation_turns"
MAX_CONVERSATION_TURNS = 6


def initialize_conversation(
    session_state: MutableMapping[str, object],
) -> None:
    """Ensure the per-session, windowed conversation store exists."""

    session_state.setdefault(CONVERSATION_STATE_KEY, [])


def get_conversation_turns(
    session_state: MutableMapping[str, object],
) -> list[dict[str, str]]:
    """Return a copy of the retained completed turns."""

    initialize_conversation(session_state)
    return list(session_state[CONVERSATION_STATE_KEY])


def format_chat_history(turns: Sequence[dict[str, str]]) -> str:
    """Format completed turns for the question-condensing prompt."""

    return "\n\n".join(
        "User: {question}\nAssistant: {answer}".format(
            question=turn["question"], answer=turn["answer"]
        )
        for turn in turns
    )


def add_conversation_turn(
    session_state: MutableMapping[str, object],
    question: str,
    answer: str,
    *,
    max_turns: int = MAX_CONVERSATION_TURNS,
) -> None:
    """Append a completed exchange, retaining only the configured window."""

    if max_turns < 1:
        raise ValueError("max_turns must be at least 1")

    turns = get_conversation_turns(session_state)
    turns.append({"question": question, "answer": answer})
    session_state[CONVERSATION_STATE_KEY] = turns[-max_turns:]


def clear_conversation(session_state: MutableMapping[str, object]) -> None:
    """Discard history when the document collection changes."""

    session_state[CONVERSATION_STATE_KEY] = []
