"""Streamlit entry point for MultiDocChat."""

from collections import Counter

import streamlit as st

from chains.qa_chain import DEFAULT_CHAT_PROVIDER, answer_question
from ingestion.loaders import CHUNK_SIZE, process_uploaded_files
from ingestion.vectorstore import (
    EMBEDDING_PROVIDER,
    create_vector_store,
    get_vector_store,
    make_temp_persist_directory,
)
from memory.session import (
    add_conversation_turn,
    clear_conversation,
    format_chat_history,
    get_conversation_turns,
    initialize_conversation,
)


SUPPORTED_FILE_TYPES = ["pdf", "docx", "txt", "md"]
INDEX_STATE_KEYS = (
    "upload_signature",
    "chunks",
    "chunk_count",
    "persist_directory",
    "indexed_files",
    "response_details",
)


def _upload_signature(files):
    return tuple((file.name, getattr(file, "size", None)) for file in files)


def _vector_store_is_active():
    try:
        get_vector_store()
    except RuntimeError:
        return False
    return True


def _clear_index():
    """Clear UI-owned document and conversation state for a fresh demo."""

    for key in INDEX_STATE_KEYS:
        st.session_state.pop(key, None)
    clear_conversation(st.session_state)
    st.session_state.upload_widget_version += 1


def _source_details(sources):
    return [
        {
            "citation": source.citation,
            "chunk_id": source.chunk_id,
            "excerpt": source.excerpt,
        }
        for source in sources
    ]


def _render_sources(details, *, cited):
    with st.expander("Sources", icon=":material/format_quote:"):
        if cited:
            st.caption("Excerpts cited in the answer.")
        elif details:
            st.caption("No inline citations were parsed; showing retrieved context.")
        else:
            st.write("No source excerpts found.")

        for index, source in enumerate(details, start=1):
            st.markdown(f"**{index}. {source['citation']}**")
            st.caption(f"Chunk ID: {source['chunk_id']}")
            st.write(source["excerpt"][:900])


def _render_assistant_response(answer, details=None):
    st.markdown(answer)
    if not details:
        return

    if details.get("conflict_summary"):
        st.warning(
            "Sources disagree\n\n" + details["conflict_summary"],
            icon=":material/warning:",
        )
    _render_sources(details["sources"], cited=details["has_citations"])


st.set_page_config(
    page_title="MultiDocChat",
    page_icon=":material/forum:",
    layout="centered",
)

initialize_conversation(st.session_state)
st.session_state.setdefault("upload_widget_version", 0)
st.session_state.setdefault("response_details", [])

with st.sidebar:
    st.header("Demo settings")
    st.segmented_control(
        "Chat model",
        [DEFAULT_CHAT_PROVIDER.title()],
        default=DEFAULT_CHAT_PROVIDER.title(),
        disabled=True,
        help="Configured centrally in chains/qa_chain.py.",
        width="stretch",
    )
    st.segmented_control(
        "Embeddings",
        [EMBEDDING_PROVIDER.title()],
        default=EMBEDDING_PROVIDER.title(),
        disabled=True,
        help="Configured centrally in ingestion/vectorstore.py.",
        width="stretch",
    )
    st.slider(
        "Chunk size",
        min_value=300,
        max_value=1_500,
        value=CHUNK_SIZE,
        step=50,
        disabled=True,
        help="The current ingestion setting. Changing it requires re-chunking.",
    )
    retrieval_k = st.slider(
        "Retrieved chunks per file",
        min_value=1,
        max_value=10,
        value=4,
        help="Top chunks retrieved independently for each indexed file.",
    )
    st.caption("Model, embeddings, and chunking stay fixed for this phase; k is live.")

    if st.button(
        "Clear indexed files and chat",
        icon=":material/restart_alt:",
        width="stretch",
        disabled=not bool(st.session_state.get("persist_directory")),
    ):
        _clear_index()
        st.rerun()

st.title("MultiDocChat")
st.caption("Ask attributed questions across your documents and follow up in context.")

with st.container(border=True):
    uploaded_files = st.file_uploader(
        "Upload documents to index",
        accept_multiple_files=True,
        type=SUPPORTED_FILE_TYPES,
        key=f"uploaded_files_{st.session_state.upload_widget_version}",
        help="Upload PDF, DOCX, TXT, or Markdown files. Replacing the selection reindexes the collection.",
    )

if uploaded_files:
    upload_signature = _upload_signature(uploaded_files)

    try:
        if st.session_state.get("upload_signature") != upload_signature:
            chunks = process_uploaded_files(uploaded_files)
            persist_directory = make_temp_persist_directory()
            create_vector_store(chunks, persist_directory=persist_directory)

            st.session_state.upload_signature = upload_signature
            st.session_state.chunks = chunks
            st.session_state.chunk_count = len(chunks)
            st.session_state.persist_directory = persist_directory
            st.session_state.indexed_files = [file.name for file in uploaded_files]
            st.session_state.response_details = []
            clear_conversation(st.session_state)
        elif not _vector_store_is_active():
            create_vector_store(
                st.session_state.chunks,
                persist_directory=st.session_state.persist_directory,
            )
    except Exception as exc:
        st.error(f"Could not process uploaded files: {exc}")

indexed_files = st.session_state.get("indexed_files", [])
if indexed_files:
    file_counts = Counter(
        chunk.metadata.get("source_file", "Unknown file")
        for chunk in st.session_state.get("chunks", [])
    )
    with st.container(border=True):
        st.subheader("Indexed files", anchor=False)
        st.caption(
            f"{len(indexed_files)} files · {st.session_state.get('chunk_count', 0)} chunks ready for chat"
        )
        for filename in indexed_files:
            st.write(f":material/description: **{filename}** — {file_counts[filename]} chunks")

for index, turn in enumerate(get_conversation_turns(st.session_state)):
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant", avatar=":material/smart_toy:"):
        details = st.session_state.response_details[index] if index < len(st.session_state.response_details) else None
        _render_assistant_response(turn["answer"], details)

prompt = st.chat_input(
    "Ask a question about the indexed documents",
    disabled="persist_directory" not in st.session_state,
    submit_mode="disable",
)

if not indexed_files:
    st.info("Upload one or more documents to begin a cited conversation.", icon=":material/upload_file:")

if prompt:
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant", avatar=":material/smart_toy:"):
        try:
            with st.spinner("Searching your documents…"):
                result = answer_question(
                    prompt,
                    k=retrieval_k,
                    chat_history=format_chat_history(
                        get_conversation_turns(st.session_state)
                    ),
                )
        except Exception as exc:
            st.error(f"Could not generate an answer: {exc}")
        else:
            sources_to_show = result.sources or result.retrieved_sources
            details = {
                "sources": _source_details(sources_to_show),
                "has_citations": bool(result.sources),
                "conflict_summary": (
                    result.conflict.summary
                    or "\n".join(result.conflict.source_positions())
                    if result.conflict and result.conflict.has_conflict
                    else ""
                ),
            }
            _render_assistant_response(result.answer, details)
            add_conversation_turn(st.session_state, prompt, result.answer)
            retained_turn_count = len(get_conversation_turns(st.session_state))
            st.session_state.response_details = (
                st.session_state.response_details + [details]
            )[-retained_turn_count:]
