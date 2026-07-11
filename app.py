"""Streamlit entry point for MultiDocChat."""

import streamlit as st

from chains.qa_chain import answer_question
from ingestion.loaders import process_uploaded_files
from ingestion.vectorstore import (
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


st.set_page_config(page_title="MultiDocChat", page_icon="MD")

st.title("MultiDocChat")
initialize_conversation(st.session_state)

uploaded_files = st.file_uploader(
    "Upload files",
    accept_multiple_files=True,
    type=["pdf", "docx", "txt", "md"],
)

def _upload_signature(files):
    return tuple((file.name, getattr(file, "size", None)) for file in files)


def _vector_store_is_active():
    try:
        get_vector_store()
    except RuntimeError:
        return False
    return True


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
            clear_conversation(st.session_state)
        elif not _vector_store_is_active():
            create_vector_store(
                st.session_state.chunks,
                persist_directory=st.session_state.persist_directory,
            )

        chunks = st.session_state.chunks
    except Exception as exc:
        st.error(f"Could not process uploaded files: {exc}")
    else:
        st.success(
            f"Embedded and stored {st.session_state.chunk_count} chunks in local Chroma."
        )
        with st.expander("Inspect chunk metadata"):
            for chunk in chunks:
                st.json(chunk.metadata)
                st.caption(" ".join(chunk.page_content.split())[:300])


for turn in get_conversation_turns(st.session_state):
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])

prompt = st.chat_input("Ask a question about your documents")

if prompt:
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        if "persist_directory" not in st.session_state:
            st.write("Upload documents first, then ask a question.")
        else:
            try:
                result = answer_question(
                    prompt,
                    k=4,
                    chat_history=format_chat_history(
                        get_conversation_turns(st.session_state)
                    ),
                )
            except Exception as exc:
                st.error(f"Could not generate an answer: {exc}")
            else:
                st.write(result.answer)
                add_conversation_turn(st.session_state, prompt, result.answer)

                if result.conflict and result.conflict.has_conflict:
                    st.warning("⚠️ Sources disagree")
                    st.markdown(
                        result.conflict.summary
                        or "\n".join(result.conflict.source_positions())
                    )

                sources_to_show = result.sources or result.retrieved_sources
                with st.expander("Sources"):
                    if result.sources:
                        st.caption("Excerpts cited in the answer.")
                    elif result.retrieved_sources:
                        st.caption(
                            "No inline citations were parsed, so showing the "
                            "retrieved excerpts used as context."
                        )
                    else:
                        st.write("No source excerpts found.")

                    for index, source in enumerate(sources_to_show, start=1):
                        st.markdown(f"**{index}. {source.citation}**")
                        st.caption(f"Chunk ID: {source.chunk_id}")
                        st.write(source.excerpt[:900])
