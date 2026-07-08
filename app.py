"""Streamlit entry point for MultiDocChat."""

import streamlit as st

from ingestion.loaders import process_uploaded_files
from ingestion.vectorstore import (
    create_vector_store,
    make_temp_persist_directory,
    similarity_search,
)


st.set_page_config(page_title="MultiDocChat", page_icon="MD")

st.title("MultiDocChat")

uploaded_files = st.file_uploader(
    "Upload files",
    accept_multiple_files=True,
    type=["pdf", "docx", "txt", "md"],
)

prompt = st.chat_input("Ask a question about your documents")


def _upload_signature(files):
    return tuple((file.name, getattr(file, "size", None)) for file in files)


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

if prompt:
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        if "persist_directory" not in st.session_state:
            st.write("Upload documents first, then ask a question.")
        else:
            results = similarity_search(prompt, k=4)
            if not results:
                st.write("No matching chunks found.")
            else:
                st.write("Most relevant chunks:")
                for index, result in enumerate(results, start=1):
                    st.markdown(f"**Result {index}**")
                    st.json(result.metadata)
                    st.write(" ".join(result.page_content.split())[:700])
