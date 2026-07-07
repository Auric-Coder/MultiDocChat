"""Streamlit entry point for MultiDocChat."""

import streamlit as st

from ingestion.loaders import process_uploaded_files


st.set_page_config(page_title="MultiDocChat", page_icon="MD")

st.title("MultiDocChat")

uploaded_files = st.file_uploader(
    "Upload files",
    accept_multiple_files=True,
    type=["pdf", "docx", "txt", "md"],
)

prompt = st.chat_input("Ask a question about your documents")

if uploaded_files:
    try:
        chunks = process_uploaded_files(uploaded_files)
    except Exception as exc:
        st.error(f"Could not process uploaded files: {exc}")
    else:
        st.success(f"Processed {len(chunks)} chunks with attribution metadata.")
        with st.expander("Inspect chunk metadata"):
            for chunk in chunks:
                st.json(chunk.metadata)
                st.caption(" ".join(chunk.page_content.split())[:300])

if prompt:
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        st.write("Chat backend will be added in a later phase.")
