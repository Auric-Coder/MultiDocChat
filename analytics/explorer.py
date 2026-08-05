"""Document chunk explorer and full-text browser for MultiDocChat."""

from __future__ import annotations

from typing import Sequence
import streamlit as st
from langchain_core.documents import Document


def render_document_explorer(chunks: Sequence[Document]) -> None:
    """Render document chunk browser and full-text search interface."""
    st.subheader("Document Chunk Explorer", anchor=False)

    if not chunks:
        st.info("Upload documents to explore indexed chunks.", icon=":material/explore:")
        return

    search_term = st.text_input(
        "🔎 Search within indexed chunks:",
        placeholder="Type a word or phrase to filter chunks...",
    )

    filtered_chunks = chunks
    if search_term.strip():
        term = search_term.strip().lower()
        filtered_chunks = [
            chunk for chunk in chunks if term in chunk.page_content.lower()
        ]
        st.caption(f"Found {len(filtered_chunks)} matching chunks out of {len(chunks)} total.")

    if not filtered_chunks:
        st.warning(f"No chunks found containing '{search_term}'.")
        return

    # --- Pagination ---
    chunks_per_page = 5
    total_pages = (len(filtered_chunks) - 1) // chunks_per_page + 1

    if total_pages > 1:
        col_page, _ = st.columns([2, 5])
        with col_page:
            page = st.selectbox("Page", range(1, total_pages + 1), key="explorer_page")
    else:
        page = 1

    start_idx = (page - 1) * chunks_per_page
    end_idx = min(start_idx + chunks_per_page, len(filtered_chunks))

    for idx in range(start_idx, end_idx):
        chunk = filtered_chunks[idx]
        metadata = chunk.metadata or {}
        source_file = metadata.get("source_file", "Unknown file")
        chunk_id = metadata.get("chunk_id", f"chunk-{idx}")
        page_num = metadata.get("page_number")
        section = metadata.get("section_heading")

        location = f"Page {page_num}" if page_num else (section or "Chunk")

        with st.expander(f"📄 [{source_file}] {location} — {chunk_id}"):
            st.markdown(f"**Content:** ({len(chunk.page_content)} characters)")
            st.text_area(
                "Chunk text",
                value=chunk.page_content,
                height=150,
                disabled=True,
                key=f"chunk_text_{chunk_id}_{idx}",
            )

            st.markdown("**Metadata:**")
            st.json(metadata)
