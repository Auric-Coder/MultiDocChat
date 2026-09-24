"""Document and retrieval analytics dashboard components for MultiDocChat."""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from langchain_core.documents import Document


THEME_COLORS = {
    "background": "#1D211C",
    "paper_bg": "#1D211C",
    "text": "#F1EDE2",
    "primary": "#C99162",
    "secondary": "#D3A173",
    "grid": "#363A31",
}


def _apply_chart_layout(fig: go.Figure, title: str) -> go.Figure:
    """Apply editorial dark-mode styling to Plotly figures."""
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family="Fraunces, serif", size=18, color=THEME_COLORS["text"]),
        ),
        paper_bgcolor=THEME_COLORS["paper_bg"],
        plot_bgcolor=THEME_COLORS["background"],
        font=dict(family="Manrope, sans-serif", color=THEME_COLORS["text"]),
        xaxis=dict(
            gridcolor=THEME_COLORS["grid"],
            zerolinecolor=THEME_COLORS["grid"],
            tickfont=dict(color=THEME_COLORS["text"]),
        ),
        yaxis=dict(
            gridcolor=THEME_COLORS["grid"],
            zerolinecolor=THEME_COLORS["grid"],
            tickfont=dict(color=THEME_COLORS["text"]),
        ),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def render_analytics_dashboard(
    chunks: Sequence[Document],
    indexed_files: Sequence[str],
    response_details: Sequence[dict[str, Any]],
    conversation_turns: Sequence[dict[str, str]],
) -> None:
    """Render full analytics dashboard in Streamlit."""
    st.subheader("Document Collection Analytics", anchor=False)

    if not chunks:
        st.info("Upload documents to view collection analytics.", icon=":material/analytics:")
        return

    # --- Top KPI Metrics ---
    total_chunks = len(chunks)
    total_files = len(indexed_files)
    total_chars = sum(len(chunk.page_content) for chunk in chunks)
    avg_chunk_len = round(total_chars / total_chunks) if total_chunks else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Indexed Files", total_files)
    with col2:
        st.metric("Total Chunks", total_chunks)
    with col3:
        st.metric("Total Characters", f"{total_chars:,}")
    with col4:
        st.metric("Avg Chunk Size", f"{avg_chunk_len} chars")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    # --- Chart 1: Chunks Per Source ---
    file_counts = Counter(
        chunk.metadata.get("source_file", "Unknown file") for chunk in chunks
    )
    df_files = pd.DataFrame(
        list(file_counts.items()), columns=["Source File", "Chunk Count"]
    )

    with col_left:
        fig_bar = px.bar(
            df_files,
            x="Chunk Count",
            y="Source File",
            orientation="h",
            color_discrete_sequence=[THEME_COLORS["primary"]],
        )
        _apply_chart_layout(fig_bar, "Chunks Per Source Document")
        st.plotly_chart(fig_bar, width="stretch")

    # --- Chart 2: Chunk Length Distribution ---
    chunk_lengths = [len(chunk.page_content) for chunk in chunks]
    df_lengths = pd.DataFrame({"Chunk Length": chunk_lengths})

    with col_right:
        fig_hist = px.histogram(
            df_lengths,
            x="Chunk Length",
            nbins=15,
            color_discrete_sequence=[THEME_COLORS["secondary"]],
        )
        _apply_chart_layout(fig_hist, "Chunk Size Distribution (Chars)")
        st.plotly_chart(fig_hist, width="stretch")

    # --- Section: Query & Citation Analytics ---
    if conversation_turns and response_details:
        st.markdown("---")
        st.subheader("Query & Citation Analytics", anchor=False)

        turns_data = []
        for idx, (turn, details) in enumerate(
            zip(conversation_turns, response_details), start=1
        ):
            cited_count = len(details.get("sources", [])) if details else 0
            has_conflict = bool(details.get("conflict_summary")) if details else False
            turns_data.append(
                {
                    "Turn": f"Q{idx}",
                    "Question": turn.get("question", "")[:40] + "...",
                    "Cited Sources": cited_count,
                    "Has Disagreement": "Yes" if has_conflict else "No",
                }
            )

        df_turns = pd.DataFrame(turns_data)

        fig_turns = px.bar(
            df_turns,
            x="Turn",
            y="Cited Sources",
            color="Has Disagreement",
            color_discrete_map={"Yes": "#E1AE7D", "No": THEME_COLORS["primary"]},
            hover_data=["Question"],
        )
        _apply_chart_layout(fig_turns, "Citations Per Conversation Turn")
        st.plotly_chart(fig_turns, width="stretch")
