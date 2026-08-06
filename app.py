"""Streamlit entry point for MultiDocChat."""

from collections import Counter
from html import escape

import streamlit as st

from analytics.dashboard import render_analytics_dashboard
from analytics.explorer import render_document_explorer
from chains.comparison import compare_documents
from chains.qa_chain import DEFAULT_CHAT_PROVIDER, answer_question
from eval.metrics import run_rag_evaluation
from export.report import export_chat_report_html, export_chat_report_markdown
from ingestion.loaders import CHUNK_SIZE, process_uploaded_files
from ingestion.web_scraper import scrape_urls
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


def _inject_styles():
    """Apply the app's editorial visual language to Streamlit primitives."""

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Manrope:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
        .stApp { background: #141713; color: #f1ede2; }
        .block-container { max-width: 860px; padding-top: 3.25rem; padding-bottom: 2.75rem; }
        h1, h2, h3 { font-family: 'Fraunces', serif !important; color: #f3eee2; letter-spacing: -0.025em; }
        h1 { font-size: clamp(2.4rem, 6vw, 3.7rem) !important; margin-bottom: 0.2rem !important; }
        [data-testid="stCaptionContainer"] { color: #b8b5a8; }
        [data-testid="stSidebar"] { background: #191c18; border-right: 1px solid #363a31; }
        [data-testid="stSidebar"] h2 { font-size: 1.35rem; }
        [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] { border: 1px solid #363a31; border-radius: 10px; background: #1d211c; box-shadow: none; }
        [data-testid="stFileUploader"] { padding: .2rem 0 .15rem; }
        [data-testid="stFileUploaderDropzone"] { border: 1px dashed #a97752; border-radius: 8px; background: #181b17; }
        [data-testid="stChatMessage"] { padding: .9rem 0; gap: .65rem; }
        [data-testid="stChatInput"] { border-radius: 9px; border-color: #4d5046; background: #1d211c; }
        .stButton > button { border-radius: 7px; border-color: #a97752; color: #e6c09c; background: transparent; font-weight: 600; }
        .stButton > button:hover { border-color: #d6a475; background: #33261e; color: #fff3e6; }
        .source-heading { margin: 1.5rem 0 .6rem; font-size: .7rem; font-weight: 700; letter-spacing: .13em; color: #d3a173; }
        .source-note { margin: 0 0 .6rem; color: #b8b5a8; font-size: .84rem; }
        .source-card { position: relative; overflow: hidden; margin: .55rem 0; padding: .9rem 1rem 1rem 1.15rem; border: 1px solid #3d4038; border-radius: 8px; background: #1d211c; }
        .source-card::before { content: ''; position: absolute; inset: 0 auto 0 0; width: 4px; background: #c99162; }
        .source-meta { margin-bottom: .35rem; font-family: 'DM Mono', monospace; font-size: .68rem; letter-spacing: .06em; color: #d3a173; text-transform: uppercase; }
        .source-citation { color: #f3eee2; font-weight: 700; font-size: .93rem; }
        .source-excerpt { margin-top: .35rem; color: #cbc8bc; font-size: .9rem; line-height: 1.58; }
        .conflict-callout { margin: 1.15rem 0 .8rem; padding: .9rem 1rem .95rem; border-left: 4px solid #c99162; border-radius: 0 8px 8px 0; background: #31251d; color: #f2dac0; }
        .conflict-label { display: block; margin-bottom: .25rem; font-family: 'DM Mono', monospace; font-size: .7rem; font-weight: 500; letter-spacing: .11em; text-transform: uppercase; color: #e1ae7d; }
        .conflict-copy { font-size: .91rem; line-height: 1.55; }
        [data-testid="stAlert"] { border-radius: 8px; border: 1px solid #45483f; background: #242822; color: #e5e1d6; }
        </style>
        """,
        unsafe_allow_html=True,
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
    if not details:
        return

    title = f"📚 Cited Sources ({len(details)})" if cited else f"🔍 Retrieved Context ({len(details)})"
    with st.expander(title, expanded=False):
        if cited:
            st.markdown('<p class="source-note">Excerpts cited in the answer.</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="source-note">Retrieved context; no inline citations were parsed.</p>', unsafe_allow_html=True)

        for index, source in enumerate(details, start=1):
            citation = escape(str(source["citation"]))
            chunk_id = escape(str(source["chunk_id"]))
            excerpt = escape(str(source["excerpt"])[:900]).replace("\n", "<br>")
            st.markdown(
                f'''<article class="source-card">
                    <div class="source-meta">Source {index} · Chunk {chunk_id}</div>
                    <div class="source-citation">{citation}</div>
                    <div class="source-excerpt">{excerpt}</div>
                </article>''',
                unsafe_allow_html=True,
            )


def _render_assistant_response(answer, details=None):
    if details and details.get("confidence_label"):
        score = details.get("confidence_score", 0.0)
        label = details.get("confidence_label", "Medium")
        color = "#85E89D" if label == "High" else ("#FFE58F" if label == "Medium" else "#FF7875")
        st.markdown(
            f'<span style="display:inline-block; font-size:0.78rem; font-family: \'DM Mono\', monospace; padding:3px 9px; border-radius:4px; background:{color}1A; color:{color}; border:1px solid {color}44; margin-bottom:10px;">'
            f'🎯 Confidence: {label} ({int(score * 100)}%)</span>',
            unsafe_allow_html=True,
        )
    st.markdown(answer)
    if not details:
        return

    if details.get("conflict_summary"):
        summary = escape(str(details["conflict_summary"])).replace("\n", "<br>")
        st.markdown(
            f'''<aside class="conflict-callout">
                <span class="conflict-label">Source difference</span>
                <div class="conflict-copy">{summary}</div>
            </aside>''',
            unsafe_allow_html=True,
        )
    _render_sources(details["sources"], cited=details["has_citations"])


st.set_page_config(
    page_title="MultiDocChat",
    page_icon=":material/forum:",
    layout="centered",
)

_inject_styles()

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
    search_strategy = st.segmented_control(
        "Search strategy",
        ["Semantic", "Keyword", "Hybrid"],
        default="Hybrid",
        help=(
            "**Semantic** — vector similarity only.  "
            "**Keyword** — BM25 term matching only.  "
            "**Hybrid** — fuses both signals (recommended)."
        ),
    )
    st.caption("Model, embeddings, and chunking stay fixed for this phase; k and strategy are live.")

    with st.expander("📊 System Status & Health", expanded=False):
        import os
        has_nvidia_key = bool(os.getenv("NVIDIA_API_KEY"))
        vs_ready = _vector_store_is_active()
        doc_count = len(st.session_state.get("indexed_files", []))
        chunk_count = st.session_state.get("chunk_count", 0)

        st.markdown(f"**NVIDIA API Key:** {'🟢 Active' if has_nvidia_key else '🔴 Not Set'}")
        st.markdown(f"**Vector Store:** {'🟢 Ready' if vs_ready else '⚪ Idle'}")
        st.markdown(f"**Documents:** {doc_count} files ({chunk_count} chunks)")
        st.markdown(f"**Strategy:** {search_strategy or 'Hybrid'}")

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
    tab_files_input, tab_urls_input = st.tabs(["📁 Local Files", "🌐 Web URLs"])
    with tab_files_input:
        uploaded_files = st.file_uploader(
            "Upload documents to index",
            accept_multiple_files=True,
            type=SUPPORTED_FILE_TYPES,
            key=f"uploaded_files_{st.session_state.upload_widget_version}",
            help="Upload PDF, DOCX, TXT, or Markdown files.",
        )
    with tab_urls_input:
        url_text = st.text_area(
            "Enter web page URLs (one per line):",
            placeholder="https://example.com/article1\nhttps://example.com/article2",
            height=100,
            key=f"url_text_{st.session_state.upload_widget_version}",
        )
        fetch_urls = st.button("🚀 Fetch & Index Web Pages")

if fetch_urls and url_text.strip():
    urls = [u.strip() for u in url_text.splitlines() if u.strip()]
    if urls:
        try:
            with st.spinner("Fetching and scraping web pages..."):
                url_chunks = scrape_urls(urls)
                if url_chunks:
                    existing_chunks = st.session_state.get("chunks", [])
                    combined_chunks = list(existing_chunks) + list(url_chunks)
                    persist_directory = st.session_state.get("persist_directory") or make_temp_persist_directory()
                    create_vector_store(combined_chunks, persist_directory=persist_directory)

                    new_sources = list(set(chunk.metadata.get("source_file", "URL") for chunk in url_chunks))
                    existing_files = st.session_state.get("indexed_files", [])

                    st.session_state.chunks = combined_chunks
                    st.session_state.chunk_count = len(combined_chunks)
                    st.session_state.persist_directory = persist_directory
                    st.session_state.indexed_files = list(set(existing_files + new_sources))
                    st.success(f"Indexed {len(url_chunks)} chunks from {len(urls)} web pages!")
                else:
                    st.warning("No readable text content extracted from the provided URLs.")
        except Exception as exc:
            st.error(f"Could not scrape URLs: {exc}")

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

tab_chat, tab_analytics, tab_explorer, tab_compare, tab_eval, tab_export = st.tabs(
    ["💬 Chat", "📊 Analytics", "🔍 Explorer", "⚔️ Comparison", "📈 RAG Eval", "📥 Export"]
)

with tab_chat:
    quick_prompt = None
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
                st.markdown(f"**{filename}** · {file_counts[filename]} chunks")

        # --- Quick Query Buttons (B3) ---
        col_q1, col_q2, col_q3, col_q4 = st.columns(4)
        with col_q1:
            if st.button("📋 Summarize All", use_container_width=True):
                quick_prompt = "Summarize all the uploaded documents concisely."
        with col_q2:
            if st.button("🔑 Key Points", use_container_width=True):
                quick_prompt = "What are the main key points across all documents?"
        with col_q3:
            if st.button("📈 Main Topics", use_container_width=True):
                quick_prompt = "List the primary topics covered in these documents."
        with col_q4:
            if st.button("💡 Key Insights", use_container_width=True):
                quick_prompt = "What are the most important insights or takeaways?"

    # --- Conversation History ---
    for index, turn in enumerate(get_conversation_turns(st.session_state)):
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            details = st.session_state.response_details[index] if index < len(st.session_state.response_details) else None
            _render_assistant_response(turn["answer"], details)

    if not indexed_files:
        st.info("Upload one or more documents to begin a cited conversation.", icon=":material/upload_file:")

    # --- Chat Input Anchored at Bottom ---
    prompt = st.chat_input(
        "Ask a question about the indexed documents",
        disabled="persist_directory" not in st.session_state,
        submit_mode="disable",
    )

    user_query = prompt or quick_prompt

    if user_query:
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching your documents…"):
                    result = answer_question(
                        user_query,
                        k=retrieval_k,
                        chat_history=format_chat_history(
                            get_conversation_turns(st.session_state)
                        ),
                        strategy=(search_strategy or "Hybrid").lower(),
                    )
            except Exception as exc:
                st.error(f"Could not generate an answer: {exc}")
            else:
                sources_to_show = result.sources or result.retrieved_sources
                details = {
                    "sources": _source_details(sources_to_show),
                    "has_citations": bool(result.sources),
                    "confidence_score": getattr(result, "confidence_score", 0.8),
                    "confidence_label": getattr(result, "confidence_label", "High"),
                    "conflict_summary": (
                        result.conflict.summary
                        or "\n".join(result.conflict.source_positions())
                        if result.conflict and result.conflict.has_conflict
                        else ""
                    ),
                }
                add_conversation_turn(st.session_state, user_query, result.answer)
                retained_turn_count = len(get_conversation_turns(st.session_state))
                st.session_state.response_details = (
                    st.session_state.response_details + [details]
                )[-retained_turn_count:]
                st.rerun()

with tab_analytics:
    render_analytics_dashboard(
        st.session_state.get("chunks", []),
        st.session_state.get("indexed_files", []),
        st.session_state.get("response_details", []),
        get_conversation_turns(st.session_state),
    )

with tab_explorer:
    render_document_explorer(st.session_state.get("chunks", []))

with tab_compare:
    st.subheader("Document Comparison & Diff Analysis", anchor=False)
    st.caption("Compare two documents or text versions line-by-line with AI executive insights.")

    col_doc1, col_doc2 = st.columns(2)
    with col_doc1:
        doc1_name = st.text_input("Document 1 Label", value="Version 2025")
        doc1_text = st.text_area("Document 1 Content", height=200, placeholder="Paste text for Document 1...")
    with col_doc2:
        doc2_name = st.text_input("Document 2 Label", value="Version 2026")
        doc2_text = st.text_area("Document 2 Content", height=200, placeholder="Paste text for Document 2...")

    if st.button("⚔️ Run Diff & Comparison Analysis", type="primary") and doc1_text.strip() and doc2_text.strip():
        with st.spinner("Analyzing differences between documents..."):
            try:
                cmp_res = compare_documents(doc1_text, doc2_text, doc1_name, doc2_name)
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Additions (+)", f"+{cmp_res.additions_count}", delta_color="normal")
                with c2:
                    st.metric("Deletions (-)", f"-{cmp_res.deletions_count}", delta_color="inverse")

                st.subheader("Executive AI Summary")
                st.markdown(cmp_res.executive_summary)

                with st.expander("🔍 View Raw Unified Line Diff"):
                    diff_text = "\n".join(cmp_res.diff_lines[:100])
                    st.code(diff_text, language="diff")
            except Exception as exc:
                st.error(f"Could not complete comparison: {exc}")

with tab_eval:
    st.subheader("RAGAS Quantitative Metric Scorecard", anchor=False)
    st.caption("Automated evaluation of Faithfulness, Relevancy, Precision, and Recall for current session.")

    turns = get_conversation_turns(st.session_state)
    details_list = st.session_state.get("response_details", [])

    if not turns:
        st.info("Ask questions in the Chat tab to view live RAG evaluation metrics.")
    else:
        eval_records = []
        for idx, (turn, details) in enumerate(zip(turns, details_list), start=1):
            q = turn.get("question", "")
            a = turn.get("answer", "")
            retrieved = details.get("sources", []) if details else []

            metrics = run_rag_evaluation(q, a, retrieved)
            eval_records.append({
                "Turn": f"Q{idx}",
                "Faithfulness": f"{int(metrics['faithfulness']*100)}%",
                "Relevancy": f"{int(metrics['relevancy']*100)}%",
                "Precision": f"{int(metrics['precision']*100)}%",
                "RAG Score": metrics["rag_score"],
            })

        import pandas as pd
        df_eval = pd.DataFrame(eval_records)
        st.dataframe(df_eval, use_container_width=True)

with tab_export:
    st.subheader("Export Session Report", anchor=False)
    st.caption("Download full transcript, citations, confidence scores, and conflict callouts.")

    turns = get_conversation_turns(st.session_state)
    details_list = st.session_state.get("response_details", [])
    indexed = st.session_state.get("indexed_files", [])

    if not turns:
        st.info("No conversation turns to export yet.")
    else:
        md_report = export_chat_report_markdown(turns, details_list, indexed)
        html_report = export_chat_report_html(turns, details_list, indexed)

        col_ex1, col_ex2 = st.columns(2)
        with col_ex1:
            st.download_button(
                "📥 Download Markdown Report (.md)",
                data=md_report,
                file_name="multidocchat_report.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with col_ex2:
            st.download_button(
                "📥 Download Printable HTML Report (.html)",
                data=html_report,
                file_name="multidocchat_report.html",
                mime="text/html",
                use_container_width=True,
            )

        with st.expander("📄 Report Preview"):
            st.markdown(md_report)


