# MultiDocChat: Cross-File Assistant — Implementation Prompt

**Stack:** Python, LangChain, ChromaDB, Streamlit
**Core goals:**
1. Answer questions that draw from multiple files with clear per-answer source attribution (file + section).
2. Keep answers grounded when files contain conflicting information — surface the conflict instead of picking one silently.
3. Maintain conversation memory so follow-up questions stay in context.

Use this document as a build spec. Each phase is self-contained — feed one phase at a time to an AI coding assistant (or work through it yourself), verify it works, then move to the next.

---

## Phase 0 — Project Scaffolding

**Goal:** Working skeleton, no logic yet.

- Create project structure:
  ```
  multidocchat/
  ├── app.py                 # Streamlit entry point
  ├── ingestion/
  │   ├── loaders.py         # file loading + chunking
  │   └── vectorstore.py     # ChromaDB setup
  ├── chains/
  │   ├── qa_chain.py        # retrieval + attribution
  │   └── conflict.py        # conflict detection logic
  ├── memory/
  │   └── session.py         # conversation memory management
  ├── requirements.txt
  ├── .env.example
  └── README.md
  ```
- `requirements.txt`: `langchain`, `langchain-community`, `langchain-chroma`, `chromadb`, `streamlit`, `sentence-transformers`, `langchain-openai` (installed now but unused until Phase 9), `pypdf`, `python-docx`, `tiktoken`, `python-dotenv`.
- Confirm Streamlit runs with a placeholder "Upload files" widget and a chat input box, no backend wired yet.
- **Embedding strategy:** build the entire project on local embeddings (`sentence-transformers/all-MiniLM-L6-v2`) first — free, offline, no rate limits while you're debugging chunking/retrieval logic. OpenAI embeddings get swapped in only once everything else works (see Phase 9). To make that swap painless later, put the embedding choice behind a single factory function from day one:
  ```python
  # ingestion/vectorstore.py
  def get_embedding_function(provider="local"):
      if provider == "local":
          from langchain_community.embeddings import HuggingFaceEmbeddings
          return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
      elif provider == "openai":
          from langchain_openai import OpenAIEmbeddings
          return OpenAIEmbeddings(model="text-embedding-3-small")
  ```
  Every other module should call `get_embedding_function()` rather than instantiating an embedding model directly — this is the only place the provider flag should live.

**Deliverable:** App launches locally, UI shell visible, no errors.

---

### Phase 0b — Environment Setup (Conda, reusing `launchpad`)

**Goal:** Get a working environment without re-downloading torch/sentence-transformers.

If you already have a `launchpad` conda environment (from a separate course, with `torch`, `sentence-transformers`, `transformers`, `openai`, etc. already installed) — **reuse it**. Don't create a second environment just for MultiDocChat; the two heaviest installs (`torch`, `sentence-transformers`) are already there.

```bash
conda activate launchpad

# add only what's missing for MultiDocChat:
pip install langchain langchain-community langchain-chroma chromadb streamlit pypdf python-docx langchain-openai
```

Notes:
- Skip `torch`/`sentence-transformers`/`transformers` in this install — they're already present from your faculty's setup.
- No Jupyter kernel registration needed. MultiDocChat runs via `streamlit run app.py` in the terminal, not notebooks, so the `ipykernel install --name launchpad` step from your faculty's instructions is irrelevant here.
- No Hugging Face login required — `all-MiniLM-L6-v2` is a public model, downloads anonymously on first use, then caches locally.
- Version check worth doing once: run `pip show sentence-transformers transformers torch` and confirm nothing conflicts when `langchain-community`'s `HuggingFaceEmbeddings` tries to import them. If `pip install` reports a dependency conflict, note the exact error before force-resolving it — don't blindly upgrade/downgrade `torch`, since that's shared with your other course's notebooks.
- Every time you work on MultiDocChat: `conda activate launchpad` first, then `cd` into the project and `streamlit run app.py`.

**Deliverable:** `conda activate launchpad && pip list` shows both the original launchpad packages and the newly added langchain/chromadb/streamlit packages with no install errors.

---

## Phase 1 — Document Ingestion Pipeline

**Goal:** Load heterogeneous files, chunk them, and tag every chunk with rich metadata.

- Support at minimum: `.pdf`, `.docx`, `.txt`, `.md`. Use LangChain's `PyPDFLoader`, `Docx2txtLoader`, `TextLoader` accordingly — dispatch by file extension.
- Chunking: use `RecursiveCharacterTextSplitter` with `chunk_size=800–1000`, `chunk_overlap=100–150`. Tune based on document density.
- **Critical for attribution:** every chunk's metadata must carry:
  - `source_file` (original filename)
  - `page_number` (for PDFs) or `section_heading` (for docx/md, extracted via heading detection)
  - `chunk_id`
- Write a `load_and_chunk(file_path) -> List[Document]` function per file type, unified under a single `process_uploaded_files(files: List[UploadedFile]) -> List[Document]`.

**Deliverable:** Given 2–3 sample files, print out chunks with metadata to confirm attribution data is correctly attached before it ever reaches the vector store.

---

## Phase 2 — Vector Store (ChromaDB)

**Goal:** Persistent, per-session or per-collection vector store.

- Initialize a ChromaDB collection (`langchain_chroma.Chroma`) using `get_embedding_function("local")` from Phase 0. Stay on `"local"` for every phase through Phase 8 — don't touch OpenAI until Phase 9.
- Decide session scoping: for a portfolio demo, simplest is a fresh in-memory/temp-dir Chroma collection per Streamlit session (cleared on new upload batch). For persistence across sessions, use a `persist_directory` keyed by a session/user ID.
- Name the persist directory with the provider baked in, e.g. `chroma_db_local/` — this avoids ever accidentally querying a MiniLM-embedded collection with OpenAI embeddings later. `all-MiniLM-L6-v2` produces 384-dim vectors vs. 1536-dim for `text-embedding-3-small`; the two are not interchangeable in the same collection, so keep them physically separate from the start.
- Add chunks from Phase 1 via `Chroma.from_documents(docs, embedding=embedding_fn, persist_directory=...)`.
- Expose a `get_retriever(k=4)` helper — this is what downstream chains will call.

**Deliverable:** Upload files → chunks embedded and stored → a manual similarity search query returns relevant chunks with correct metadata attached.

---

## Phase 3 — Retrieval QA with Source Attribution

**Goal:** Answers that explicitly cite which file/section they drew from.

- Use `RetrievalQAWithSourcesChain` as a starting point, or build a custom chain if you want more control over attribution formatting (recommended for a portfolio project — shows deeper understanding than using the chain off-the-shelf).
- Custom approach: retrieve top-k chunks → construct a prompt that includes each chunk labeled with its `source_file`/`section` → instruct the LLM (via system prompt) to cite `[filename, section/page]` inline after every claim it makes.
- Prompt template should explicitly say: *"Only answer using the provided excerpts. For every factual statement, cite the source in the format [filename — section/page]. If the excerpts don't contain the answer, say so."*
- Parse the LLM output to extract citations for a structured "Sources" panel in the UI (regex or ask the model to also return a JSON sources list alongside the prose answer).

**Deliverable:** A query returns an answer with inline citations, and a separate expandable "Sources" section listing exact file + page/section for each excerpt used.

---

## Phase 4 — Conflict Detection

**Goal:** When two files disagree, say so instead of silently picking one.

- After retrieving top-k chunks across files, group them by `source_file`.
- If the top relevant chunks come from 2+ distinct files, run a lightweight secondary LLM call: *"Here are excerpts from different sources on the same question. Do they agree or conflict? If they conflict, summarize each position and which source it came from."*
- Surface this as a distinct UI element (e.g., a warning callout: "⚠️ Sources disagree") rather than burying it in the main answer.
- Keep this cheap: only trigger the conflict check when retrieved chunks span multiple files AND a simple heuristic (e.g., embedding distance between chunks, or asking the LLM directly) suggests disagreement — don't run it on every query if cost/latency matters for the demo.

**Deliverable:** A test case with two intentionally contradictory sample files (e.g., two versions of a policy doc with different numbers) correctly triggers the conflict callout with both positions shown.

---

## Phase 5 — Conversation Memory

**Goal:** Follow-ups like "what about the second one?" resolve correctly.

- Use `ConversationalRetrievalChain` or manually maintain a `ConversationBufferMemory` / `ConversationBufferWindowMemory` (windowed is safer for cost — cap at last 5–8 turns).
- Store memory in `st.session_state` so it persists across Streamlit reruns within a session.
- Make sure the question-condensing step (rephrasing follow-up questions into standalone queries using chat history) happens before retrieval, not after — this is what makes pronouns/follow-ups resolve to the right documents.

**Deliverable:** A 3-turn conversation where turn 3 references turn 1 ambiguously still retrieves correctly and cites sources correctly.

---

## Phase 6 — Streamlit UI Polish

**Goal:** Demo-ready interface.

- File upload widget (multi-file), with a visible list of currently indexed files and a "clear/reset" button.
- Chat interface using `st.chat_message` / `st.chat_input`.
- Each assistant response should render: (1) the answer text with inline citation markers, (2) an expandable "Sources" section with file/page snippets, (3) a conflict warning banner when applicable.
- Sidebar: model/embedding choice toggle, chunk size, k (number of retrieved chunks) — useful for a demo to show you understand the tuning knobs.

**Deliverable:** End-to-end demo: upload 3 files, ask a cross-file question, get attributed answer, ask a follow-up, get a correctly contextualized answer.

---

## Phase 7 — Testing & Evaluation

**Goal:** Have concrete numbers for your resume bullet.

- Build a small eval set: 10–15 Q&A pairs against your test documents, including at least 2–3 designed to trigger conflicts and 2–3 multi-hop follow-ups.
- Metrics worth capturing: retrieval precision@k (manually labeled relevant chunks), citation accuracy (does the cited source actually support the claim?), and conflict-detection recall (did it catch the conflicts you planted?).
- This gives you a defensible number like "achieved 90% citation accuracy across a 15-query eval set" instead of a vague claim.
- Run this eval set now on `all-MiniLM-L6-v2` and save the scores — you'll re-run the identical set in Phase 9 on OpenAI embeddings for a direct before/after comparison.

**Deliverable:** A short `eval_results.md` with your test set and scores — useful both for validation and for resume/portfolio writeups later.

---

## Phase 8 — README & Deployment

**Goal:** Portfolio-ready packaging.

- README: problem statement, architecture diagram (text or image), stack, setup instructions, screenshot/GIF of the demo, eval results summary.
- Deploy on Streamlit Community Cloud (free tier) for a live demo link — this matters more for your resume than the GitHub repo alone.
- Add a `.env.example` and make sure no API keys are committed.

**Deliverable:** Public GitHub repo + live demo link, ready to drop into your resume/portfolio.

---

---

## Phase 9 — Swap in OpenAI Embeddings (after everything above works)

**Goal:** Upgrade retrieval quality once the architecture is proven, without touching app logic.

- Get an OpenAI API key, add it to `.env` as `OPENAI_API_KEY`, and confirm `.env` is in `.gitignore`.
- Change exactly one thing: call `get_embedding_function("openai")` instead of `"local"` wherever the vector store is built. Point `persist_directory` at a new path, e.g. `chroma_db_openai/` — do not reuse the MiniLM collection (see the dimension mismatch note in Phase 2).
- Re-ingest all documents into the new collection (this is a full re-embed, not an in-place upgrade).
- Re-run the exact eval set from Phase 7 against the OpenAI-backed collection. Compare precision@k and citation accuracy against your MiniLM baseline.
- Add both numbers to your README: e.g. *"Switched to OpenAI text-embedding-3-small; citation accuracy improved from X% to Y% on the same 15-query eval set."* This before/after comparison is a stronger portfolio signal than either number alone — it shows you understand the tradeoff, not just that you called an API.
- Add a toggle in the Streamlit sidebar (or an env var) so you can flip between local and OpenAI embeddings for demo purposes — useful if you want to show both live without maintaining two separate deployments.

**Deliverable:** Two eval score sets (local vs. OpenAI) side by side, a working OpenAI-backed deployment, and a README section documenting the comparison.

---

### Notes for use with an AI coding assistant
Paste one phase at a time as a prompt, e.g.:
> "Implement Phase 1 of this spec: [paste Phase 1 section]. Here's my current project structure: [paste tree/files]."

This keeps context tight and each phase independently testable — better results than asking for the whole app in one shot.
