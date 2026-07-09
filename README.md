# MultiDocChat

MultiDocChat is a Streamlit app for chatting with multiple documents.

## Phase 1

This phase provides the initial ingestion pipeline:

- Streamlit entry point with an upload widget and chat input
- File dispatch for `.pdf`, `.docx`, `.txt`, and `.md`
- Chunking with LangChain's `RecursiveCharacterTextSplitter`
- Attribution metadata on every chunk, including `source_file`, `chunk_id`,
  and either `page_number` for PDFs or `section_heading` where headings can be
  detected
- Central embedding factory using local embeddings by default

## Phase 2

This phase adds local ChromaDB retrieval:

- Upload batches are embedded with
  `sentence-transformers/all-MiniLM-L6-v2` through `get_embedding_function("local")`
- Chroma persist directories include the provider name (`chroma_db_local`) so local
  384-dimensional MiniLM vectors stay physically separate from future OpenAI vectors
- The Streamlit app creates a fresh temporary local Chroma store whenever the upload
  batch changes
- `ingestion.vectorstore.get_retriever(k=4)` exposes the active retriever for later
  chains
- Chat input currently runs a manual similarity search and displays relevant chunks
  with source metadata

## Phase 3

This phase adds retrieval QA with source attribution:

- Chat questions retrieve the top matching Chroma chunks and pass them to a custom
  QA prompt labeled by `source_file` and page or section metadata
- The model is instructed to answer only from the provided excerpts and cite each
  factual statement inline as `[filename — section/page]`
- Inline citations are parsed back into structured source records
- The Streamlit response includes an expandable Sources panel with the exact file,
  page or section, chunk ID, and excerpt text used for the answer

Conflict detection and session memory will be expanded in later phases.

## Run

Use the existing `launchpad` Conda environment. It already contains the heavy ML
packages used by local embeddings (`torch`, `transformers`, and
`sentence-transformers`), so do not reinstall them for this project.

```bash
conda activate launchpad
cd "D:\Launchpad project\multidocchat"
pip install -r requirements.txt
```

Optional one-time version check:

```bash
pip show sentence-transformers transformers torch
```

Start the app:

```bash
streamlit run app.py
```

Inspect chunk attribution before vector storage:

```bash
python -m ingestion.loaders samples\example.pdf samples\notes.md samples\brief.docx
```

Embed files into local Chroma and run a manual similarity search:

```bash
python -m ingestion.vectorstore "What does the policy say about invoices?" samples\example.pdf
```
