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

Vector storage, retrieval, chains, conflict detection, and session memory will be
expanded in later phases.

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
