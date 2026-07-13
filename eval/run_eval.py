"""Run the fixed Phase 7 document-QA evaluation from the command line.

Usage (from the repository root)::

    python -m eval.run_eval

The script intentionally creates a new temporary Chroma collection every run,
so it never changes the Streamlit app's active collection or local persisted
data. It prints evidence for manual scoring; language-model answer quality and
citation support require human judgement.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chains.qa_chain import QAResult, answer_question
from eval.eval_questions import EVAL_CASES
from ingestion.loaders import SUPPORTED_EXTENSIONS, load_and_chunk
from ingestion.vectorstore import create_vector_store, make_temp_persist_directory
from memory.session import format_chat_history


def _sample_files(samples_dir: Path) -> list[Path]:
    files = sorted(
        path for path in samples_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if len(files) != 4:
        raise RuntimeError(
            f"Expected exactly four supported files in {samples_dir}, found {len(files)}: "
            + ", ".join(path.name for path in files)
        )
    return files


def _print_result(case: dict[str, Any], result: QAResult) -> None:
    print("=" * 88)
    print(f"[{case['id']}] {case['question']}")
    print(f"Expected: {case['expected']}")
    print(f"Answer:\n{result.answer}")
    print("Cited sources:")
    if result.sources:
        for source in result.sources:
            print(f"  - [{source.citation}] ({source.chunk_id})")
    else:
        print("  - No valid inline citations parsed.")
    print("Retrieved sources (manual precision@k evidence):")
    for source in result.retrieved_sources:
        preview = " ".join(source.excerpt.split())[:220]
        print(f"  - [{source.citation}] ({source.chunk_id}): {preview}")
    conflict = result.conflict
    print(
        "Conflict: "
        + ("YES" if conflict and conflict.has_conflict else "NO")
        + (" (checked)" if conflict and conflict.checked else " (not checked)")
    )
    if conflict and conflict.summary:
        print(f"Conflict summary: {conflict.summary}")


def run_eval(*, k: int = 8, samples_dir: Path | None = None) -> int:
    """Index samples and print every fixed evaluation case. Return failures."""

    samples_dir = samples_dir or PROJECT_ROOT / "samples"
    files = _sample_files(samples_dir)
    chunks = [chunk for file_path in files for chunk in load_and_chunk(str(file_path))]
    persist_directory = make_temp_persist_directory()
    failures = 0
    histories: dict[str, list[dict[str, str]]] = defaultdict(list)

    try:
        create_vector_store(chunks, persist_directory=persist_directory, collection_name="phase7_eval")
        print(f"Indexed {len(files)} files and {len(chunks)} chunks with all-MiniLM-L6-v2.")
        print("Score correctness, citation support, and retrieval relevance manually in eval_results.md.\n")

        for case in EVAL_CASES:
            conversation = case.get("conversation")
            history = format_chat_history(histories[conversation]) if conversation else ""
            try:
                result = answer_question(case["question"], k=k, chat_history=history)
            except Exception as exc:  # continue so a transient model failure is visible per case
                failures += 1
                print("=" * 88)
                print(f"[{case['id']}] {case['question']}")
                print(f"ERROR: {type(exc).__name__}: {exc}")
                continue

            _print_result(case, result)
            if conversation:
                histories[conversation].append(
                    {"question": case["question"], "answer": result.answer}
                )
    finally:
        shutil.rmtree(persist_directory, ignore_errors=True)

    print("=" * 88)
    print(f"Evaluation complete: {len(EVAL_CASES) - failures}/{len(EVAL_CASES)} cases completed.")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Phase 7 MultiDocChat evaluation.")
    parser.add_argument("--k", type=int, default=8, help="Retrieved chunks per source (default: 8).")
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be at least 1")
    return 1 if run_eval(k=args.k) else 0


if __name__ == "__main__":
    raise SystemExit(main())
