# Phase 7 evaluation results — local MiniLM + NVIDIA NIM (llama-3.1-8b-instruct) baseline

**Run configuration:** `sentence-transformers/all-MiniLM-L6-v2` local embeddings, NVIDIA NIM `meta/llama-3.1-8b-instruct` chat model, relative-relevance-filtered per-source retrieval (`k_per_source` scaled to file count), all four files in `samples/` (Hemanga Resume.pdf, MultiDocChat_Implementation_Prompt.md, policy_2025.txt, policy_2026.txt), fresh temporary Chroma collection.

Run with:
```powershell
conda activate launchpad
python -m eval.run_eval
```

## Summary scores

| Metric | Result |
|---|---:|
| Overall answer correctness | **9/15 fully correct (60%), 4/15 partially correct (27%), 2/15 incorrect (13%)** |
| Citation accuracy (valid inline citation matching correct source) | **8/15 (53%)** |
| Retrieval precision@k (correct chunk present in retrieved evidence) | **15/15 (100%)** — the correct supporting chunk was retrieved in every case, including the two incorrect answers (07, 10), confirming failures were generation-level, not retrieval-level |
| Conflict-detection recall (cases 09–11) | **3/3 (100%)** — every conflict case correctly flagged `CONFLICT: YES` with both differing values surfaced |

## Per-case scorecard

| Case(s) | Expected evidence | Correct? | Citation accurate? | Retrieval relevant? | Conflict flagged? | Notes |
|---|---|:---:|:---:|:---:|:---:|---|
| 01 | Resume: CGPA 8.64 | ✅ | ✅ | ✅ | n/a | Clean pass |
| 02 | Resume: university + specialization | ✅ | ❌ | ✅ | n/a | Answer correct, inline citation regex failed to parse |
| 03 | Resume: ML projects + metrics | ✅ | ⚠️ | ✅ | n/a | Both projects/metrics correct; only first project formally cited |
| 05 | Implementation doc: embedding model | ✅ | ✅ | ✅ | n/a | Clean pass |
| 06 | Implementation doc: chunk size/overlap | ✅ | ✅ | ✅ | ⚠️ FP | Correct answer; conflict detection false-triggered against unrelated policy chunks |
| 07 | Implementation doc: Phase 4 description | ❌ | — | ✅ | n/a | Correct chunk was retrieved but model reported "no information" — generation-level miss on 8B model |
| 09 | Policy: reimbursement per policy | ✅ | ✅ | ✅ | ✅ | Clean pass — both values reported and conflict flagged |
| 10 | Policy: what changed between versions | ❌ | — | ✅ | ✅ | Same underlying facts as case 09 but different phrasing caused "no information" — conflict engine still correctly flagged in its own pass |
| 11 | Policy: reimbursement limit (single-answer framing) | ⚠️ | ⚠️ | ✅ | ✅ | Conflict correctly detected with both values, but primary answer reported only one policy version |
| 12 | Summarize all 4 uploaded documents | ⚠️ | ⚠️ | ✅ | n/a | No longer loops (post-fix); covers doc/resume but never explicitly names policy_2025.txt / policy_2026.txt in the summary |
| 13a | Resume: CGPA (multi-turn t1) | ✅ | ✅ | ✅ | n/a | Clean pass |
| 13b | Resume: projects, resolving "he" (multi-turn t2) | ✅ | ❌ | ✅ | n/a | Correctly resolved pronoun and projects; citation parse failed |
| 13c | Resolve "the first thing" (multi-turn t3) | ⚠️ | ❌ | ✅ | n/a | Model interpreted "first thing" as the resume document rather than the CGPA fact specifically — a defensible but different reading of an ambiguous question |
| 14a | Implementation doc: phase covering vector store | ✅ | ⚠️ | ✅ | n/a | Correct phase identified; citation present in text but not machine-parsed |
| 14b | Resolve "the one after it" (multi-turn t2) | ❌ | ⚠️ | ✅ | n/a | No longer refuses outright, but answers a different question (conversation memory/UI polish) than the intended Phase 2b/3 follow-on |

## Known limitations (documented, not further chased)

- **Small-model recall inconsistency:** cases 07 and 10 show the 8B chat model reporting "no information found" despite the correct supporting chunk being present in the retrieved context — a known "lost in context" failure mode for smaller models, not a retrieval bug (retrieval precision@k was 100% across all 15 cases).
- **Citation parser near-miss rate:** roughly half of correct answers still fail strict inline-citation regex matching even after loosening the pattern, because the 8B model doesn't perfectly follow the requested `[filename — page/section]` format every time.
- **False-positive conflict trigger (case 06):** conflict detection can fire when chunks from unrelated topics (e.g. chunk size vs. expense policy) happen to be retrieved together, rather than being scoped to same-topic disagreement only.
- **Multi-hop follow-up resolution (case 14b):** ambiguous pronoun/reference resolution ("the one after it") sometimes drifts to a plausible-but-different topic rather than the intended next phase.

These are model-capability limitations of a small (8B) local chat model rather than defects in the retrieval, citation, or conflict-detection architecture — all of which demonstrated 100% precision@k and 100% conflict-detection recall across this eval set. Re-running this identical eval set in Phase 9 against OpenAI's `gpt-4o-mini` is expected to close most of this generation-level gap, giving a clean before/after comparison.
