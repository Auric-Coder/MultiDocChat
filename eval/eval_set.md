# MultiDocChat — Phase 7 Eval Set

**Fixed document set used for this eval (keep constant across runs):**
- `Hemanga Resume.pdf`
- `MultiDocChat_Implementation_Prompt.md`
- `samples/policy_2025.txt`
- `samples/policy_2026.txt`

**How to run:** upload all four files together in one session, ask each question exactly as written, and fill in the three result columns per question. Run this same file after Phase 9 (OpenAI swap) for direct before/after comparison.

**Scoring key:**
- **Correct?** — Y/N, does the answer match ground truth
- **Cited correctly?** — Y/N, does it cite the actual file/section that contains the answer (not a wrong file, not "no source")
- **Notes** — anything odd (slow, garbled, over-verbose, missed a source, etc.)

---

## Category A — Single-file factual (resume)

*Fill in the "Expected Answer" column yourself from the actual resume before running — you know it, I don't have the file content.*

| # | Question | Expected Answer | Correct? | Cited correctly? | Notes |
|---|---|---|---|---|---|
| 1 | What is Hemanga's CGPA? | _(fill in)_ | | | |
| 2 | What university does Hemanga attend and what is his specialization? | _(fill in)_ | | | |
| 3 | What ML projects are listed on the resume, and what were their reported metrics? | _(fill in)_ | | | |
| 4 | What competitive programming or competitive math achievements are listed? | _(fill in)_ | | | |

## Category B — Single-file factual (implementation doc)

*Ground truth confirmed against the doc content directly — these should be reliable checks.*

| # | Question | Expected Answer | Correct? | Cited correctly? | Notes |
|---|---|---|---|---|---|
| 5 | What embedding model is used through Phase 8? | `sentence-transformers/all-MiniLM-L6-v2` (local) | | | |
| 6 | What chunk size and overlap are used in the text splitter? | chunk_size=900, chunk_overlap=125 | | | |
| 7 | What happens in Phase 4? | Conflict detection — groups retrieved chunks by source file; if 2+ files contribute top chunks, runs a secondary LLM call to check agreement/disagreement and surfaces a warning callout if they conflict | | | |
| 8 | List all 9 phases in order with a one-line description of each. | Phase 0 (scaffolding) → 0b (env setup) → 1 (ingestion) → 2 (vector store) → 2b (per-source retrieval fix) → 3 (QA + citations) → 4 (conflict detection) → 5 (memory) → 6 (UI) → 7 (eval) → 8 (README/deploy) → 9 (OpenAI swap) | | | This is the enumeration-style query that tests your k-boosting fix — should now return all phases, not stop at Phase 4/9 |

## Category C — Conflict detection (policy files)

*Confirm the actual content of policy_2025.txt vs policy_2026.txt and fill in what specifically differs before running.*

| # | Question | Expected Answer | Correct? | Conflict flagged? | Notes |
|---|---|---|---|---|---|
| 9 | What does the policy say about [specific term/number that differs between the two files]? | _(fill in both versions' values)_ | | | Should trigger the "⚠️ Sources disagree" callout |
| 10 | Compare policy_2025 and policy_2026 — what changed? | _(fill in)_ | | | |
| 11 | What is [same question phrased differently] according to company policy? | _(fill in)_ | | | Tests whether conflict detection triggers regardless of question phrasing, not just exact wording |

## Category D — Cross-file (multiple sources in one answer)

| # | Question | Expected Answer | Correct? | Cited correctly? | Notes |
|---|---|---|---|---|---|
| 12 | Summarize all the documents I've uploaded. | Should list each unique file once (not once per chunk), with a real synthesis of content — tests the dedup fix | | | |
| 13 | What information do you have about "Hemanga" across all files? | Should pull from the resume specifically, not miss it even with the doc file also uploaded — tests the Phase 2b per-source retrieval fix | | | |

## Category E — Conversation memory (multi-turn)

| # | Turn | Question | Expected Behavior | Correct? | Cited correctly? | Notes |
|---|---|---|---|---|---|---|
| 14a | 1 | What is Hemanga's CGPA? | Direct factual answer, cited | | | |
| 14b | 2 | What projects has he built? | Direct factual answer, cited | | | |
| 14c | 3 | How does the first thing you mentioned compare to typical requirements? | Should resolve "the first thing" back to Turn 1 (CGPA), not confuse it with Turn 2's topic — tests question-condensing | | | |
| 15a | 1 | What phase covers vector store setup? | Should answer "Phase 2" | | | |
| 15b | 2 | What about the one after it? | Should resolve to "Phase 2b" or "Phase 3" depending on intended reading — note which one it picks and whether that's reasonable | | | |

---

## Summary scores (fill in after running)

- **Precision@k check:** for 3-4 questions above, manually inspect the retrieved chunks (not just the final answer) — do the top-k chunks actually contain the relevant info? Note the score as `X/Y questions had relevant chunks in top-k`.
- **Citation accuracy:** `X/15 questions cited the correct file/section`
- **Conflict-detection recall:** `X/3 conflict questions correctly triggered the warning`
- **Overall correct answers:** `X/15`

Copy these four numbers into your README for the Phase 7 deliverable, and re-run this exact file in Phase 9 against the OpenAI-backed stack for the before/after comparison.
