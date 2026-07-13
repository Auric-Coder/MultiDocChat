"""Fixed Phase 7 evaluation cases.

Keep this file unchanged when rerunning the evaluation with Phase 9 embeddings.
``conversation`` groups consecutive turns that share chat history; ``None``
starts a new conversation.
"""

from __future__ import annotations


EVAL_CASES = [
    {
        "id": "01",
        "question": "What is Hemanga's CGPA?",
        "expected": "The CGPA shown in Hemanga Resume.pdf.",
        "relevant_sources": ["Hemanga Resume.pdf"],
        "conversation": None,
    },
    {
        "id": "02",
        "question": "What university does Hemanga attend and what is his specialization?",
        "expected": "The university and specialization shown in Hemanga Resume.pdf.",
        "relevant_sources": ["Hemanga Resume.pdf"],
        "conversation": None,
    },
    {
        "id": "03",
        "question": "What ML projects are listed on the resume, and what were their reported metrics?",
        "expected": "The listed ML projects and metrics from Hemanga Resume.pdf.",
        "relevant_sources": ["Hemanga Resume.pdf"],
        "conversation": None,
    },
    {
        "id": "05",
        "question": "What embedding model is used through Phase 8?",
        "expected": "sentence-transformers/all-MiniLM-L6-v2, using local embeddings.",
        "relevant_sources": ["MultiDocChat_Implementation_Prompt.md"],
        "conversation": None,
    },
    {
        "id": "06",
        "question": "What chunk size and overlap are used in the text splitter?",
        "expected": "chunk_size=900 and chunk_overlap=125.",
        "relevant_sources": ["MultiDocChat_Implementation_Prompt.md"],
        "conversation": None,
    },
    {
        "id": "07",
        "question": "What happens in Phase 4?",
        "expected": "Conflict detection compares relevant multi-source excerpts and surfaces a warning for disagreements.",
        "relevant_sources": ["MultiDocChat_Implementation_Prompt.md"],
        "conversation": None,
    },
    {
        "id": "09",
        "question": "How much home internet reimbursement does each policy allow per month?",
        "expected": "2025 allows $100/month and 2026 allows $150/month; the disagreement is flagged.",
        "relevant_sources": ["policy_2025.txt", "policy_2026.txt"],
        "expect_conflict": True,
        "conversation": None,
    },
    {
        "id": "10",
        "question": "Compare policy_2025 and policy_2026: what changed?",
        "expected": "The monthly home-internet reimbursement increased from $100 to $150; the disagreement is flagged.",
        "relevant_sources": ["policy_2025.txt", "policy_2026.txt"],
        "expect_conflict": True,
        "conversation": None,
    },
    {
        "id": "11",
        "question": "According to company policy, what is the home internet reimbursement limit?",
        "expected": "Both policy versions' limits are reported and the disagreement is flagged.",
        "relevant_sources": ["policy_2025.txt", "policy_2026.txt"],
        "expect_conflict": True,
        "conversation": None,
    },
    {
        "id": "12",
        "question": "Summarize all the documents I've uploaded.",
        "expected": "A synthesis that identifies each of the four source files once.",
        "relevant_sources": [
            "Hemanga Resume.pdf",
            "MultiDocChat_Implementation_Prompt.md",
            "policy_2025.txt",
            "policy_2026.txt",
        ],
        "conversation": None,
    },
    {
        "id": "13a",
        "question": "What is Hemanga's CGPA?",
        "expected": "The CGPA shown in Hemanga Resume.pdf.",
        "relevant_sources": ["Hemanga Resume.pdf"],
        "conversation": "resume_follow_up",
    },
    {
        "id": "13b",
        "question": "What projects has he built?",
        "expected": "Projects from Hemanga Resume.pdf, using the prior turn to resolve 'he'.",
        "relevant_sources": ["Hemanga Resume.pdf"],
        "conversation": "resume_follow_up",
    },
    {
        "id": "13c",
        "question": "How does the first thing you mentioned compare to typical requirements?",
        "expected": "The question is resolved to the CGPA from the first turn; the response remains grounded and cited.",
        "relevant_sources": ["Hemanga Resume.pdf"],
        "conversation": "resume_follow_up",
    },
    {
        "id": "14a",
        "question": "What phase covers vector store setup?",
        "expected": "Phase 2.",
        "relevant_sources": ["MultiDocChat_Implementation_Prompt.md"],
        "conversation": "phase_follow_up",
    },
    {
        "id": "14b",
        "question": "What about the one after it?",
        "expected": "A reasonable resolution to Phase 2b (or Phase 3 if explicitly justified), with a citation.",
        "relevant_sources": ["MultiDocChat_Implementation_Prompt.md"],
        "conversation": "phase_follow_up",
    },
]
