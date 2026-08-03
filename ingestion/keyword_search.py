"""BM25-based keyword search for MultiDocChat."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

from langchain_core.documents import Document

_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if",
    "in", "into", "is", "it", "no", "not", "of", "on", "or", "such",
    "that", "the", "their", "then", "there", "these", "they", "this",
    "to", "was", "will", "with"
}

def tokenize(text: str) -> list[str]:
    """Tokenize text by lowercasing, splitting on non-alphanumeric, and removing stopwords."""
    text = text.lower()
    tokens = re.split(r'\W+', text)
    return [t for t in tokens if t and t not in _STOP_WORDS]

class KeywordIndex:
    """BM25 keyword search index."""

    def __init__(self, docs: Iterable[Document], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = list(docs)
        self.doc_tokens = [tokenize(doc.page_content) for doc in self.docs]
        
        self.num_docs = len(self.docs)
        if self.num_docs == 0:
            self.avg_doc_len = 0
            self.doc_lens = []
            self.term_freqs = []
            self.idf = {}
            return
            
        self.doc_lens = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_len = sum(self.doc_lens) / self.num_docs
        
        self.term_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        
        # Calculate IDF
        doc_count_per_term: Counter[str] = Counter()
        for tf in self.term_freqs:
            for term in tf:
                doc_count_per_term[term] += 1
                
        self.idf = {}
        for term, doc_count in doc_count_per_term.items():
            # Standard Okapi BM25 IDF formulation
            self.idf[term] = math.log(1 + (self.num_docs - doc_count + 0.5) / (doc_count + 0.5))
            
    def _score_doc(self, query_tokens: list[str], doc_idx: int) -> float:
        """Calculate BM25 score for a document given query tokens."""
        if self.avg_doc_len == 0:
            return 0.0
            
        score = 0.0
        tf = self.term_freqs[doc_idx]
        doc_len = self.doc_lens[doc_idx]
        
        for term in query_tokens:
            if term not in tf:
                continue
                
            term_freq = tf[term]
            idf = self.idf.get(term, 0.0)
            
            # BM25 tf normalization
            numerator = term_freq * (self.k1 + 1)
            denominator = term_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
            
            score += idf * (numerator / denominator)
            
        return score

    def search(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        """Search the index and return top-k documents with BM25 scores."""
        query_tokens = tokenize(query)
        if not query_tokens or self.num_docs == 0:
            return []
            
        scores = [
            (doc, self._score_doc(query_tokens, i))
            for i, doc in enumerate(self.docs)
        ]
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
        
    def search_per_source(self, query: str, k_per_source: int = 5) -> list[tuple[Document, float]]:
        """Retrieve top-k documents per source file with BM25 scores."""
        query_tokens = tokenize(query)
        if not query_tokens or self.num_docs == 0:
            return []
            
        # Group by source file
        scores_by_source: dict[str, list[tuple[Document, float]]] = {}
        for i, doc in enumerate(self.docs):
            source = doc.metadata.get("source_file", "")
            if source not in scores_by_source:
                scores_by_source[source] = []
            score = self._score_doc(query_tokens, i)
            if score > 0: # Only include matching docs
                scores_by_source[source].append((doc, score))
            
        results = []
        for source, source_scores in scores_by_source.items():
            source_scores.sort(key=lambda x: x[1], reverse=True)
            results.extend(source_scores[:k_per_source])
            
        # Sort the overall results just in case, though per source retrieval shouldn't strictly require it
        results.sort(key=lambda x: x[1], reverse=True)
        return results
