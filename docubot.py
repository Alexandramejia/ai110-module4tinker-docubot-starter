"""
Core DocuBot class responsible for:
- Loading documents from the docs/ folder
- Building a simple retrieval index (Phase 1)
- Retrieving relevant snippets (Phase 1)
- Supporting retrieval only answers
- Supporting RAG answers when paired with Gemini (Phase 2)
"""

import os
import glob
import re
from collections import Counter

class DocuBot:
    # Common words that carry no real evidence of topical relevance.
    STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "of", "to", "in", "on", "for", "and", "or", "but", "with", "as",
        "at", "by", "from", "it", "its", "this", "that", "these", "those",
        "what", "how", "why", "does", "do", "did", "can", "could",
        "should", "would", "i", "you", "we", "they", "about", "into",
    }

    # Fraction of the query's meaningful (non-stopword) words that must
    # actually appear in a snippet before it counts as usable evidence.
    # A single incidental common-word hit isn't enough on its own.
    MIN_EVIDENCE_RATIO = 0.5

    def __init__(self, docs_folder="docs", llm_client=None):
        """
        docs_folder: directory containing project documentation files
        llm_client: optional Gemini client for LLM based answers
        """
        self.docs_folder = docs_folder
        self.llm_client = llm_client

        # Load documents into memory
        self.documents = self.load_documents()  # List of (filename, text)

        # Build a retrieval index (implemented in Phase 1)
        self.index = self.build_index(self.documents)

    # -----------------------------------------------------------
    # Document Loading
    # -----------------------------------------------------------

    def load_documents(self):
        """
        Loads all .md and .txt files inside docs_folder.
        Returns a list of tuples: (filename, text)
        """
        docs = []
        pattern = os.path.join(self.docs_folder, "*.*")
        for path in glob.glob(pattern):
            if path.endswith(".md") or path.endswith(".txt"):
                with open(path, "r", encoding="utf8") as f:
                    text = f.read()
                filename = os.path.basename(path)
                docs.append((filename, text))
        return docs

    # -----------------------------------------------------------
    # Tokenization helper (shared by indexing, scoring, retrieval)
    # -----------------------------------------------------------

    def _tokenize(self, text):
        """Split text into lowercase alphanumeric word tokens."""
        tokens = []
        chars = []
        for char in text:
            code = ord(char)
            if 65 <= code <= 90:  # 'A'-'Z'
                char = chr(code + 32)
            if char.isalnum():
                chars.append(char)
            elif chars:
                tokens.append("".join(chars))
                chars = []
        if chars:
            tokens.append("".join(chars))
        return tokens

    def _split_into_paragraphs(self, text):
        """Split a document into blank-line-separated paragraphs."""
        chunks = re.split(r"\n\s*\n", text)
        return [chunk.strip() for chunk in chunks if chunk.strip()]

    # -----------------------------------------------------------
    # Index Construction (Phase 1)
    # -----------------------------------------------------------

    def build_index(self, documents):
        """
        Build a tiny inverted index mapping lowercase words to the documents
        they appear in.

        Example structure:
        {
            "token": ["AUTH.md", "API_REFERENCE.md"],
            "database": ["DATABASE.md"]
        }
        """
        index = {}
        for filename, text in documents:
            for token in self._tokenize(text):
                index.setdefault(token, [])
                if filename not in index[token]:
                    index[token].append(filename)
        return index

    # -----------------------------------------------------------
    # Scoring and Retrieval (Phase 2: precise snippets + guardrails)
    # -----------------------------------------------------------

    def _meaningful_tokens(self, query):
        """Query tokens with stopwords removed (the words that carry signal)."""
        return [t for t in self._tokenize(query) if t not in self.STOPWORDS]

    def score_document(self, query, text):
        """
        Score how well a snippet of text matches a query, for ranking
        purposes. Uses whole-word token matching (not substrings) and
        ignores stopwords, so common words can't manufacture false
        relevance.
        """
        text_counts = Counter(self._tokenize(text))
        score = 0
        for word in self._meaningful_tokens(query):
            score += text_counts[word]
        return score

    def _has_sufficient_evidence(self, query, text):
        """
        Guardrail: a snippet only counts as evidence if a real share of
        the query's meaningful words actually appear in it. This stops a
        single incidental common-word match (e.g. "like") from making an
        unrelated query look answerable.
        """
        query_tokens = set(self._meaningful_tokens(query))
        if not query_tokens:
            return False
        text_tokens = set(self._tokenize(text))
        matched = query_tokens & text_tokens
        return len(matched) / len(query_tokens) >= self.MIN_EVIDENCE_RATIO

    def retrieve(self, query, top_k=3):
        """
        Use the index and scoring function to select the top_k most
        relevant paragraph-sized snippets (not whole documents).

        Only snippets passing the evidence guardrail are eligible, so a
        query with no real support in the docs returns nothing.

        Return a list of (filename, snippet) sorted by score descending.
        """
        candidate_filenames = set()
        for word in self._tokenize(query):
            if word in self.index:
                candidate_filenames.update(self.index[word])

        scored = []
        for filename, text in self.documents:
            if candidate_filenames and filename not in candidate_filenames:
                continue
            for paragraph in self._split_into_paragraphs(text):
                if not self._has_sufficient_evidence(query, paragraph):
                    continue
                score = self.score_document(query, paragraph)
                scored.append((score, filename, paragraph))

        scored.sort(key=lambda item: item[0], reverse=True)
        results = [(filename, snippet) for _, filename, snippet in scored]
        return results[:top_k]

    # -----------------------------------------------------------
    # Answering Modes
    # -----------------------------------------------------------

    def answer_retrieval_only(self, query, top_k=3):
        """
        Phase 1 retrieval only mode.
        Returns raw snippets and filenames with no LLM involved.
        """
        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        formatted = []
        for filename, text in snippets:
            formatted.append(f"[{filename}]\n{text}\n")

        return "\n---\n".join(formatted)

    def answer_rag(self, query, top_k=3):
        """
        Phase 2 RAG mode.
        Uses student retrieval to select snippets, then asks Gemini
        to generate an answer using only those snippets.
        """
        if self.llm_client is None:
            raise RuntimeError(
                "RAG mode requires an LLM client. Provide a GeminiClient instance."
            )

        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        return self.llm_client.answer_from_snippets(query, snippets)

    # -----------------------------------------------------------
    # Bonus Helper: concatenated docs for naive generation mode
    # -----------------------------------------------------------

    def full_corpus_text(self):
        """
        Returns all documents concatenated into a single string.
        This is used in Phase 0 for naive 'generation only' baselines.
        """
        return "\n\n".join(text for _, text in self.documents)
