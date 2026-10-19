"""RAG kinds for this project — a vector search over the files in notes/.

promptmap imports this file before the models render, so every
``@promptmap.rag_kind("name")`` here is callable as ``rag_<name>(...)`` in any
.prompt file:

    {% for hit in rag_notes(ref('question'), k=3) %}{{ hit }}{% endfor %}

The index is built on the first call and rebuilt when a note changes. It is a
plain in-memory vector store so the example runs with no extra installs; to use
a real one, keep the shape and swap the two marked pieces:

    embed()      → your embedding model, e.g.
                   genai.Client().models.embed_content(model="gemini-embedding-001", contents=texts)
    _NotesIndex  → a vector DB, e.g. chromadb.PersistentClient(".chroma"):
                   collection.upsert(ids, documents, embeddings) on build,
                   collection.query(query_embeddings=[...], n_results=k) on search

Pay the indexing cost once: persist the store (chroma does) or key it on file
mtimes as below — the rendered prompt, hits included, is what promptmap caches,
so an unchanged index means unchanged prompts and cache hits downstream.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import promptmap

NOTES_DIR = Path(__file__).parent / "notes"
DIMS = 512
_WORD = re.compile(r"[a-z0-9]+")


# ---------------------------------------------------------------------------
# 1. Embedding — swap for a real embedding model
# ---------------------------------------------------------------------------

def embed(texts: list[str]) -> list[list[float]]:
    """Hashed bag-of-words vectors, unit length. Good enough to find keyword overlap."""
    vectors = []
    for text in texts:
        vec = [0.0] * DIMS
        for word in _WORD.findall(text.lower()):
            if len(word) > 2:
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIMS] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        vectors.append([v / norm for v in vec])
    return vectors


# ---------------------------------------------------------------------------
# 2. Vector store — swap for chroma / lancedb / pgvector / ...
# ---------------------------------------------------------------------------

class _NotesIndex:
    """Paragraph chunks of every note, with their vectors; rebuilt when a file changes."""

    def __init__(self) -> None:
        self._stamp: tuple = ()
        self._chunks: list[str] = []
        self._vectors: list[list[float]] = []

    def _build(self) -> None:
        files = sorted(NOTES_DIR.glob("**/*.md"))
        stamp = tuple((str(f), f.stat().st_mtime_ns) for f in files)
        if stamp == self._stamp:
            return
        self._chunks = [
            f"[{f.name}] {para.strip()}"
            for f in files
            for para in f.read_text(encoding="utf-8").split("\n\n")
            if para.strip() and not para.lstrip().startswith("#")
        ]
        self._vectors = embed(self._chunks)
        self._stamp = stamp

    def search(self, query: str, k: int) -> list[str]:
        self._build()
        [q] = embed([query])
        scored = sorted(
            ((sum(a * b for a, b in zip(q, v)), chunk) for v, chunk in zip(self._vectors, self._chunks)),
            reverse=True,
        )
        return [chunk for score, chunk in scored[:k] if score > 0]


_index = _NotesIndex()


# ---------------------------------------------------------------------------
# 3. Register — this is the only promptmap-specific part
# ---------------------------------------------------------------------------

@promptmap.rag_kind("notes")
def search_notes(query: str, k: int = 3) -> list[str]:
    """The *k* note paragraphs closest to *query*, each tagged with its file name."""
    return _index.search(str(query), int(k)) or None  # None → [False]: "no results"
