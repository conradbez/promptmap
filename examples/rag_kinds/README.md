# RAG kinds

A RAG kind is a named search function every template can call as
`rag_<name>(...)`. This example registers one of its own, `rag_notes`, a vector
search over the files in `notes/`. It can also use `rag_brave_search`, a web
search from `promptmap-extra-model-kinds`.

```bash
export GEMINI_API_KEY=your_key_here
promptmap run
promptmap run --promptdata question="When is standup?"

# Add web results too (needs `pip install "promptmap[extras]"` and BRAVE_API_KEY):
promptmap run --promptdata web=1
```

## What is here

| File | What it does |
|---|---|
| `rag.py` | Registers `notes` with `@promptmap.rag_kind("notes")`: embeds note paragraphs, returns the `k` closest |
| `notes/*.md` | The documents searched |
| `map/question.prompt` | The question (`template` kind, no LLM call) |
| `map/answer.prompt` | Calls `rag_notes(ref('question'), k=3)`, and `rag_brave_search(...)` when `web` is set |
| `client.py` | The LLM backend |

## Registering your own

```python
# rag.py, beside map/
import promptmap

@promptmap.rag_kind("notes")
def search_notes(query: str, k: int = 3) -> list[str]:
    ...
```

- Arguments are whatever the template passes, keywords included.
- Return `list[str]`, a bare `str` (wrapped), or `None` / `False` for "no results" (becomes `[False]`, so `{% if hits[0] %}` is safe).
- The function is a plain `def`: templates render synchronously.

`rag.py` uses a small in-memory vector store so it runs with no installs. Its
docstring shows where to swap in a real embedding model and vector DB (e.g.
Chroma); only the `@promptmap.rag_kind` line is promptmap-specific.

A package ships RAG kinds the same way it ships model kinds, under its own
entry point group:

```toml
[project.entry-points."promptmap.rag_kinds"]
brave_search = "promptmap_extra_model_kinds.brave_search"
```
