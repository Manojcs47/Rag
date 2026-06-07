"""Shared pytest fixtures: a tiny self-contained corpus + settings + a store factory.

Tests never touch the real 165 MB corpus or the network. The fixture corpus carries
the same noise patterns (MDX, anchors, Lil'Log cruft, References) so parsing/chunking
are exercised against realistic input.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from research_navigator.config import Settings
from research_navigator.ingest.qdrant_store import QdrantStore

_HF_MD = """# Hugging Face NLP Course — Chapter 9

Source: https://huggingface.co/learn/nlp-course/chapter9

---

<!-- Section 9.1 -->

# Introduction[[introduction]]

<CourseFloatingBanner chapter={9} classNames="absolute" />

<Youtube id="abc123" />

Transformers are a family of neural network architectures. They rely on attention.

## Setup[[setup]]

Install the library and import it:

```python
from transformers import pipeline
clf = pipeline("sentiment-analysis")
```

That code block must survive chunking intact.
"""

_LILLOG_MD = """# Prompt Engineering

Author: Lilian Weng
Published: 2023-03-15
Source: https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/

---

#  Prompt Engineering

Date: March 15, 2023 | Estimated Reading Time: 21 min | Author: Lilian Weng

Table of Contents

  * Basic Prompting
  * Few-shot

Prompt engineering refers to methods for communicating with LLMs to steer behaviour.

# Basic Prompting#

Zero-shot and few-shot prompting are the two most basic approaches.

# References

[1] Some Author. "A Paper." 2022.

  * [Nlp](<https://lilianweng.github.io/tags/nlp/>)
  * [Prompting](<https://lilianweng.github.io/tags/prompting/>)
"""

_DOCS = [
    {
        "doc_id": "hf-nlp-ch09",
        "content_type": "course_chapter",
        "title": "HF NLP Course — Chapter 9",
        "authors": ["Hugging Face"],
        "year": 2023,
        "month": None,
        "primary_category": "course",
        "secondary_categories": [],
        "tags": ["nlp", "transformers"],
        "is_foundational": False,
        "citation_count": None,
        "source_url": "https://huggingface.co/learn/nlp-course/chapter9",
        "local_path": "documents/hf-learn/hf-nlp-ch09.md",
    },
    {
        "doc_id": "lillog-prompt-eng-2023-03",
        "content_type": "survey_blog",
        "title": "Prompt Engineering",
        "authors": ["Lilian Weng"],
        "year": 2023,
        "month": 3,
        "primary_category": "survey",
        "secondary_categories": [],
        "tags": ["prompting", "nlp"],
        "is_foundational": False,
        "citation_count": None,
        "source_url": "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "local_path": "documents/lillog/lillog-prompt-eng-2023-03.md",
    },
]


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    """Materialise the tiny fixture corpus on disk and return its root."""
    docs = tmp_path / "documents"
    (docs / "hf-learn").mkdir(parents=True)
    (docs / "lillog").mkdir(parents=True)
    (docs / "hf-learn" / "hf-nlp-ch09.md").write_text(_HF_MD, encoding="utf-8")
    (docs / "lillog" / "lillog-prompt-eng-2023-03.md").write_text(_LILLOG_MD, encoding="utf-8")
    manifest = {"schema_version": "1.0", "generated_for": "tests", "documents": _DOCS}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


@pytest.fixture
def settings(corpus_dir: Path) -> Settings:
    """Settings pointing at the fixture corpus, in-memory Qdrant, offline embedder."""
    return Settings(
        corpus_dir=corpus_dir,
        manifest_path=corpus_dir / "manifest.json",
        qdrant_url=":memory:",
        use_offline_embedder=True,
        dense_dim=64,
    )


@pytest.fixture
def store(settings: Settings) -> Iterator[QdrantStore]:
    """A reusable in-memory Qdrant store (survives across ingest calls in one test)."""
    yield QdrantStore(settings)


@pytest.fixture
def hf_md() -> str:
    return _HF_MD


@pytest.fixture
def lillog_md() -> str:
    return _LILLOG_MD
