"""
tests/conftest.py
=================
Shared fixtures: a FakeClient-backed LLM and helpers to build sources.
Ensures the project root is importable and no network/keys are needed.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.llm import LLM, FakeClient, set_llm  # noqa: E402
from models.state import RetrievedSource, SourceType  # noqa: E402


def make_source(title, url, source_type=SourceType.WEB, content="content", relevance=0.6):
    return RetrievedSource(
        title=title, url=url, source_type=source_type, content=content,
        relevance_score=relevance, credibility_score=0.5,
    )


@pytest.fixture
def sources():
    return [
        make_source("A", "https://a.com/1", SourceType.RESEARCH_PAPER, "AI speeds code generation"),
        make_source("B", "https://b.com/2", SourceType.REPORT, "Developer demand may grow"),
        make_source("C", "https://c.com/3", SourceType.NEWS, "AI code needs review"),
    ]


@pytest.fixture
def fake_llm_factory():
    """Return a function that installs a FakeClient returning the given responder."""
    def _install(responder):
        set_llm(LLM(FakeClient(responder)))
    yield _install
    set_llm(LLM(None))  # reset after test


@pytest.fixture
def no_llm():
    set_llm(LLM(None))
    yield
    set_llm(LLM(None))
