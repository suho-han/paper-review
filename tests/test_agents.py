"""Unit tests for the multi-agent components."""

from __future__ import annotations
from src.agents.reviewer import reviewer_node
from src.agents.retriever import retrieve_node
from src.agents.rating import RatingAgent
from src.agents.parser import ParserAgent
from src.agents.arxiv import ArxivPaperRetriever, arxiv_node
from src.agents import AgentState, CoordinatorAgent

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_parser_agent_extracts_key_sections():
    agent = ParserAgent()
    sample_text = (
        "Abstract\nThis paper studies transformers.\n"
        "Methodology\nWe introduce a sparse attention scheme.\n"
        "Experiments\nResults show state-of-the-art accuracy.\n"
        "Conclusion\nSummary text.\n"
    )
    sections = agent.parse(pdf_path=None, raw_text=sample_text)

    assert sections["abstract"].startswith("Abstract")
    assert "Methodology" in sections["methodology"]
    assert "Experiments" in sections["experiments"]
    assert sections["full_text"].startswith("Abstract")


def test_retrieve_node_returns_expected_state(monkeypatch):
    class FakeRetriever:
        def __init__(self, *args, **kwargs):  # pragma: no cover - simple stub
            pass

        def retrieve_similar_reviews(self, *args, **kwargs):
            return (
                ["great paper"],
                [{"forum_id": "abc123"}],
                ["abstract text"],
                ["https://openreview.net/forum?id=abc123"],
                ["https://openreview.net/pdf?id=abc123"],
                ["outputs/pdfs/abc123.pdf"],
            )

    monkeypatch.setattr("src.agents.retriever.ReviewRetriever", FakeRetriever)

    state = AgentState(paper_text="transformer advances")
    result = retrieve_node(state)

    assert result["retrieved_reviews"] == ["great paper"]
    assert result["retrieved_paper_urls"][0].endswith("abc123")
    assert "RAG Agent" in result["progress_log"][-1]


def test_arxiv_node_updates_state(monkeypatch):
    class FakeArxiv:
        def __init__(self, *args, **kwargs):
            pass

        def retrieve_similar_papers(self, **kwargs):
            return (
                [{"title": "Latest Vision Model"}],
                ["Title: Latest Vision Model"],
                "vision model query",
                False,
            )

    monkeypatch.setattr("src.agents.arxiv.ArxivPaperRetriever", FakeArxiv)

    state = AgentState(paper_text="vision transformer", parsed_sections={"abstract": "Vision study"})
    result = arxiv_node(state)

    assert result["arxiv_results"][0]["title"] == "Latest Vision Model"
    assert result["arxiv_query"] == "vision model query"
    assert result["arxiv_cache_hit"] is False
    assert "ArXiv Agent" in result["progress_log"][-1]


def test_reviewer_node_receives_arxiv_context(monkeypatch):
    captured = {}

    def fake_generate_review(self, *, paper_text, similar_reviews, arxiv_references=None, paper_title=None):
        captured["paper_text"] = paper_text
        captured["reviews"] = similar_reviews
        captured["arxiv"] = arxiv_references
        captured["title"] = paper_title
        return "Structured Review"

    monkeypatch.setattr("src.agents.reviewer.ReviewerAgent.generate_review", fake_generate_review)

    state = AgentState(
        paper_text="Novel model",
        paper_title="Novel Model",
        retrieved_reviews=["Legacy work"],
        arxiv_reference_texts=["Title: Similar Work"],
    )
    result = reviewer_node(state)

    assert captured["reviews"] == ["Legacy work"]
    assert captured["arxiv"] == ["Title: Similar Work"]
    assert result["final_review"] == "Structured Review"
    assert "Reviewer Agent" in result["progress_log"][-1]


def test_rating_agent_extracts_inline_score():
    agent = RatingAgent(llm=None)
    score, rationale = agent.predict_rating("Overall impression. Rating: 7.5 because...")

    assert pytest.approx(score, 0.01) == 7.5
    assert rationale == "Extracted from reviewer output."


def test_arxiv_retriever_caches_results(tmp_path, monkeypatch):
    cache_file = tmp_path / "cache.json"
    log_file = tmp_path / "queries.log"
    call_count = {"value": 0}

    def fake_search(self, query):
        call_count["value"] += 1
        return [{
            "title": f"Paper {call_count['value']}",
            "summary": "",
            "authors": [],
            "published": "",
            "url": "",
            "pdf_url": "",
        }]

    monkeypatch.setattr(ArxivPaperRetriever, "_search", fake_search, raising=False)

    retriever = ArxivPaperRetriever(cache_path=cache_file, log_path=log_file)

    _, _, _, cache_hit_first = retriever.retrieve_similar_papers(title="Cache Test", abstract=None, raw_text="")
    assert call_count["value"] == 1
    assert cache_hit_first is False

    _, _, _, cache_hit_second = retriever.retrieve_similar_papers(title="Cache Test", abstract=None, raw_text="")
    assert call_count["value"] == 1  # cache served result
    assert cache_hit_second is True
    assert cache_file.exists()
    assert log_file.exists()


def test_coordinator_agent_runs_with_stubbed_dependencies():
    class StubParser:
        def parse(self, pdf_path, raw_text):
            return {"full_text": raw_text or "parser text", "abstract": "parser abstract"}

    class StubRetriever:
        def retrieve_similar_reviews(self, *args, **kwargs):
            return (
                ["historical review"],
                [{}],
                ["abstract"],
                ["paper-url"],
                ["pdf-url"],
                ["pdf-path"],
            )

    class StubArxiv:
        def retrieve_similar_papers(self, **kwargs):
            return ([{"title": "Parallel work"}], ["Title: Parallel work"], "parallel query", True)

    class StubReviewer:
        def generate_review(self, **kwargs):
            return "coordinated review"

    class StubRating:
        def predict_rating(self, review_text):
            return 8.0, "consistent with evidence"

    coordinator = CoordinatorAgent(
        parser_agent=StubParser(),
        retriever_agent=StubRetriever(),
        arxiv_agent=StubArxiv(),
        reviewer_agent=StubReviewer(),
        rating_agent=StubRating(),
    )

    final_state = coordinator.run(paper_text="input text", paper_title="Demo Title")

    assert final_state["final_review"] == "coordinated review"
    assert final_state["predicted_rating"] == 8.0
    assert final_state["arxiv_reference_texts"] == ["Title: Parallel work"]
    assert final_state["arxiv_cache_hit"] is True
    assert final_state["progress_log"][-1] == "Coordinator: Rating Agent finalized the score."
