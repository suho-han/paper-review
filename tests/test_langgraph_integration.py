"""Integration tests for the LangGraph workflow."""

from __future__ import annotations
from src.agents import AgentState, build_agent_graph

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_langgraph_workflow_with_stubs(monkeypatch):
    class DummyParser:
        def __init__(self, *args, **kwargs):
            pass

        def parse(self, pdf_path, raw_text):
            return {"full_text": raw_text or "parsed", "abstract": "Stub abstract"}

    class DummyRetriever:
        def __init__(self, *args, **kwargs):
            pass

        def retrieve_similar_reviews(self, *args, **kwargs):
            return (
                ["Reference review"],
                [{}],
                ["Reference abstract"],
                ["paper-url"],
                ["pdf-url"],
                ["pdf-path"],
            )

    class DummyArxiv:
        def __init__(self, *args, **kwargs):
            pass

        def retrieve_similar_papers(self, **kwargs):
            return (
                [{"title": "Parallel study"}],
                ["Title: Parallel study"],
                "stub query",
                False,
            )

    class DummyReviewer:
        def __init__(self, *args, **kwargs):
            pass

        def generate_review(self, **kwargs):
            return "Integration review"

    class DummyRating:
        def __init__(self, *args, **kwargs):
            pass

        def predict_rating(self, review_text):
            return 9.0, "High confidence"

    monkeypatch.setattr("src.agents.parser.ParserAgent", DummyParser)
    monkeypatch.setattr("src.agents.retriever.ReviewRetriever", DummyRetriever)
    monkeypatch.setattr("src.agents.arxiv.ArxivPaperRetriever", DummyArxiv)
    monkeypatch.setattr("src.agents.reviewer.ReviewerAgent", DummyReviewer)
    monkeypatch.setattr("src.agents.rating.RatingAgent", DummyRating)

    app = build_agent_graph()
    result = app.invoke(AgentState(paper_text="Input", paper_title="Stub Title"))

    assert result["final_review"] == "Integration review"
    assert result["predicted_rating"] == 9.0
    assert result["retrieved_reviews"] == ["Reference review"]
    assert result["arxiv_reference_texts"] == ["Title: Parallel study"]
    assert result["arxiv_cache_hit"] is False
