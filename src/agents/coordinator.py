"""Coordinator agent orchestrating parser, retriever, reviewer, and rating agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Optional

from .arxiv import ArxivPaperRetriever
from .parser import ParserAgent
from .rating import RatingAgent
from .retriever import ReviewRetriever
from .reviewer import ReviewerAgent
from .state import AgentState


@dataclass
class CoordinatorAgent:
    llm: Optional[Any] = None
    parser_agent: ParserAgent = field(default_factory=ParserAgent)
    retriever_agent: ReviewRetriever = field(default_factory=ReviewRetriever)
    arxiv_agent: ArxivPaperRetriever = field(default_factory=ArxivPaperRetriever)
    reviewer_agent: ReviewerAgent = field(default_factory=ReviewerAgent)
    rating_agent: RatingAgent = field(default_factory=RatingAgent)

    def run(
        self,
        *,
        paper_text: Optional[str] = None,
        paper_pdf_path: Optional[str] = None,
        paper_title: Optional[str] = None,
    ) -> AgentState:
        state: AgentState = AgentState(
            paper_title=paper_title,
            paper_pdf_path=paper_pdf_path,
            agent_traces={},
            paper_text=paper_text or "",
            progress_log=[],
        )

        for streamed_state in self.run_stream(
            paper_text=paper_text,
            paper_pdf_path=paper_pdf_path,
            paper_title=paper_title,
        ):
            state = streamed_state

        return state

    def run_stream(
        self,
        *,
        paper_text: Optional[str] = None,
        paper_pdf_path: Optional[str] = None,
        paper_title: Optional[str] = None,
    ) -> Iterator[AgentState]:
        """Run the pipeline and yield state snapshots as stages complete."""

        state: AgentState = AgentState(
            paper_title=paper_title,
            paper_pdf_path=paper_pdf_path,
            agent_traces={},
            paper_text=paper_text or "",
            progress_log=[],
        )

        # Parse
        state["progress_log"].append("Coordinator: Parser Agent starting...")
        yield dict(state)
        parsed = self.parser_agent.parse(pdf_path=paper_pdf_path, raw_text=state.get("paper_text"))
        state["paper_text"] = parsed.get("full_text", state.get("paper_text", ""))
        state["parsed_sections"] = {k: v for k, v in parsed.items() if k != "full_text"}
        state["progress_log"].append("Coordinator: Parser Agent completed.")
        state["agent_traces"].setdefault("parser", {})
        state["agent_traces"]["parser"].update({
            "input_type": "pdf" if paper_pdf_path else "text",
            "sections_found": list(state.get("parsed_sections", {}).keys()),
            "text_chars": len(state.get("paper_text", "")),
        })

        yield dict(state)

        # Retrieve similar reviews
        state["progress_log"].append("Coordinator: RAG Agent starting retrieval...")
        yield dict(state)
        (
            state["retrieved_reviews"],
            state["retrieved_metadatas"],
            state["retrieved_abstracts"],
            state["retrieved_paper_urls"],
            state["retrieved_pdf_urls"],
            state["retrieved_pdf_paths"],
        ) = self.retriever_agent.retrieve_similar_reviews(state["paper_text"], k=3)
        state["progress_log"].append("Coordinator: RAG Agent retrieved references.")
        state["agent_traces"].setdefault("retriever", {})
        state["agent_traces"]["retriever"].update({
            "query_preview": state.get("paper_text", "")[:600],
            "top_results": [
                {
                    "forum_id": meta.get("forum_id"),
                    "collection": meta.get("collection_name"),
                    "distance": meta.get("distance"),
                    "rating": meta.get("rating"),
                }
                for meta in state.get("retrieved_metadatas", [])
            ],
        })

        yield dict(state)

        state["progress_log"].append("Coordinator: ArXiv Agent starting retrieval...")
        yield dict(state)
        (
            state["arxiv_results"],
            state["arxiv_reference_texts"],
            state["arxiv_query"],
            state["arxiv_cache_hit"],
        ) = self.arxiv_agent.retrieve_similar_papers(
            title=paper_title,
            abstract=state.get("parsed_sections", {}).get("abstract"),
            raw_text=state["paper_text"],
        )
        state["progress_log"].append("Coordinator: ArXiv Agent fetched related literature.")
        state["agent_traces"].setdefault("arxiv", {})
        state["agent_traces"]["arxiv"].update({
            "query": state.get("arxiv_query"),
            "cache_hit": state.get("arxiv_cache_hit"),
            "titles": [paper.get("title", "") for paper in state.get("arxiv_results", [])],
        })

        yield dict(state)

        # Generate review
        review_result = None
        state["progress_log"].append("Coordinator: Reviewer Agent starting...")
        yield dict(state)
        try:
            review_result = self.reviewer_agent.generate_review(
                paper_text=state["paper_text"],
                similar_reviews=state["retrieved_reviews"],
                arxiv_references=state.get("arxiv_reference_texts", []),
                paper_title=paper_title,
                return_trace=True,
            )
        except TypeError:
            review_result = self.reviewer_agent.generate_review(
                paper_text=state["paper_text"],
                similar_reviews=state["retrieved_reviews"],
                arxiv_references=state.get("arxiv_reference_texts", []),
                paper_title=paper_title,
            )

        if isinstance(review_result, tuple) and len(review_result) == 2:
            review, review_trace = review_result
        else:
            review = str(review_result)
            review_trace = {}
        state["draft_review"] = review
        state["final_review"] = review
        state["progress_log"].append("Coordinator: Reviewer Agent drafted report.")
        state["agent_traces"]["reviewer"] = review_trace

        yield dict(state)

        # Predict rating
        rating_result = None
        state["progress_log"].append("Coordinator: Rating Agent starting...")
        yield dict(state)
        try:
            rating_result = self.rating_agent.predict_rating(review, return_trace=True)
        except TypeError:
            rating_result = self.rating_agent.predict_rating(review)

        if isinstance(rating_result, tuple) and len(rating_result) == 3:
            rating, rationale, rating_trace = rating_result
        elif isinstance(rating_result, tuple) and len(rating_result) == 2:
            rating, rationale = rating_result
            rating_trace = {}
        else:
            rating = None
            rationale = ""
            rating_trace = {"error": "Unexpected rating_agent return type"}
        state["predicted_rating"] = rating
        state["rating_rationale"] = rationale
        if isinstance(rating_trace, dict):
            breakdown = rating_trace.get("breakdown")
            if isinstance(breakdown, dict):
                # Keep only numeric entries.
                clean_breakdown = {}
                for k, v in breakdown.items():
                    try:
                        clean_breakdown[str(k)] = float(v)
                    except Exception:
                        continue
                if clean_breakdown:
                    state["rating_breakdown"] = clean_breakdown
        state["progress_log"].append("Coordinator: Rating Agent finalized the score.")
        state["agent_traces"]["rating"] = rating_trace

        yield dict(state)

    def get_summary(self, state: AgentState) -> Dict[str, Optional[str]]:
        return {
            "title": state.get("paper_title"),
            "rating": state.get("predicted_rating"),
            "rationale": state.get("rating_rationale"),
            "progress": state.get("progress_log", []),
        }
