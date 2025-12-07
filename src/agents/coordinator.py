"""Coordinator agent orchestrating parser, retriever, reviewer, and rating agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

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
            paper_text=paper_text or "",
            progress_log=[],
        )

        # Parse
        parsed = self.parser_agent.parse(pdf_path=paper_pdf_path, raw_text=state.get("paper_text"))
        state["paper_text"] = parsed.get("full_text", state.get("paper_text", ""))
        state["parsed_sections"] = {k: v for k, v in parsed.items() if k != "full_text"}
        state["progress_log"].append("Coordinator: Parser Agent completed.")

        # Retrieve similar reviews
        (
            state["retrieved_reviews"],
            state["retrieved_metadatas"],
            state["retrieved_abstracts"],
            state["retrieved_paper_urls"],
            state["retrieved_pdf_urls"],
            state["retrieved_pdf_paths"],
        ) = self.retriever_agent.retrieve_similar_reviews(state["paper_text"], k=3)
        state["progress_log"].append("Coordinator: RAG Agent retrieved references.")

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

        # Generate review
        review = self.reviewer_agent.generate_review(
            paper_text=state["paper_text"],
            similar_reviews=state["retrieved_reviews"],
            arxiv_references=state.get("arxiv_reference_texts", []),
            paper_title=paper_title,
        )
        state["draft_review"] = review
        state["final_review"] = review
        state["progress_log"].append("Coordinator: Reviewer Agent drafted report.")

        # Predict rating
        rating, rationale = self.rating_agent.predict_rating(review)
        state["predicted_rating"] = rating
        state["rating_rationale"] = rationale
        state["progress_log"].append("Coordinator: Rating Agent finalized the score.")

        return state

    def get_summary(self, state: AgentState) -> Dict[str, Optional[str]]:
        return {
            "title": state.get("paper_title"),
            "rating": state.get("predicted_rating"),
            "rationale": state.get("rating_rationale"),
            "progress": state.get("progress_log", []),
        }
