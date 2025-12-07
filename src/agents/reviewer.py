"""Reviewer agent responsible for drafting structured reviews."""

from __future__ import annotations

from typing import List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from .llm import get_llm
from .state import AgentState


class ReviewerAgent:
    def __init__(
        self,
        llm: Optional[Runnable] = None,
        *,
        max_paper_tokens: int = 5500,
        max_review_tokens: int = 1000,
        max_arxiv_tokens: int = 1000,
        max_similar_reviews: int = 3,
        max_arxiv_refs: int = 3,
    ) -> None:
        self.llm = llm or get_llm()
        self.max_paper_tokens = max_paper_tokens
        self.max_review_tokens = max_review_tokens
        self.max_arxiv_tokens = max_arxiv_tokens
        self.max_similar_reviews = max_similar_reviews
        self.max_arxiv_refs = max_arxiv_refs
        self.prompt = ChatPromptTemplate.from_template(
            """You are an expert reviewer for top-tier AI conferences (ICLR, NeurIPS, ICML).
Use the retrieved human-written reviews strictly as reference for tone and rigor.

Context reviews:
{context}

Similar arXiv papers:
{arxiv_context}

Paper text:
{paper_text}

Title: {paper_title}

Provide a review with the following structure:
1. Summary
2. Strengths
3. Weaknesses
4. Rating (1-10)
5. Detailed Review
"""
        )

    def generate_review(
        self,
        *,
        paper_text: str,
        similar_reviews: List[str],
        arxiv_references: Optional[List[str]] = None,
        paper_title: Optional[str] = None,
    ) -> str:
        if not self.llm:
            return "LLM not configured."
        truncated_reviews = self._prepare_context(similar_reviews, self.max_similar_reviews, self.max_review_tokens)
        truncated_arxiv = self._prepare_context(arxiv_references or [], self.max_arxiv_refs, self.max_arxiv_tokens)
        context = "\n\n".join(truncated_reviews) if truncated_reviews else "(No similar reviews found)"
        arxiv_context = "\n\n".join(truncated_arxiv) if truncated_arxiv else "(No arXiv matches)"
        truncated_paper_text = self._truncate_text(paper_text, self.max_paper_tokens)
        chain = self.prompt | self.llm | StrOutputParser()
        return chain.invoke(
            {
                "context": context,
                "arxiv_context": arxiv_context,
                "paper_text": truncated_paper_text,
                "paper_title": paper_title or "Unknown Title",
            }
        )

    def _prepare_context(self, items: List[str], max_items: int, max_tokens: int) -> List[str]:
        # Keep only the most relevant chunks and trim them to avoid exceeding context limits.
        return [self._truncate_text(item, max_tokens) for item in items[:max_items]]

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        max_chars = max_tokens * 4  # Rough token-to-char conversion.
        if len(text) <= max_chars:
            return text
        truncated = text[:max_chars]
        return f"{truncated}\n...[truncated]"


def reviewer_node(state: AgentState) -> AgentState:
    print("--- GENERATE REVIEW ---")
    reviewer = ReviewerAgent()
    review = reviewer.generate_review(
        paper_text=state.get("paper_text", ""),
        similar_reviews=state.get("retrieved_reviews", []),
        arxiv_references=state.get("arxiv_reference_texts", []),
        paper_title=state.get("paper_title"),
    )

    progress = list(state.get("progress_log", []))
    progress.append("Reviewer Agent: drafted review.")

    return AgentState(draft_review=review, final_review=review, progress_log=progress)
