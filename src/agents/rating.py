"""Rating agent that normalizes review scores and produces rationales."""

from __future__ import annotations

import re
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from .llm import get_llm
from .state import AgentState


class RatingAgent:
    def __init__(self, llm: Optional[Runnable] = None) -> None:
        self.llm = llm or get_llm()
        self.prompt = ChatPromptTemplate.from_template(
            """Given the following draft review, provide a single line containing:
Rating: <float between 1 and 10>
Rationale: <one sentence justification>

Review:
{review}
"""
        )

    def predict_rating(self, review_text: str) -> tuple[float, str]:
        extracted = self._extract_rating(review_text)
        if extracted is not None:
            return extracted, "Extracted from reviewer output."

        if not self.llm:
            return 5.0, "LLM unavailable. Defaulted to neutral rating."

        chain = self.prompt | self.llm | StrOutputParser()
        response = chain.invoke({"review": review_text})
        score = self._extract_rating(response) or 5.0
        return score, response.strip()

    def _extract_rating(self, text: str) -> Optional[float]:
        rating_match = re.search(r"rating[^0-9]*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
        if rating_match:
            try:
                score = float(rating_match.group(1))
            except ValueError:
                return None
            return max(1.0, min(10.0, score))
        return None


def rating_node(state: AgentState) -> AgentState:
    reviewer_output = state.get("final_review") or state.get("draft_review") or ""
    rating_agent = RatingAgent()
    predicted_rating, rationale = rating_agent.predict_rating(reviewer_output)

    progress = list(state.get("progress_log", []))
    progress.append("Rating Agent: predicted final score.")

    return AgentState(
        predicted_rating=predicted_rating,
        rating_rationale=rationale,
        progress_log=progress,
    )
