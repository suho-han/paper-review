"""Rating agent that normalizes review scores and produces rationales."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from .llm import get_llm
from .state import AgentState


class RatingAgent:
    def __init__(self, llm: Optional[Runnable] = None, *, use_default_llm: bool = True) -> None:
        # For scoring, slightly higher temperature helps avoid collapsing to a single
        # "safe" score (e.g., always 8) while still being stable.
        if llm is not None:
            self.llm = llm
        elif use_default_llm:
            self.llm = get_llm(temperature=0.6)
        else:
            self.llm = None
        self.max_prompt_tokens_for_trace = 400
        self.criteria_keys = [
            "novelty",
            "significance",
            "technical_quality",
            "clarity",
            "reproducibility",
        ]

        self.prompt_breakdown = ChatPromptTemplate.from_template(
            """Given the following draft review, output a single JSON object (no markdown) with keys:
- overall: number between 1 and 10
- criteria: object with numeric scores between 1 and 10 for keys: {criteria_keys}
- rationale: one sentence justification

Review:
{review}
"""
        )
        self.prompt = ChatPromptTemplate.from_template(
            """Given the following draft review, provide a single line containing:
Rating: <float between 1 and 10>
Rationale: <one sentence justification>

    Important: Do NOT default to 8. Base the rating on the evidence in the review.

Review:
{review}
"""
        )

    def predict_rating(self, review_text: str, *, return_trace: bool = False) -> tuple[float, str] | tuple[float, str, dict]:
        truncated_review = self._truncate_text(review_text, 600)

        # If no LLM is available, fall back to extracting a rating already present
        # in the review text (useful for offline runs / tests).
        if not self.llm:
            extracted = self._extract_rating(review_text)
            if extracted is not None:
                trace = {
                    "prompt": None,
                    "response": "Extracted from reviewer output.",
                    "input_excerpt": truncated_review,
                    "method": "inline_extraction",
                    "inline_rating": extracted,
                    "breakdown": {},
                }
                return (extracted, trace["response"], trace) if return_trace else (extracted, trace["response"])

        # When LLM is available, do NOT reuse the review's self-declared rating.
        # That tends to anchor the final score (often collapsing to a constant like 8).
        review_for_scoring = self._strip_existing_rating_sections(review_text)
        truncated_for_scoring = self._truncate_text(review_for_scoring, 600)

        # Try to obtain an overall score + per-criterion breakdown (best-effort).
        breakdown_trace: Dict[str, Any] = {}
        try:
            rendered_prompt = self._render_prompt_breakdown(truncated_for_scoring)
            chain = self.prompt_breakdown | self.llm | StrOutputParser()
            response = chain.invoke({"review": truncated_for_scoring, "criteria_keys": ", ".join(self.criteria_keys)})
            parsed = self._extract_json_object(response)

            if isinstance(parsed, dict):
                overall = self._safe_float(parsed.get("overall"))
                rationale = str(parsed.get("rationale") or "").strip()
                criteria = parsed.get("criteria") if isinstance(parsed.get("criteria"), dict) else {}
                breakdown = self._normalize_breakdown(criteria)

                score = overall or self._extract_rating(response) or 5.0
                if not rationale:
                    rationale = "Generated from LLM breakdown prompt."

                trace = {
                    "prompt": self._truncate_text(rendered_prompt, self.max_prompt_tokens_for_trace),
                    "response": str(response).strip(),
                    "input_excerpt": truncated_for_scoring,
                    "method": "llm_breakdown_json",
                    "llm_score": score,
                    "breakdown": breakdown,
                }
                return (score, rationale, trace) if return_trace else (score, rationale)
        except Exception as exc:  # pragma: no cover - best-effort enhancement
            breakdown_trace = {"breakdown_error": str(exc)}

        rendered_prompt = self._render_prompt(truncated_for_scoring)
        chain = self.prompt | self.llm | StrOutputParser()
        response = chain.invoke({"review": truncated_for_scoring})
        score = self._extract_rating(response) or 5.0

        trace = {
            "prompt": self._truncate_text(rendered_prompt, self.max_prompt_tokens_for_trace),
            "response": response.strip(),
            "input_excerpt": truncated_for_scoring,
            "method": "llm_prompt",
            "llm_score": score,
            "breakdown": {},
            **breakdown_trace,
        }

        return (score, trace["response"], trace) if return_trace else (score, trace["response"])

    def _extract_rating(self, text: str) -> Optional[float]:
        if not text:
            return None

        # Prefer explicit, labeled rating lines to avoid accidentally capturing
        # unrelated numbers (e.g., years, section numbers, dataset sizes).
        labeled_patterns = [
            # e.g., "Rating: 7.5" / "Rating - 7" / "4. Rating: 8/10"
            r"(?im)^\s*(?:\d+\.)?\s*(?:rating|overall\s*(?:score|rating)|score)\s*[:\-]\s*([0-9]+(?:\.[0-9]+)?)\s*(?:/\s*10)?\s*$",
            # e.g., "Rating (1-10): 8"
            r"(?im)^\s*(?:\d+\.)?\s*rating\s*\(\s*1\s*[-–]\s*10\s*\)\s*[:\-]\s*([0-9]+(?:\.[0-9]+)?)\s*$",
        ]

        for pattern in labeled_patterns:
            match = re.search(pattern, str(text))
            if not match:
                continue
            score = self._safe_float(match.group(1))
            if score is not None:
                return score

        # Inline labeled form (not necessarily isolated on its own line).
        inline_match = re.search(
            r"(?i)\b(?:rating|overall\s*(?:score|rating)|score)\b\s*[:\-]\s*([0-9]+(?:\.[0-9]+)?)\s*(?:/\s*10)?",
            str(text),
        )
        if inline_match:
            score = self._safe_float(inline_match.group(1))
            if score is not None:
                return score

        # As a last resort, accept X/10 only when the same line mentions rating/score/overall.
        for line in str(text).splitlines():
            if not re.search(r"\b(rating|overall|score)\b", line, re.IGNORECASE):
                continue
            match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*/\s*10", line)
            if not match:
                continue
            score = self._safe_float(match.group(1))
            if score is not None:
                return score

        return None

    def _strip_existing_rating_sections(self, review_text: str) -> str:
        """Remove self-declared rating blocks from draft reviews.

        Many reviewer drafts include a dedicated "Rating" section. If we pass that
        through to the rating model, it anchors the score and can collapse to a
        constant (e.g., always 8). We keep the rest of the review (strengths/
        weaknesses/detail) as evidence.
        """

        if not review_text:
            return ""

        lines = str(review_text).splitlines()
        out: list[str] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            # Drop lines like "4. Rating ..." or "Rating: 8" and also drop a
            # following line if it's just a bare number.
            if re.match(r"^\s*(?:\d+\.)?\s*rating\b", line, flags=re.IGNORECASE):
                i += 1
                if i < len(lines) and re.match(r"^\s*[0-9]+(?:\.[0-9]+)?\s*(?:/\s*10)?\s*$", lines[i]):
                    i += 1
                continue
            out.append(line)
            i += 1
        return "\n".join(out).strip()

    def _render_prompt_breakdown(self, review_text: str) -> str:
        messages = self.prompt_breakdown.format_messages(review=review_text, criteria_keys=", ".join(self.criteria_keys))
        parts = []
        for message in messages:
            role = getattr(message, "type", message.__class__.__name__).upper()
            parts.append(f"{role}: {getattr(message, 'content', '')}")
        return "\n\n".join(parts)

    def _extract_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        s = str(text).strip()
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

        # Best-effort: find the first {...} block.
        match = re.search(r"\{[\s\S]*\}", s)
        if not match:
            return None
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    def _safe_float(self, value: Any) -> Optional[float]:
        try:
            f = float(value)
        except Exception:
            return None
        if 0.0 < f <= 10.0:
            return f
        return None

    def _normalize_breakdown(self, criteria: Dict[str, Any]) -> Dict[str, float]:
        normalized: Dict[str, float] = {}
        for key in self.criteria_keys:
            if key not in criteria:
                continue
            f = self._safe_float(criteria.get(key))
            if f is None:
                continue
            normalized[key] = f
        # Also accept arbitrary extra numeric keys if provided.
        for key, value in criteria.items():
            if key in normalized:
                continue
            if not isinstance(key, str):
                continue
            f = self._safe_float(value)
            if f is None:
                continue
            normalized[key] = f
        return normalized

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        truncated = text[:max_chars]
        return f"{truncated}\n...[truncated]"

    def _render_prompt(self, review_text: str) -> str:
        messages = self.prompt.format_messages(review=review_text)
        parts = []
        for message in messages:
            role = getattr(message, "type", message.__class__.__name__).upper()
            parts.append(f"{role}: {getattr(message, 'content', '')}")
        return "\n\n".join(parts)


def rating_node(state: AgentState) -> AgentState:
    reviewer_output = state.get("final_review") or state.get("draft_review") or ""
    rating_agent = RatingAgent()
    rating_result = None
    try:
        rating_result = rating_agent.predict_rating(reviewer_output, return_trace=True)
    except TypeError:
        rating_result = rating_agent.predict_rating(reviewer_output)

    rating_trace = {}
    rating_breakdown = None
    if isinstance(rating_result, tuple) and len(rating_result) == 3:
        predicted_rating, rationale, rating_trace = rating_result
        if isinstance(rating_trace, dict) and isinstance(rating_trace.get("breakdown"), dict):
            rating_breakdown = rating_trace.get("breakdown")
    else:
        predicted_rating, rationale = rating_result  # type: ignore[misc]

    progress = list(state.get("progress_log", []))
    progress.append("Rating Agent: predicted final score.")

    traces = dict(state.get("agent_traces") or {})
    if isinstance(rating_trace, dict):
        traces["rating"] = rating_trace

    return AgentState(
        predicted_rating=predicted_rating,
        rating_rationale=rationale,
        rating_breakdown=rating_breakdown or state.get("rating_breakdown", {}),
        agent_traces=traces,
        progress_log=progress,
    )
