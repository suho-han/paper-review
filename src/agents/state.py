"""Typed state shared across LangGraph nodes and the coordinator."""

from typing import Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    paper_title: Optional[str]
    paper_pdf_path: Optional[str]
    paper_text: str

    agent_traces: Dict[str, Dict[str, Any]]

    parsed_sections: Dict[str, str]

    retrieved_reviews: List[str]
    retrieved_metadatas: List[Dict[str, Any]]
    retrieved_abstracts: List[str]
    retrieved_paper_urls: List[str]
    retrieved_pdf_urls: List[str]
    retrieved_pdf_paths: List[str]

    arxiv_results: List[Dict[str, Any]]
    arxiv_reference_texts: List[str]
    arxiv_query: Optional[str]
    arxiv_cache_hit: Optional[bool]

    draft_review: Optional[str]
    final_review: Optional[str]

    predicted_rating: Optional[float]
    rating_rationale: Optional[str]

    # Optional per-criterion ratings (1-10) if available.
    rating_breakdown: Dict[str, float]

    progress_log: List[str]
