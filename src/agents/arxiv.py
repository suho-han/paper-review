"""ArXiv retrieval agent fetching papers similar to the input manuscript."""

from __future__ import annotations

import json
import textwrap
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

try:  # Optional dependency
    import arxiv  # type: ignore
except ImportError:  # pragma: no cover
    arxiv = None

from .state import AgentState


class ArxivPaperRetriever:
    """Lightweight wrapper around the arXiv API for similarity search with caching."""

    def __init__(
        self,
        max_results: int = 3,
        cache_path: str | Path = "logs/arxiv_cache.json",
        log_path: str | Path = "logs/arxiv_queries.log",
        enable_cache: bool = True,
    ) -> None:
        self.max_results = max_results
        self.enable_cache = enable_cache
        self.cache_path = Path(cache_path)
        self.log_path = Path(log_path)
        self._cache: Dict[str, Dict[str, Any]] = self._load_cache()

    def retrieve_similar_papers(
        self,
        *,
        title: Optional[str],
        abstract: Optional[str],
        raw_text: str,
    ) -> Tuple[List[Dict[str, Any]], List[str], str, bool]:
        query = self._build_query(title, abstract, raw_text)
        if not query:
            return [], [], "", False

        cache_key = self._normalize_key(query)
        cache_hit = False
        if self.enable_cache and cache_key in self._cache:
            papers = self._cache[cache_key]["papers"]
            cache_hit = True
        else:
            papers = self._search(query)
            if self.enable_cache and papers:
                self._cache[cache_key] = {
                    "papers": papers,
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                }
                self._save_cache()

        snippets = [self._format_reference(paper) for paper in papers]
        self._log_query(query, len(papers), cache_hit)
        return papers, snippets, query, cache_hit

    def _build_query(self, title: Optional[str], abstract: Optional[str], raw_text: str) -> str:
        if title and title.strip():
            return self._clean_text(title)
        if abstract and abstract.strip():
            return self._clean_text(abstract)[:300]
        return self._clean_text(raw_text)[:300]

    def _clean_text(self, text: str) -> str:
        return " ".join(text.split())

    def _normalize_key(self, query: str) -> str:
        return query.strip().lower()

    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        if not self.enable_cache:
            return {}
        if not self.cache_path.exists():
            return {}
        try:
            with self.cache_path.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
                if isinstance(data, dict):
                    return data
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Warning: failed to load arXiv cache: {exc}")
        return {}

    def _save_cache(self) -> None:
        if not self.enable_cache:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("w", encoding="utf-8") as fp:
            json.dump(self._cache, fp, ensure_ascii=False, indent=2)

    def _log_query(self, query: str, result_count: int, cache_hit: bool) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as fp:
            fp.write(
                f"[{timestamp}] cache_hit={cache_hit} results={result_count} query=\"{query}\"\n"
            )

    def _search(self, query: str) -> List[Dict[str, Any]]:
        if arxiv:
            return self._search_with_sdk(query)
        return self._fallback_search(query)

    def _search_with_sdk(self, query: str) -> List[Dict[str, Any]]:
        assert arxiv is not None
        search = arxiv.Search(
            query=query,
            max_results=self.max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        client = arxiv.Client(page_size=self.max_results)
        results: List[Dict[str, Any]] = []
        for result in client.results(search):
            results.append({
                "title": result.title.strip(),
                "summary": result.summary.strip(),
                "authors": [author.name for author in result.authors],
                "published": result.published.strftime("%Y-%m-%d") if result.published else "",
                "url": result.entry_id,
                "pdf_url": result.pdf_url,
            })
        return results

    def _fallback_search(self, query: str) -> List[Dict[str, Any]]:
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": self.max_results,
            "sortBy": "relevance",
        }
        url = "https://export.arxiv.org/api/query"
        response = requests.get(url, params=params, timeout=15)
        if response.status_code != 200:
            print(f"ArXiv fallback request failed with status {response.status_code}.")
            return []

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(response.text)
        papers: List[Dict[str, Any]] = []
        for entry in root.findall("atom:entry", ns):
            title = entry.findtext("atom:title", default="", namespaces=ns).strip()
            summary = entry.findtext("atom:summary", default="", namespaces=ns).strip()
            link = ""
            pdf_link = ""
            for link_elem in entry.findall("atom:link", ns):
                href = link_elem.attrib.get("href", "")
                rel = link_elem.attrib.get("rel")
                link_type = link_elem.attrib.get("type")
                if rel == "alternate":
                    link = href
                elif link_type == "application/pdf":
                    pdf_link = href
            authors = [author.findtext("atom:name", default="", namespaces=ns).strip() for author in entry.findall("atom:author", ns)]
            published = entry.findtext("atom:published", default="", namespaces=ns)[:10]
            papers.append({
                "title": title,
                "summary": summary,
                "authors": [a for a in authors if a],
                "published": published,
                "url": link,
                "pdf_url": pdf_link,
            })
        return papers

    def _format_reference(self, paper: Dict[str, Any]) -> str:
        authors = ", ".join(paper.get("authors", [])[:3]) or "Unknown"
        summary = textwrap.shorten(paper.get("summary", ""), width=600, placeholder="...")
        return (
            f"Title: {paper.get('title', 'N/A')}\n"
            f"Authors: {authors}\n"
            f"Published: {paper.get('published', '')}\n"
            f"Summary: {summary}\n"
            f"URL: {paper.get('url', '')}\n"
        )


def arxiv_node(state: AgentState) -> AgentState:
    retriever = ArxivPaperRetriever()
    parsed_sections = state.get("parsed_sections") or {}
    papers, snippets, query, cache_hit = retriever.retrieve_similar_papers(
        title=state.get("paper_title"),
        abstract=parsed_sections.get("abstract"),
        raw_text=state.get("paper_text", ""),
    )

    progress = list(state.get("progress_log", []))
    if papers:
        hit_note = "(cache hit)" if cache_hit else "(fresh query)"
        progress.append(f"ArXiv Agent: fetched related literature {hit_note}.")
    else:
        progress.append("ArXiv Agent: no related papers found.")

    return AgentState(
        arxiv_results=papers,
        arxiv_reference_texts=snippets,
        arxiv_query=query,
        arxiv_cache_hit=cache_hit if query else None,
        progress_log=progress,
    )
