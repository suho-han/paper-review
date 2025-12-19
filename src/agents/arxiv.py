"""ArXiv retrieval agent fetching papers similar to the input manuscript."""

from __future__ import annotations

import json
import re
import textwrap
import time
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
        max_retries: int = 3,
        backoff_seconds: int = 10,
    ) -> None:
        self.max_results = max_results
        self.enable_cache = enable_cache
        self.cache_path = Path(cache_path)
        self.log_path = Path(log_path)
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
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

        # Remove exact duplicates (commonly caused by arXiv versioned IDs like v1/v2).
        papers = self._dedupe_papers(papers)

        if not cache_hit and self.enable_cache and papers:
            # Cache the de-duplicated results to avoid persisting duplicates.
            self._cache[cache_key] = {
                "papers": papers,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
            self._save_cache()

        snippets = [self._format_reference(paper) for paper in papers]
        self._log_query(query, len(papers), cache_hit)
        return papers, snippets, query, cache_hit

    def _dedupe_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Drop duplicate papers while preserving order.

        Primary key: normalized arXiv ID (version suffix removed).
        Fallback key: normalized title + first author.
        """

        if not papers:
            return []

        seen: set[str] = set()
        unique: List[Dict[str, Any]] = []

        for paper in papers:
            key = self._paper_dedupe_key(paper)
            if not key:
                # If we can't build a key, keep it (better recall than false drops).
                unique.append(paper)
                continue
            if key in seen:
                continue
            seen.add(key)
            unique.append(paper)

        return unique

    def _paper_dedupe_key(self, paper: Dict[str, Any]) -> Optional[str]:
        arxiv_id = self._extract_arxiv_id(paper)
        if arxiv_id:
            return f"arxiv:{arxiv_id}"

        title = str(paper.get("title") or "").strip()
        if not title:
            return None
        title_key = self._normalize_title(title)

        authors = paper.get("authors")
        first_author = ""
        if isinstance(authors, list) and authors:
            first_author = str(authors[0] or "").strip().lower()

        return f"title:{title_key}|a:{first_author}"

    def _normalize_title(self, title: str) -> str:
        return " ".join(title.lower().split())

    def _extract_arxiv_id(self, paper: Dict[str, Any]) -> Optional[str]:
        """Extract arXiv identifier from URL/PDF URL.

        Examples:
        - https://arxiv.org/abs/2501.12345v2 -> 2501.12345
        - http://arxiv.org/pdf/2501.12345v1.pdf -> 2501.12345
        - https://arxiv.org/abs/cs/9901012v1 -> cs/9901012
        """

        candidates: List[str] = []
        for key in ("url", "pdf_url"):
            value = paper.get(key)
            if isinstance(value, str) and value:
                candidates.append(value)

        for url in candidates:
            # abs
            m = re.search(r"arxiv\.org/abs/([^?#\s]+)", url, flags=re.IGNORECASE)
            if m:
                return self._normalize_arxiv_id(m.group(1))

            # pdf
            m = re.search(r"arxiv\.org/pdf/([^?#\s]+)", url, flags=re.IGNORECASE)
            if m:
                path = m.group(1)
                if path.lower().endswith(".pdf"):
                    path = path[:-4]
                return self._normalize_arxiv_id(path)

        return None

    def _normalize_arxiv_id(self, raw_id: str) -> Optional[str]:
        if not raw_id:
            return None
        cleaned = raw_id.strip().strip("/")
        # Drop version suffix (v1, v2, ...)
        cleaned = re.sub(r"v\d+$", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip().strip("/")
        return cleaned or None

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
        try:
            if arxiv:
                return self._search_with_sdk(query)
        except Exception as exc:  # pragma: no cover - fallback on API issues
            print(f"ArXiv SDK search failed, falling back to HTTP: {exc}")
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

        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, params=params, timeout=15)
            except Exception as exc:  # pragma: no cover - network issues
                if attempt == self.max_retries - 1:
                    print(f"ArXiv fallback request failed: {exc}")
                    return []
                time.sleep(self.backoff_seconds * (attempt + 1))
                continue

            if response.status_code == 429:
                wait = self.backoff_seconds * (attempt + 1)
                print(f"ArXiv 429 rate limit. Retrying in {wait}s (attempt {attempt+1}/{self.max_retries})")
                time.sleep(wait)
                continue

            if response.status_code != 200:
                if attempt == self.max_retries - 1:
                    print(f"ArXiv fallback request failed with status {response.status_code}.")
                    return []
                time.sleep(self.backoff_seconds * (attempt + 1))
                continue

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

    traces = dict(state.get("agent_traces") or {})
    traces.setdefault("arxiv", {})
    traces["arxiv"].update({
        "query": query,
        "cache_hit": cache_hit if query else None,
        "titles": [paper.get("title", "") for paper in (papers or [])],
    })

    return AgentState(
        arxiv_results=papers,
        arxiv_reference_texts=snippets,
        arxiv_query=query,
        arxiv_cache_hit=cache_hit if query else None,
        agent_traces=traces,
        progress_log=progress,
    )
