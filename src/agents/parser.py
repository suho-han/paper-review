"""Parser agent responsible for extracting structured text from PDFs or raw strings."""

from __future__ import annotations

import re
from typing import Dict, Optional

from .state import AgentState

try:  # Optional dependency for PDF parsing
    import fitz  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    fitz = None


class ParserAgent:
    def __init__(self) -> None:
        self._has_pdf_support = fitz is not None

    def parse(self, *, pdf_path: Optional[str], raw_text: Optional[str]) -> Dict[str, str]:
        if pdf_path and self._has_pdf_support:
            text = self._extract_pdf_text(pdf_path)
        else:
            text = raw_text or ""

        if not text:
            raise ValueError("ParserAgent requires either a valid pdf_path or raw_text input.")

        sections = self._split_sections(text)
        sections["full_text"] = text
        return sections

    def _extract_pdf_text(self, pdf_path: str) -> str:
        if not fitz:
            raise RuntimeError("PyMuPDF is not installed but was required for PDF parsing.")
        document = fitz.open(pdf_path)
        pages = [page.get_text("text") for page in document]
        document.close()
        return "\n".join(pages)

    def _split_sections(self, text: str) -> Dict[str, str]:
        patterns = {
            "abstract": r"(?i)(abstract)([\s\S]*?)(?=\n[A-Z][A-Za-z ]{3,}\n)",
            "methodology": r"(?i)(method|approach|architecture)([\s\S]*?)(?=\n[A-Z][A-Za-z ]{3,}\n)",
            "experiments": r"(?i)(experiment|evaluation|results)([\s\S]*?)(?=\n[A-Z][A-Za-z ]{3,}\n)",
        }
        sections: Dict[str, str] = {}
        for name, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                sections[name] = match.group(0).strip()
        return sections


def parser_node(state: AgentState) -> AgentState:
    parser = ParserAgent()
    sections = parser.parse(pdf_path=state.get("paper_pdf_path"), raw_text=state.get("paper_text"))

    progress = list(state.get("progress_log", []))
    progress.append("Parser Agent: extracted core sections.")

    return AgentState(
        paper_text=sections.get("full_text", state.get("paper_text", "")),
        parsed_sections={k: v for k, v in sections.items() if k != "full_text"},
        progress_log=progress,
    )
