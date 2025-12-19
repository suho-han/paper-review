"""Parser agent responsible for extracting structured text from PDFs or raw strings."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional

from dotenv import dotenv_values, load_dotenv
from langchain_core.prompts import ChatPromptTemplate

from .llm import get_llm
from .state import AgentState

load_dotenv()

_ENV_CONFIG = {**dotenv_values(), **os.environ}


def _env_var(name: str, default: Optional[str] = None) -> Optional[str]:
    return _ENV_CONFIG.get(name, default)


def _env_bool(name: str, default: str = "0") -> bool:
    value = (_env_var(name, default) or "").lower()
    return value not in {"0", "false", "none", "off", "no", "n"}


def _env_int(name: str, default: str = "0") -> int:
    value = _env_var(name, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _env_float(name: str, default: str = "0.0") -> float:
    value = _env_var(name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


try:  # Optional dependency for PDF parsing (PyMuPDF)
    import fitz  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    fitz = None

try:  # Optional dependency for table extraction
    import camelot  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    camelot = None


class ParserAgent:
    """PDF parsing that first asks a local LLM to clean up raw extraction output."""

    def __init__(self) -> None:
        self._has_pdf_support = fitz is not None
        self._has_pdftotext = shutil.which("pdftotext") is not None

        self._use_local_llm = _env_bool("PARSER_USE_LOCAL_LLM", "1")
        self._local_llm_model = "meta-llama/Llama-3.2-1B"  # Explicitly set to 1B model
        self._local_llm_provider = _env_var("PARSER_LOCAL_LLM_PROVIDER", "vllm")
        self._local_llm_temperature = _env_float("PARSER_LOCAL_LLM_TEMPERATURE", "0.3")
        self._local_llm_max_tokens = _env_int("PARSER_LOCAL_LLM_MAX_TOKENS", "1024")
        self._local_llm_char_limit = _env_int("PARSER_LOCAL_LLM_CHAR_LIMIT", "3000")

    def parse(self, *, pdf_path: Optional[str], raw_text: Optional[str]) -> Dict[str, str]:
        text = raw_text or ""
        tables_md = ""

        if pdf_path:
            if (not text or len(text) < 200) and self._use_local_llm:
                print("ParserAgent: Using local LLM for PDF cleanup.")
                llm_text = self._extract_with_local_llm(pdf_path)
                if llm_text:
                    text = llm_text
            if (not text or len(text) < 200) and self._has_pdftotext:
                print("ParserAgent: pdftotext for PDF extraction.")
                text = self._extract_with_pdftotext(pdf_path)
            if (not text or len(text) < 200) and self._has_pdf_support:
                print("ParserAgent: Falling back to PyMuPDF for PDF extraction.")
                text = self._extract_with_pymupdf(pdf_path)
            if self._has_pdf_support and camelot:  # best-effort table capture
                print("ParserAgent: Using Camelot for table extraction.")
                tables_md = self._extract_tables_md(pdf_path)

        if not text:
            raise ValueError("ParserAgent requires either a valid pdf_path or raw_text input.")

        sections = self._split_sections(text)
        cleaned = self._clean_text(text)

        # Fallback: if abstract missing, take the first chunk before common intro markers
        if "abstract" not in sections:
            fallback_abs = self._fallback_abstract(cleaned)
            if fallback_abs:
                sections["abstract"] = fallback_abs
            elif cleaned:
                sections["abstract"] = cleaned[:1200].strip()

        sections["full_text"] = cleaned
        if tables_md:
            sections["tables_md"] = tables_md

        # Write a markdown snapshot if source PDF is provided
        if pdf_path:
            md_path = self._write_markdown(sections, pdf_path)
            sections["markdown_path"] = str(md_path)

        return sections

    def _extract_with_pdftotext(self, pdf_path: str) -> str:
        """Use pdftotext -layout for fast draft extraction."""
        try:
            result = subprocess.run(
                ["pdftotext", "-layout", pdf_path, "-"],
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout
        except Exception:
            return ""

    def _extract_with_pymupdf(self, pdf_path: str) -> str:
        if not fitz:
            return ""
        document = fitz.open(pdf_path)
        pages = [page.get_text("text") for page in document]
        document.close()
        return "\n".join(pages)

    def _extract_with_local_llm(self, pdf_path: str) -> str:
        raw_text = self._extract_with_pymupdf(pdf_path)
        if not raw_text:
            return ""

        llm = get_llm(
            model=self._local_llm_model,  # Ensure 1B model is used
            provider=self._local_llm_provider,
            temperature=self._local_llm_temperature,
            max_tokens=self._local_llm_max_tokens,
        )
        if not llm:
            return raw_text

        prompt = ChatPromptTemplate.from_template(
            """Clean the following raw text extracted from a PDF. Remove weird line breaks, headers/footers, and repeated numbering. Return plain text only, honoring the original paragraph order.

{pdf_text}
"""
        )

        truncated = self._truncate_for_local_llm(raw_text)
        try:
            response = (prompt | llm).invoke({"pdf_text": truncated})
            return response.strip() if response else raw_text
        except Exception:
            return raw_text

    def _truncate_for_local_llm(self, text: str) -> str:
        if self._local_llm_char_limit <= 0 or len(text) <= self._local_llm_char_limit:
            return text
        return f"{text[:self._local_llm_char_limit]}\n...[truncated]"

    def _extract_tables_md(self, pdf_path: str) -> str:
        if not camelot:
            return ""
        try:
            tables = camelot.read_pdf(pdf_path, pages="all")
        except Exception:
            return ""

        md_tables = []
        for table in tables:
            df = table.df
            if df.empty:
                continue
            # Build a simple Markdown table
            header = "| " + " | ".join(df.iloc[0]) + " |"
            separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
            rows = ["| " + " | ".join(row) + " |" for row in df.iloc[1:].values]
            md_tables.append("\n".join([header, separator, *rows]))
        return "\n\n".join(md_tables)

    def _clean_text(self, text: str) -> str:
        # De-hyphenate line breaks like "hy-
        # phen" -> "hyphen"
        text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
        # Normalize line breaks: collapse single newlines within paragraphs
        text = re.sub(r"\n(?!\n)", " ", text)
        # Normalize multiple blank lines
        text = re.sub(r"\n{2,}", "\n\n", text)
        return self._strip_page_artifacts(text.strip())

    def _split_sections(self, text: str) -> Dict[str, str]:
        section_aliases = {
            "abstract": "abstract",
            "methodology": "methodology",
            "methods": "methodology",
            "approach": "methodology",
            "architecture": "methodology",
            "experiments": "experiments",
            "evaluation": "experiments",
            "results": "experiments",
            "conclusion": "conclusion",
        }

        heading_pattern = re.compile(
            r"(?im)^(?:\d+[\.)]\s*)?(?P<heading>abstract|methodology|methods|approach|architecture|experiments|evaluation|results|conclusion)\s*$",
            re.MULTILINE,
        )

        matches = list(heading_pattern.finditer(text))
        sections: Dict[str, str] = {}
        for idx, match in enumerate(matches):
            heading = match.group("heading").lower()
            canonical = section_aliases.get(heading)
            if not canonical or canonical in sections:
                continue

            start = match.end()
            while start < len(text) and text[start] in ": \t":
                start += 1
            if start < len(text) and text[start] == "\n":
                start += 1
            while start < len(text) and text[start] in "\r\n":
                start += 1

            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            section_text = text[start:end].strip()
            if not section_text:
                continue

            sections[canonical] = self._ensure_min_paragraphs(section_text, min_paragraphs=1, max_generated=2)
        return sections

    def _ensure_min_paragraphs(
        self, text: str, min_paragraphs: int = 1, max_generated: int = 2
    ) -> str:
        paragraphs = self._split_into_paragraphs(text)
        if len(paragraphs) >= min_paragraphs:
            return "\n\n".join(paragraphs[:max_generated])

        sentences = self._split_into_sentences(text)
        if not sentences:
            return text.strip()

        desired = max(min_paragraphs, min(len(sentences), max_generated))
        base = len(sentences) // desired
        extras = len(sentences) % desired
        idx = 0
        rebuilt: list[str] = []
        for i in range(desired):
            count = base + (1 if i < extras else 0)
            chunk = sentences[idx: idx + max(1, count)]
            idx += len(chunk)
            paragraph = " ".join(chunk).strip()
            if paragraph:
                rebuilt.append(paragraph)

        if not rebuilt:
            return text.strip()
        return "\n\n".join(rebuilt)

    def _split_into_paragraphs(self, text: str) -> list[str]:
        return [chunk.strip() for chunk in re.split(r"\n{2,}", text.strip()) if chunk.strip()]

    def _split_into_sentences(self, text: str) -> list[str]:
        return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text.strip()) if sentence.strip()]

    def _fallback_abstract(self, text: str) -> str:
        """If no abstract detected, take the first chunk up to an intro-like heading."""
        prefix = text[:4000]
        splitter = re.search(r"\n\s*(introduction|1\s*\.\s|I\.)", prefix, flags=re.IGNORECASE)
        end = splitter.start() if splitter else min(len(prefix), 2000)
        abstract = prefix[:end].strip()
        return abstract if len(abstract) > 50 else ""

    def _write_markdown(self, sections: Dict[str, str], pdf_path: str) -> Path:
        lines = []
        pdf_name = Path(pdf_path).name
        lines.append(f"# Parsed Output: {pdf_name}")
        lines.append("")

        order = ["abstract", "methodology", "experiments", "tables_md", "full_text"]
        for key in order:
            if key not in sections:
                continue
            title = key.replace("_md", "").replace("_", " ").title()
            lines.append(f"## {title}")
            lines.append("")
            lines.append("```text")
            lines.append(sections[key].strip())
            lines.append("```")
            lines.append("")

        out_dir = Path("outputs/parsed")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{Path(pdf_name).stem}.md"
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return out_path

    def _strip_page_artifacts(self, text: str) -> str:
        """Remove obvious header/footer noise and line numbers if they look like margins."""
        # Drop repeated CVPR-style headers/footers and confidential copy notes.
        text = re.sub(r"CVPR\s+#?\d+\b.*?COPY\. DO NOT DISTRIBUTE\.\s*", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"Submission #\d+\. CONFIDENTIAL REVIEW COPY\. DO NOT DISTRIBUTE\.?", "\n", text, flags=re.IGNORECASE)
        # Drop standalone line-number columns like "  001" inside paragraphs.
        text = re.sub(r"\s{2,}\d{3}(?=\s)", " ", text)
        return text


def parser_node(state: AgentState) -> AgentState:
    parser = ParserAgent()
    sections = parser.parse(pdf_path=state.get("paper_pdf_path"), raw_text=state.get("paper_text"))

    progress = list(state.get("progress_log", []))
    progress.append("Parser Agent: extracted core sections.")

    traces = dict(state.get("agent_traces") or {})
    traces.setdefault("parser", {})
    traces["parser"].update({
        "input_type": "pdf" if state.get("paper_pdf_path") else "text",
        "sections_found": [k for k in sections.keys() if k not in {"full_text"}],
        "text_chars": len(sections.get("full_text") or ""),
    })

    parsed_sections = {k: v for k, v in sections.items() if k not in {"full_text", "tables_md"}}
    if "tables_md" in sections:
        parsed_sections["tables_md"] = sections["tables_md"]

    return AgentState(
        paper_text=sections.get("full_text", state.get("paper_text", "")),
        parsed_sections=parsed_sections,
        agent_traces=traces,
        progress_log=progress,
    )
