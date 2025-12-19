import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Dict, Optional

import autorootcwd
import click
import streamlit as st

try:
    import fitz  # type: ignore
except ImportError:
    fitz = None

PAPER_TITLE_SESSION_KEY = "paper_title_input"

# Ensure src is on the path when running via `streamlit run run_demo.py`
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from src.agents.coordinator import CoordinatorAgent  # noqa: E402
from src.utils.llm_server_manager import start_default_servers  # noqa: E402


def _parse_cli_flags():
    """Parse CLI flags once (supports streamlit run run_demo.py -- --skip-llm-start)."""
    if st.session_state.get("_cli_parsed"):
        return

    @click.command()
    @click.option("--skip-llm-start", is_flag=True, help="Do not auto-start LLM servers")
    def _cli(skip_llm_start: bool):
        st.session_state["cli_skip_llm_start"] = skip_llm_start

    try:
        _cli(standalone_mode=False)
    except SystemExit:
        # click might attempt sys.exit on parse errors; ignore inside streamlit
        pass
    st.session_state["_cli_parsed"] = True


def _save_uploaded_pdf(uploaded_file) -> Optional[str]:
    """Persist uploaded PDF to a temp file so ParserAgent can read it."""
    if not uploaded_file:
        return None
    suffix = Path(uploaded_file.name).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


def _extract_pdf_metadata_from_bytes(data: bytes) -> Dict[str, str]:
    """Extract metadata from raw PDF bytes when an upload happens."""
    if not data or fitz is None:
        return {}
    try:
        with fitz.open(stream=data, filetype="pdf") as doc:
            return {
                key.lower(): value
                for key, value in doc.metadata.items()
                if value
            }
    except Exception:
        return {}


def _extract_pdf_metadata(pdf_path: str) -> Dict[str, str]:
    """Capture metadata like author/title from an uploaded PDF."""
    if not pdf_path or fitz is None:
        return {}
    try:
        with fitz.open(pdf_path) as doc:
            return {
                key.lower(): value
                for key, value in doc.metadata.items()
                if value
            }
    except Exception:
        return {}


def _format_metadata_key(key: str) -> str:
    clean_key = key.replace("dc:", "")
    return clean_key.replace("_", " ").title()


def _render_pdf_metadata(metadata: Dict[str, str]) -> None:
    metadata_order = [
        "title",
        "author",
        "subject",
        "keywords",
        "producer",
        "creationdate",
        "moddate",
    ]
    shown_keys = set()
    for key in metadata_order:
        value = metadata.get(key)
        if not value:
            continue
        st.write(f"**{_format_metadata_key(key)}**: {value}")
        shown_keys.add(key)
    for key in sorted(metadata):
        if key in shown_keys:
            continue
        st.write(f"**{_format_metadata_key(key)}**: {metadata[key]}")


def _get_metadata_title(metadata: Dict[str, str]) -> Optional[str]:
    for candidate in ("title", "dc:title"):
        value = metadata.get(candidate)
        if value:
            return value
    return None


def _resolve_paper_title(input_title: str, pdf_metadata: Dict[str, str]) -> str:
    cleaned = input_title.strip()
    metadata_title = next(
        (pdf_metadata.get(candidate) for candidate in ("title", "dc:title") if pdf_metadata.get(candidate)),
        None,
    )
    if (not cleaned or cleaned.lower() == "untitled") and metadata_title:
        return metadata_title
    return cleaned or "Untitled"


def _ensure_llm_servers_started() -> bool:
    """Start vLLM servers (1B/8B) via Python once per session unless opted out.

    Also waits for readiness on ports to avoid connection-refused during first calls.
    """
    if os.environ.get("DEMO_SKIP_LLM_START") == "1" or st.session_state.get("cli_skip_llm_start"):
        return False
    if st.session_state.get("llm_started"):
        return True
    try:
        handles = start_default_servers(wait=True)
        st.session_state["llm_started"] = True
        st.session_state["llm_pids"] = {k: v.pid for k, v in handles.items()}
        st.toast("LLM servers ready (Python-managed)", icon="✅")
        return True
    except Exception as exc:  # pragma: no cover - runtime dependency
        traceback.print_exc()
        st.error(f"Failed to start LLM servers: {exc}")
        return False


def run_pipeline(paper_text: str, paper_pdf_path: Optional[str], paper_title: str):
    agent = CoordinatorAgent()
    return agent.run(paper_text=paper_text, paper_pdf_path=paper_pdf_path, paper_title=paper_title)


def run_pipeline_stream(paper_text: str, paper_pdf_path: Optional[str], paper_title: str):
    agent = CoordinatorAgent()
    return agent.run_stream(paper_text=paper_text, paper_pdf_path=paper_pdf_path, paper_title=paper_title)


def _render_progress_log_panel(progress: list[str]) -> None:
    """Render a compact log panel with newest entries first.

    Streamlit doesn't reliably support auto-scroll of markdown blocks.
    Showing newest-first approximates "always scrolled to bottom" while keeping
    the latest item visible and emphasized.
    """

    if not progress:
        st.caption("(no logs yet)")
        return

    tail = progress[-80:]
    newest_first = list(reversed(tail))
    lines = []
    for idx, entry in enumerate(newest_first):
        if idx == 0:
            lines.append(f"- **{entry}**")
        else:
            lines.append(f"- {entry}")
    st.markdown("\n".join(lines), unsafe_allow_html=False)


st.set_page_config(page_title="AI Paper Reviewer", layout="wide")
st.title("AI Paper Reviewer Demo (RAG + Agentic Workflow)")

with st.expander("Instructions", expanded=False):
    st.markdown(
        """
        - Upload a PDF **or** paste paper text. If both are provided, PDF takes priority.
        - ChromaDB collections should be built (ICLR/NeurIPS/ICML/TMLR) for retrieval.
        """
    )

# Parse CLI flags (e.g., streamlit run run_demo.py -- --skip-llm-start)
_parse_cli_flags()

# Try to start LLM servers once per session unless skipped
_ensure_llm_servers_started()

col_input, col_settings = st.columns([3, 1])

if PAPER_TITLE_SESSION_KEY not in st.session_state:
    st.session_state[PAPER_TITLE_SESSION_KEY] = "Untitled"

pdf_metadata: Dict[str, str] = {}
with col_input:
    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_pdf:
        file_bytes = uploaded_pdf.read()
        pdf_metadata = _extract_pdf_metadata_from_bytes(file_bytes)
        uploaded_pdf.seek(0)
        metadata_title = _get_metadata_title(pdf_metadata)
        current_title = st.session_state[PAPER_TITLE_SESSION_KEY]
        if metadata_title and (not current_title or current_title.strip().lower() == "untitled"):
            st.session_state[PAPER_TITLE_SESSION_KEY] = metadata_title
    paper_text_input = st.text_area("Or paste paper text", height=200)
    paper_title = st.text_input("Paper title", key=PAPER_TITLE_SESSION_KEY)
    if pdf_metadata:
        st.caption("Detected metadata from the uploaded PDF:")
        _render_pdf_metadata(pdf_metadata)

with col_settings:
    st.write("Run Options")
    st.caption(
        "Uses defaults from src/agents/llm.py (vLLM on localhost).\n"
        "Set DEMO_SKIP_LLM_START=1 to prevent auto-start."
    )

run_clicked = st.button("Run Review", type="primary")

if run_clicked:
    if not uploaded_pdf and not paper_text_input.strip():
        st.warning("Provide a PDF or paper text to continue.")
        st.stop()

    with st.status("Running agentic pipeline...", expanded=True) as status:
        pdf_path = _save_uploaded_pdf(uploaded_pdf)
        paper_text = paper_text_input.strip()
        metadata_for_title = pdf_metadata or (_extract_pdf_metadata(pdf_path) if pdf_path else {})
        resolved_title = _resolve_paper_title(paper_title, metadata_for_title)

        log_placeholder = st.empty()
        last_log_len = 0
        last_state = None

        try:
            status.write("Streaming progress...")
            state = None
            for snapshot in run_pipeline_stream(
                paper_text=paper_text,
                paper_pdf_path=pdf_path,
                paper_title=resolved_title,
            ):
                state = snapshot
                last_state = state
                progress = state.get("progress_log", []) if state else []
                if len(progress) != last_log_len:
                    last_log_len = len(progress)
                    # Keep the log compact and keep newest visible.
                    with log_placeholder.container():
                        _render_progress_log_panel(progress)
            status.update(label="Completed", state="complete")
        except Exception as exc:  # pragma: no cover - runtime errors surfaced to user
            traceback.print_exc()
            status.update(label="Failed", state="error")
            st.toast("Pipeline failed", icon="❌")
            st.error(f"Pipeline error: {exc}")
            with st.expander("Error details", expanded=False):
                st.code(traceback.format_exc())
            if last_state:
                st.subheader("Progress Log (last snapshot)")
                _render_progress_log_panel(last_state.get("progress_log", []))
            st.stop()

    if state:
        st.subheader("Rating")
        st.metric("Predicted Rating", f"{state.get('predicted_rating', 'N/A')}")
        st.caption(state.get("rating_rationale", ""))

        st.subheader("Review")
        st.markdown(state.get("final_review", ""), unsafe_allow_html=False)

        st.subheader("Progress Log")
        _render_progress_log_panel(state.get("progress_log", []))

        st.subheader("Retrieved Human Reviews (top-k)")
        retrieved = state.get("retrieved_reviews") or []
        retrieved_urls = state.get("retrieved_paper_urls") or []
        for idx, review in enumerate(retrieved):
            url = retrieved_urls[idx] if idx < len(retrieved_urls) else ""
            with st.expander(f"Review {idx+1}"):
                if url:
                    st.markdown(f"[OpenReview Link]({url})")
                st.write(review)

        st.subheader("arXiv References")
        arxiv_refs = state.get("arxiv_reference_texts") or []
        arxiv_results = state.get("arxiv_results") or []
        st.caption(f"Query: {state.get('arxiv_query', '')} | Cache hit: {state.get('arxiv_cache_hit', False)}")
        for idx, ref in enumerate(arxiv_refs):
            with st.expander(f"arXiv Match {idx+1}"):
                st.write(ref)
                if idx < len(arxiv_results):
                    title = arxiv_results[idx].get("title")
                    url = arxiv_results[idx].get("url")
                    if title:
                        st.write(f"Title: {title}")
                    if url:
                        st.write(f"URL: {url}")

        st.subheader("Parsed Sections (from ParserAgent)")
        parsed = state.get("parsed_sections", {})
        for key, val in parsed.items():
            with st.expander(key.title()):
                st.write(val)

        if state.get("retrieved_pdf_paths"):
            st.caption("Downloaded PDFs cached under outputs/pdfs/ for retrieved papers.")
