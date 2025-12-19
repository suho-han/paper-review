"""Utilities for saving evaluation outputs in a consistent format."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class ResultPaths:
    run_dir: Path
    report_json: Path
    summary_md: Path
    details_csv: Path


def _utc_timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def resolve_run_paths(
    *,
    output: str,
    run_name: str,
    timestamp: Optional[str] = None,
) -> ResultPaths:
    """Resolve output destinations.

    Rules:
    - If output ends with `.json`, use it as report path, and create siblings `summary.md` + `details.csv`.
    - Otherwise treat output as a directory. A run directory is created under it: <output>/<run_name>/<timestamp>/
    """

    timestamp = timestamp or _utc_timestamp()
    output_path = Path(output)

    if output_path.suffix.lower() == ".json":
        report_json = output_path
        run_dir = report_json.parent
        summary_md = run_dir / "summary.md"
        details_csv = run_dir / "details.csv"
        return ResultPaths(run_dir=run_dir, report_json=report_json, summary_md=summary_md, details_csv=details_csv)

    run_dir = output_path / run_name / timestamp
    report_json = run_dir / "report.json"
    summary_md = run_dir / "summary.md"
    details_csv = run_dir / "details.csv"
    return ResultPaths(run_dir=run_dir, report_json=report_json, summary_md=summary_md, details_csv=details_csv)


def write_report_bundle(
    *,
    paths: ResultPaths,
    report: Dict[str, Any],
    summary_lines: List[str],
    details_rows: Iterable[Dict[str, Any]],
    details_fieldnames: List[str],
) -> None:
    paths.run_dir.mkdir(parents=True, exist_ok=True)

    with open(paths.report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with open(paths.summary_md, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines).rstrip() + "\n")

    with open(paths.details_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=details_fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in details_rows:
            writer.writerow(row)
