"""Run side-by-side evaluation for baseline (zero-RAG) vs agentic RAG pipeline.

Baseline: ReviewerAgent without retrieved reviews or arXiv context.
Agent: CoordinatorAgent full workflow (retrieval + arXiv + rating).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import autorootcwd
import numpy as np
from tqdm import tqdm

from src.agents.coordinator import CoordinatorAgent
from src.agents.llm import get_llm
from src.agents.rating import RatingAgent
from src.agents.reviewer import ReviewerAgent
from src.evaluation.dataset import extract_ground_truth, load_test_dataset
from src.evaluation.metrics import calculate_correlation, evaluate_hallucination, evaluate_weakness_recall
from src.evaluation.reporting import resolve_run_paths, write_report_bundle

# Ensure repo root is on path when executed as a script
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _avg_human_rating(ground_truth: Dict) -> Optional[float]:
    ratings = [r["rating"] for r in ground_truth["reviews"] if r.get("rating") is not None]
    return float(np.mean(ratings)) if ratings else None


def _collect_metric_avg(results: List[Dict], key: str) -> float:
    values = [r[key] for r in results if r.get(key) is not None]
    return float(np.mean(values)) if values else 0.0


def run_baseline_vs_agent(
    data_file: str,
    output_file: str,
    sample_size: int = 10,
    use_llm_eval: bool = True,
) -> None:
    print(f"Loading dataset from {data_file}...")
    dataset = load_test_dataset(data_file, sample_size=sample_size)
    print(f"Loaded {len(dataset)} samples.")

    coordinator = CoordinatorAgent()
    baseline_reviewer = ReviewerAgent()
    rating_agent = RatingAgent()

    eval_llm = get_llm() if use_llm_eval else None
    if use_llm_eval and not eval_llm:
        print("Warning: Could not initialize LLM for evaluation. LLM-based metrics will be skipped.")
        use_llm_eval = False

    results: List[Dict] = []

    # Correlation tracking
    baseline_pred: List[float] = []
    agent_pred: List[float] = []
    actual_mean: List[float] = []

    for paper in tqdm(dataset, desc="Baseline vs Agent"):
        ground_truth = extract_ground_truth(paper)
        human_ratings = [r.get("rating") for r in ground_truth["reviews"] if r.get("rating") is not None]
        avg_human = _avg_human_rating(ground_truth)

        paper_text = paper.get("abstract", "")
        paper_title = f"Paper {paper.get('forum_id', 'unknown')}"

        # Baseline: zero-RAG reviewer
        baseline_review = baseline_reviewer.generate_review(
            paper_text=paper_text,
            similar_reviews=[],
            arxiv_references=[],
            paper_title=paper_title,
        )
        baseline_rating, baseline_rationale = rating_agent.predict_rating(baseline_review)

        # Agentic pipeline
        agent_review = None
        agent_rating = None
        agent_rationale = None
        try:
            state = coordinator.run(paper_text=paper_text, paper_title=paper_title)
            agent_review = state.get("final_review") or state.get("draft_review")
            agent_rating = state.get("predicted_rating")
            agent_rationale = state.get("rating_rationale")
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Error running agentic pipeline for {paper.get('forum_id')}: {exc}")

        # Metrics using LLM (optional)
        human_reviews_text = [r["content"] for r in ground_truth["reviews"]]
        baseline_recall = None
        agent_recall = None
        baseline_hallu = None
        agent_hallu = None
        baseline_hallu_ratio = None
        agent_hallu_ratio = None
        baseline_claims_total = None
        baseline_claims_hallu = None
        agent_claims_total = None
        agent_claims_hallu = None

        if use_llm_eval:
            baseline_recall_details = evaluate_weakness_recall(
                baseline_review,
                human_reviews_text,
                llm=eval_llm,
            )
            baseline_recall = baseline_recall_details.get("recall")

            baseline_hallu_details = evaluate_hallucination(
                paper_text,
                baseline_review,
                llm=eval_llm,
            )
            baseline_hallu = baseline_hallu_details.get("hallucination_score")
            baseline_hallu_ratio = baseline_hallu_details.get("hallucination_ratio")
            baseline_claims_total = baseline_hallu_details.get("total_claims")
            baseline_claims_hallu = baseline_hallu_details.get("hallucinated_claims")

            if agent_review:
                agent_recall_details = evaluate_weakness_recall(
                    agent_review,
                    human_reviews_text,
                    llm=eval_llm,
                )
                agent_recall = agent_recall_details.get("recall")

                agent_hallu_details = evaluate_hallucination(
                    paper_text,
                    agent_review,
                    llm=eval_llm,
                )
                agent_hallu = agent_hallu_details.get("hallucination_score")
                agent_hallu_ratio = agent_hallu_details.get("hallucination_ratio")
                agent_claims_total = agent_hallu_details.get("total_claims")
                agent_claims_hallu = agent_hallu_details.get("hallucinated_claims")

        # Collect correlations
        if avg_human is not None:
            actual_mean.append(avg_human)
            baseline_pred.append(baseline_rating)
            if agent_rating is not None:
                agent_pred.append(agent_rating)
            else:
                agent_pred.append(np.nan)

        results.append(
            {
                "forum_id": paper.get("forum_id"),
                "actual_ratings": human_ratings,
                "avg_actual_rating": avg_human,
                "baseline": {
                    "review": baseline_review,
                    "predicted_rating": baseline_rating,
                    "rating_rationale": baseline_rationale,
                    "weakness_recall": baseline_recall,
                    "hallucination_score": baseline_hallu,
                    "hallucination_ratio": baseline_hallu_ratio,
                    "hallucination_total_claims": baseline_claims_total,
                    "hallucination_hallucinated_claims": baseline_claims_hallu,
                },
                "agent": {
                    "review": agent_review,
                    "predicted_rating": agent_rating,
                    "rating_rationale": agent_rationale,
                    "weakness_recall": agent_recall,
                    "hallucination_score": agent_hallu,
                    "hallucination_ratio": agent_hallu_ratio,
                    "hallucination_total_claims": agent_claims_total,
                    "hallucination_hallucinated_claims": agent_claims_hallu,
                },
            }
        )

    # Aggregate metrics
    baseline_corr = calculate_correlation(
        [p for p in baseline_pred if not np.isnan(p)],
        [a for b, a in zip(baseline_pred, actual_mean) if not np.isnan(b)],
    )
    agent_corr = calculate_correlation(
        [p for p in agent_pred if not np.isnan(p)],
        [a for p, a in zip(agent_pred, actual_mean) if not np.isnan(p)],
    )

    avg_baseline_recall = _collect_metric_avg([r["baseline"] for r in results], "weakness_recall")
    avg_agent_recall = _collect_metric_avg([r["agent"] for r in results], "weakness_recall")

    avg_baseline_hallu = _collect_metric_avg([r["baseline"] for r in results], "hallucination_score")
    avg_agent_hallu = _collect_metric_avg([r["agent"] for r in results], "hallucination_score")

    avg_baseline_hallu_ratio = _collect_metric_avg([r["baseline"] for r in results], "hallucination_ratio")
    avg_agent_hallu_ratio = _collect_metric_avg([r["agent"] for r in results], "hallucination_ratio")

    report = {
        "sample_size": len(dataset),
        "metrics": {
            "baseline": {
                "correlation": baseline_corr,
                "avg_weakness_recall": avg_baseline_recall,
                "avg_hallucination": avg_baseline_hallu,
                "avg_hallucination_ratio": avg_baseline_hallu_ratio,
            },
            "agent": {
                "correlation": agent_corr,
                "avg_weakness_recall": avg_agent_recall,
                "avg_hallucination": avg_agent_hallu,
                "avg_hallucination_ratio": avg_agent_hallu_ratio,
            },
        },
        "details": results,
    }

    paths = resolve_run_paths(output=output_file, run_name="baseline_vs_agent")
    summary_lines = [
        "# Baseline vs Agent",
        "",
        f"- sample_size: {len(dataset)}",
        "",
        "## Baseline",
        f"- pearson: {baseline_corr['pearson']:.4f}",
        f"- spearman: {baseline_corr['spearman']:.4f}",
        f"- avg_weakness_recall: {avg_baseline_recall:.4f}",
        f"- avg_hallucination_score: {avg_baseline_hallu:.4f}",
        f"- avg_hallucination_ratio: {avg_baseline_hallu_ratio:.4f}",
        "",
        "## Agent",
        f"- pearson: {agent_corr['pearson']:.4f}",
        f"- spearman: {agent_corr['spearman']:.4f}",
        f"- avg_weakness_recall: {avg_agent_recall:.4f}",
        f"- avg_hallucination_score: {avg_agent_hallu:.4f}",
        f"- avg_hallucination_ratio: {avg_agent_hallu_ratio:.4f}",
    ]

    details_fieldnames = [
        "forum_id",
        "avg_actual_rating",
        "baseline_predicted_rating",
        "agent_predicted_rating",
        "baseline_weakness_recall",
        "agent_weakness_recall",
        "baseline_hallucination_score",
        "agent_hallucination_score",
        "baseline_hallucination_ratio",
        "agent_hallucination_ratio",
        "baseline_hallucination_total_claims",
        "baseline_hallucination_hallucinated_claims",
        "agent_hallucination_total_claims",
        "agent_hallucination_hallucinated_claims",
    ]

    def _detail_rows() -> List[Dict]:
        rows: List[Dict] = []
        for entry in results:
            rows.append(
                {
                    "forum_id": entry.get("forum_id"),
                    "avg_actual_rating": entry.get("avg_actual_rating"),
                    "baseline_predicted_rating": (entry.get("baseline") or {}).get("predicted_rating"),
                    "agent_predicted_rating": (entry.get("agent") or {}).get("predicted_rating"),
                    "baseline_weakness_recall": (entry.get("baseline") or {}).get("weakness_recall"),
                    "agent_weakness_recall": (entry.get("agent") or {}).get("weakness_recall"),
                    "baseline_hallucination_score": (entry.get("baseline") or {}).get("hallucination_score"),
                    "agent_hallucination_score": (entry.get("agent") or {}).get("hallucination_score"),
                    "baseline_hallucination_ratio": (entry.get("baseline") or {}).get("hallucination_ratio"),
                    "agent_hallucination_ratio": (entry.get("agent") or {}).get("hallucination_ratio"),
                    "baseline_hallucination_total_claims": (entry.get("baseline") or {}).get("hallucination_total_claims"),
                    "baseline_hallucination_hallucinated_claims": (entry.get("baseline") or {}).get("hallucination_hallucinated_claims"),
                    "agent_hallucination_total_claims": (entry.get("agent") or {}).get("hallucination_total_claims"),
                    "agent_hallucination_hallucinated_claims": (entry.get("agent") or {}).get("hallucination_hallucinated_claims"),
                }
            )
        return rows

    write_report_bundle(
        paths=paths,
        report=report,
        summary_lines=summary_lines,
        details_rows=_detail_rows(),
        details_fieldnames=details_fieldnames,
    )

    print("\nEvaluation complete (Baseline vs Agent)")
    print(f"Baseline correlation: Pearson={baseline_corr['pearson']:.4f}, Spearman={baseline_corr['spearman']:.4f}")
    print(f"Agent correlation:    Pearson={agent_corr['pearson']:.4f}, Spearman={agent_corr['spearman']:.4f}")
    if use_llm_eval:
        print(f"Avg Weakness Recall - Baseline: {avg_baseline_recall:.4f} | Agent: {avg_agent_recall:.4f}")
        print(f"Avg Hallucination  - Baseline: {avg_baseline_hallu:.4f} | Agent: {avg_agent_hallu:.4f}")
        print(f"Avg Hallucination Ratio - Baseline: {avg_baseline_hallu_ratio:.4f} | Agent: {avg_agent_hallu_ratio:.4f}")
    print(f"Report saved to {paths.report_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate baseline vs agentic RAG pipeline.")
    parser.add_argument("--data", type=str, default="data/evaluation/test_set_2025.json", help="Path to test set JSON")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/results",
        help="Output directory (preferred) or a JSON file path",
    )
    parser.add_argument("--sample", type=int, default=5, help="Number of samples to evaluate")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM-based weakness/hallucination eval")

    args = parser.parse_args()

    if args.sample == 0:
        args.sample = None  # Evaluate all samples if 0 is specified

    run_baseline_vs_agent(
        data_file=args.data,
        output_file=args.output,
        sample_size=args.sample,
        use_llm_eval=not args.no_llm,
    )
