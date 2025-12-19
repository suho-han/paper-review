from src.evaluation.reporting import resolve_run_paths, write_report_bundle
from src.evaluation.metrics import calculate_correlation, evaluate_hallucination, evaluate_weakness_recall
from src.evaluation.dataset import extract_ground_truth, load_test_dataset
from src.agents.llm import get_llm
from src.agents.coordinator import CoordinatorAgent
import argparse
import json
import os
import sys

import numpy as np
from tqdm import tqdm

# Ensure repo root is on path when executed as a script
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def run_evaluation(
    data_file: str,
    output_file: str,
    sample_size: int = 10,
    use_llm_eval: bool = True
):
    print(f"Loading dataset from {data_file}...")
    dataset = load_test_dataset(data_file, sample_size=sample_size)
    print(f"Loaded {len(dataset)} samples.")

    coordinator = CoordinatorAgent()

    # Initialize LLM for evaluation if needed
    eval_llm = None
    if use_llm_eval:
        print("Initializing LLM for evaluation...")
        eval_llm = get_llm()
        if not eval_llm:
            print("Warning: Could not initialize LLM. Skipping LLM-based metrics.")
            use_llm_eval = False

    results = []

    predicted_scores = []
    actual_scores_mean = []

    hallucination_ratios = []

    for i, paper_data in enumerate(tqdm(dataset, desc="Evaluating")):
        ground_truth = extract_ground_truth(paper_data)

        # Use abstract as paper text (limitation: no full text)
        paper_text = paper_data['abstract']
        paper_title = f"Paper {paper_data['forum_id']}"  # Title is not in the JSON currently?

        # Run Coordinator
        try:
            state = coordinator.run(
                paper_text=paper_text,
                paper_title=paper_title
            )

            generated_review = state.get("final_review") or state.get("draft_review")
            predicted_rating = state.get("predicted_rating")

            if generated_review is None:
                print(f"Warning: No review generated for {paper_data['forum_id']}")
                continue

            # Calculate metrics for this sample

            # 1. Rating Correlation (prepare data)
            # Ground truth rating is the average of human ratings
            human_ratings = [r['rating'] for r in ground_truth['reviews'] if r['rating'] is not None]
            avg_human_rating = np.mean(human_ratings) if human_ratings else None

            if predicted_rating is not None and avg_human_rating is not None:
                predicted_scores.append(predicted_rating)
                actual_scores_mean.append(avg_human_rating)

            # 2. Weakness Recall
            recall_result = {}
            if use_llm_eval:
                human_reviews_text = [r['content'] for r in ground_truth['reviews']]
                recall_result = evaluate_weakness_recall(
                    generated_review,
                    human_reviews_text,
                    llm=eval_llm
                )

            # 3. Hallucination Check
            hallucination_result = {}
            if use_llm_eval:
                hallucination_result = evaluate_hallucination(
                    paper_text,
                    generated_review,
                    llm=eval_llm
                )

            hallucination_ratio = hallucination_result.get("hallucination_ratio")
            if hallucination_ratio is not None:
                try:
                    hallucination_ratios.append(float(hallucination_ratio))
                except Exception:
                    pass

            # Store result
            result_entry = {
                "forum_id": paper_data['forum_id'],
                "predicted_rating": predicted_rating,
                "actual_ratings": human_ratings,
                "avg_actual_rating": avg_human_rating,
                "weakness_recall": recall_result.get("recall"),
                "weakness_eval_details": recall_result,
                "hallucination_score": hallucination_result.get("hallucination_score"),
                "hallucination_ratio": hallucination_ratio,
                "hallucination_total_claims": hallucination_result.get("total_claims"),
                "hallucination_hallucinated_claims": hallucination_result.get("hallucinated_claims"),
                "hallucination_details": hallucination_result,
                "generated_review_snippet": generated_review[:200] + "..." if generated_review else ""
            }
            results.append(result_entry)

        except Exception as e:
            print(f"Error processing {paper_data['forum_id']}: {e}")
            import traceback
            traceback.print_exc()

    # Calculate aggregate metrics
    correlation = calculate_correlation(predicted_scores, actual_scores_mean)

    avg_recall = 0.0
    valid_recall_count = 0
    for r in results:
        if r.get("weakness_recall") is not None:
            avg_recall += r["weakness_recall"]
            valid_recall_count += 1
    if valid_recall_count > 0:
        avg_recall /= valid_recall_count

    avg_hallucination_ratio = float(np.mean(hallucination_ratios)) if hallucination_ratios else 0.0

    final_report = {
        "sample_size": len(dataset),
        "successful_runs": len(results),
        "metrics": {
            "correlation": correlation,
            "avg_weakness_recall": avg_recall,
            "avg_hallucination_ratio": avg_hallucination_ratio,
        },
        "details": results
    }

    paths = resolve_run_paths(output=output_file, run_name="run_eval")
    summary_lines = [
        "# Evaluation Report",
        "",
        f"- sample_size: {len(dataset)}",
        f"- successful_runs: {len(results)}",
        f"- pearson: {correlation['pearson']:.4f}",
        f"- spearman: {correlation['spearman']:.4f}",
        f"- avg_weakness_recall: {avg_recall:.4f}",
        f"- avg_hallucination_ratio: {avg_hallucination_ratio:.4f}",
    ]
    details_fieldnames = [
        "forum_id",
        "predicted_rating",
        "avg_actual_rating",
        "weakness_recall",
        "hallucination_score",
        "hallucination_ratio",
        "hallucination_total_claims",
        "hallucination_hallucinated_claims",
    ]
    write_report_bundle(
        paths=paths,
        report=final_report,
        summary_lines=summary_lines,
        details_rows=results,
        details_fieldnames=details_fieldnames,
    )

    print("\nEvaluation Complete!")
    print(f"Correlation: Pearson={correlation['pearson']:.4f}, Spearman={correlation['spearman']:.4f}")
    print(f"Average Weakness Recall: {avg_recall:.4f}")
    print(f"Average Hallucination Ratio: {avg_hallucination_ratio:.4f}")
    print(f"Report saved to {paths.report_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation on paper reviews.")
    parser.add_argument("--data", type=str, default="data/evaluation/test_set_2025.json", help="Path to data file")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/results",
        help="Output directory (preferred) or a JSON file path",
    )
    parser.add_argument("--sample", type=int, default=5, help="Number of samples to evaluate")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM-based evaluation")

    args = parser.parse_args()

    run_evaluation(
        args.data,
        args.output,
        sample_size=args.sample,
        use_llm_eval=not args.no_llm
    )
