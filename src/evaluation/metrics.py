import re
from typing import Dict, List, Optional

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from scipy.stats import pearsonr, spearmanr

from src.agents.llm import get_llm


def calculate_correlation(
    predicted_scores: List[float],
    actual_scores: List[float]
) -> Dict[str, float]:
    """
    Calculate Pearson and Spearman correlation coefficients.
    """
    if len(predicted_scores) < 2 or len(actual_scores) < 2:
        return {"pearson": 0.0, "spearman": 0.0}

    pearson_corr, _ = pearsonr(predicted_scores, actual_scores)
    spearman_corr, _ = spearmanr(predicted_scores, actual_scores)

    return {
        "pearson": pearson_corr,
        "spearman": spearman_corr
    }


def extract_section(text: str, section_name: str) -> str:
    """
    Extract a specific section (e.g., 'Weaknesses') from the review text.
    Assumes format like 'Weaknesses: ...' or '## Weaknesses ...'
    """
    # Normalize text
    text_lower = text.lower()
    section_lower = section_name.lower()

    # Try to find the section start
    # Patterns: "weaknesses:", "## weaknesses", "**weaknesses**"
    pattern = f"(?:^|\\n)(?:##|\\*\\*)?\\s*{section_lower}[:\\s]*(.*?)(?:(?:\\n(?:##|\\*\\*)\\s*\\w+)|$)"

    match = re.search(pattern, text_lower, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def evaluate_weakness_recall(
    generated_review: str,
    human_reviews: List[str],
    llm=None
) -> Dict[str, float]:
    """
    Evaluate how many of the human-identified weaknesses were covered by the AI.
    Uses LLM to match semantic similarity.
    """
    if llm is None:
        llm = get_llm()
        if llm is None:
            print("Warning: No LLM available for evaluation.")
            return {"recall": 0.0}

    # Extract weaknesses from generated review
    gen_weakness = extract_section(generated_review, "Weaknesses")
    if not gen_weakness:
        # Fallback: treat the whole text as potential source if extraction fails?
        # Or just return 0. Let's try to use the whole text if extraction fails,
        # but it might be noisy. For now, strict extraction.
        pass

    # Extract weaknesses from human reviews and concatenate
    human_weaknesses = []
    for hr in human_reviews:
        hw = extract_section(hr, "Weaknesses")
        if hw:
            human_weaknesses.append(hw)

    if not human_weaknesses:
        return {"recall": 0.0}  # No ground truth weaknesses

    # Prompt for LLM to check coverage
    # We ask LLM: "Here is a list of ground truth weaknesses. Here is a generated weakness section.
    # For each ground truth point, is it covered by the generated section?"

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert reviewer evaluator. Your task is to determine if the AI-generated review covers the weaknesses mentioned by human reviewers."),
        ("human", """
Ground Truth Weaknesses (from Human Reviewers):
{human_weaknesses}

AI Generated Weaknesses:
{gen_weakness}

Task:
1. Identify distinct weakness points in the Ground Truth.
2. For each point, determine if it is mentioned or covered in the AI Generated Weaknesses (even if phrased differently).
3. Calculate the Recall score: (Number of covered points) / (Total number of distinct ground truth points).

Output ONLY a JSON object with the following format:
{{
    "total_points": <int>,
    "covered_points": <int>,
    "recall": <float>,
    "explanation": "<string>"
}}
""")
    ])

    chain = prompt | llm

    try:
        response = chain.invoke({
            "human_weaknesses": "\n---\n".join(human_weaknesses),
            "gen_weakness": gen_weakness
        })

        # Parse JSON from response content
        import json
        content = response.content
        # Clean up markdown code blocks if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        result = json.loads(content.strip())
        return result

    except Exception as e:
        print(f"Error in LLM evaluation: {e}")
        return {"recall": 0.0, "error": str(e)}


def evaluate_hallucination(
    paper_text: str,
    generated_review: str,
    llm=None
) -> Dict[str, float]:
    """
    Check if the generated review contains hallucinations (claims not supported by the paper text).
    """
    if llm is None:
        llm = get_llm()
        if llm is None:
            return {"hallucination_score": 0.0}  # Default to 0 if no LLM

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a fact-checking assistant. Your task is to verify if the claims in the review are supported by the paper text."),
        ("human", """
Paper Text (Abstract/Excerpt):
{paper_text}

Generated Review:
{generated_review}

Task:
1. Identify key claims in the Generated Review about the paper's method, results, or contribution.
2. Check if each claim is supported by the Paper Text.
3. If a claim contradicts the text or mentions specific details (numbers, baseline names) NOT present in the text, it is a hallucination.
   (Note: General knowledge or reasonable inferences are acceptable, but specific fabricated details are not.)

Calculate a Hallucination Score (0.0 to 1.0):
- 0.0: No hallucinations (all claims supported).
- 1.0: Pure hallucination (completely unrelated or fabricated).

Output ONLY a JSON object:
{{
    "hallucination_score": <float>,
    "explanation": "<string>"
}}
""")
    ])

    chain = prompt | llm

    try:
        response = chain.invoke({
            "paper_text": paper_text,
            "generated_review": generated_review
        })

        # Parse JSON
        import json
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        result = json.loads(content.strip())
        return result
    except Exception as e:
        print(f"Error in hallucination check: {e}")
        return {"hallucination_score": 0.0, "error": str(e)}
