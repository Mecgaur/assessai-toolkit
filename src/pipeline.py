"""
Closed-loop pipeline for AssessAI.

Wires together question generation, evaluation, and prompt tuning:

    generate -> evaluate -> if quality is low, tune prompt -> regenerate

Stops when composite quality crosses a threshold or max_iterations is hit.

Author: Aditya Gaur
"""

import logging
from typing import Callable, Dict, List, Optional, Sequence

from .evaluation_metrics import Evaluator
from .prompt_tuning import suggest_prompt

logger = logging.getLogger(__name__)

# An LLM function takes (prompt, sub_categories) and returns a list of
# question strings. It can call any real API or be a local mock.
LLMFunction = Callable[[str, Sequence[str]], List[str]]


def composite_quality(report: Dict, weights: Optional[Dict[str, float]] = None) -> float:
    """
    Collapse the three metrics into a scalar in [0, 1] for the loop.

    specificity: rescale from [1, 5] to [0, 1]
    repetition:  invert mean similarity (higher = less repetitive = better)
    coverage:    already in [0, 1]

    Default weights are equal. Users can override.
    """
    weights = weights or {"specificity": 1/3, "repetition": 1/3, "coverage": 1/3}

    spec = float(report["specificity"]["mean"])
    spec_norm = max(0.0, min(1.0, (spec - 1.0) / 4.0))

    rep = float(report["repetition"]["mean_similarity"])
    rep_norm = max(0.0, min(1.0, 1.0 - rep))

    cov = float(report.get("coverage", {}).get("score", 0.0))

    return (
        weights["specificity"] * spec_norm
        + weights["repetition"] * rep_norm
        + weights["coverage"] * cov
    )


def run(
    base_prompt: str,
    llm: LLMFunction,
    sub_categories: Optional[Sequence[str]] = None,
    max_iterations: int = 3,
    quality_threshold: float = 0.5,
    verbose: bool = True,
) -> Dict:
    """
    Run the closed-loop pipeline.

    Parameters
    ----------
    base_prompt : starting prompt sent to the LLM
    llm         : callable (prompt, sub_categories) -> List[str]
    sub_categories : list of sub-categories for coverage; may be None
    max_iterations : upper bound on loop rounds
    quality_threshold : composite quality above which we stop early
    verbose     : print per-iteration summary

    Returns
    -------
    dict with final_prompt, final_questions, history, quality_threshold.
    """
    evaluator = Evaluator()
    prompt = base_prompt
    history: List[Dict] = []
    final_questions: List[str] = []

    for it in range(1, max_iterations + 1):
        logger.info("Iteration %d", it)
        questions = llm(prompt, sub_categories or [])

        report = evaluator.evaluate(questions, sub_categories)
        quality = composite_quality(report)

        entry = {
            "iteration": it,
            "n_questions": len(questions),
            "specificity_mean": float(report["specificity"]["mean"]),
            "repetition_mean": float(report["repetition"]["mean_similarity"]),
            "coverage_score": float(report.get("coverage", {}).get("score", 0.0)),
            "quality": float(quality),
        }
        history.append(entry)
        final_questions = questions

        if verbose:
            print(
                f"  iter {it}: n={entry['n_questions']:3d}  "
                f"spec={entry['specificity_mean']:.2f}  "
                f"rep={entry['repetition_mean']:.2f}  "
                f"cov={entry['coverage_score']:.2f}  "
                f"quality={entry['quality']:.3f}"
            )

        if quality >= quality_threshold:
            logger.info("Quality threshold reached at iteration %d", it)
            break

        if it == max_iterations:
            break

        tuned = suggest_prompt(prompt, questions)
        prompt = tuned["refined_prompt"]

    return {
        "final_prompt": prompt,
        "final_questions": final_questions,
        "history": history,
        "quality_threshold": quality_threshold,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Simple mock LLM for the smoke test: always returns the same two
    # questions, one specific and one vague.
    def mock_llm(prompt, sub_categories):
        return [
            "How many hours did the CNC machine run last week?",
            "Is maintenance good?",
            "Does the CMMS log every failure with a timestamp and root cause?",
            "Is there a documented procedure for handling equipment anomalies?",
        ]

    subcats = ["Identification", "Reporting", "Execution"]

    result = run(
        base_prompt="Generate questions for Industry 4.0 maintenance readiness",
        llm=mock_llm,
        sub_categories=subcats,
        max_iterations=2,
        quality_threshold=0.9,  # force it to try both iterations
        verbose=True,
    )
    print()
    print("Final prompt:")
    print(result["final_prompt"])