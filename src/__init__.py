"""
AssessAI: toolkit for evaluating LLM-generated assessment questions.

Public API:
    evaluate(questions, sub_categories=None) -> dict
    Evaluator
    specificity, repetition, coverage
    Embedder
    suggest_prompt, cluster_questions, extract_keywords
"""

from .embeddings import Embedder
from .evaluation_metrics import (
    Evaluator,
    SpecificityClassifier,
    coverage,
    evaluate,
    repetition,
    specificity,
)
from .prompt_tuning import (
    cluster_questions,
    extract_keywords,
    suggest_prompt,
)

__version__ = "0.1.0"

__all__ = [
    "Embedder",
    "Evaluator",
    "SpecificityClassifier",
    "coverage",
    "evaluate",
    "repetition",
    "specificity",
    "cluster_questions",
    "extract_keywords",
    "suggest_prompt",
]