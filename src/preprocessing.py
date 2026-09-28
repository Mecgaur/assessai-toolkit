"""
Preprocessing for AssessAI question datasets.

Loads questions from CSV, does basic cleaning (unicode, numbering,
whitespace), and optionally applies the POS filtering described in my
M.Tech thesis.

Author: Aditya Gaur
"""

import logging
import re
import unicodedata
from pathlib import Path
from typing import List, Union

import pandas as pd

logger = logging.getLogger(__name__)

# POS tags removed in the thesis before embedding. Kept here so the
# behaviour is reproducible if a user opts in.
UNWANTED_POS = {"AUX", "PRON", "DET", "PUNCT"}

# Strips leading "1. ", "Q3: ", "- ", "• " etc. from question strings.
LEADING_NUMBERING = re.compile(
    r"^\s*(?:[\(\[]?\d+[\)\].:\-]\s*|[-*•]\s*|[Qq]\d+[.:]\s*)"
)


def load_spacy_model(model_name: str = "en_core_web_sm"):
    """Load a spaCy model. Raises a clear error if it is not installed."""
    try:
        import spacy
    except ImportError as exc:
        raise ImportError(
            "spaCy is not installed. Run: pip install spacy"
        ) from exc

    try:
        return spacy.load(model_name)
    except OSError as exc:
        raise OSError(
            f"spaCy model '{model_name}' not found. Run: "
            f"python -m spacy download {model_name}"
        ) from exc


def clean_question_text(text: str) -> str:
    """
    Basic cleaning for a single question.

    Normalises unicode, strips leading numbering, collapses whitespace.
    Does not remove any words.
    """
    if not isinstance(text, str):
        return ""

    # NFKC handles curly quotes, ligatures, and other unicode oddities
    text = unicodedata.normalize("NFKC", text)
    text = LEADING_NUMBERING.sub("", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def remove_unwanted_pos(text: str, nlp) -> str:
    """
    Remove AUX, PRON, DET, PUNCT tokens using spaCy.

    This is the preprocessing step from the thesis. It is disabled by
    default in preprocess_questions() because modern sentence embedding
    models are trained on natural sentences, and stripping grammar can
    actually hurt embedding quality. See the README for the discussion.

    NOTE: I plan to run an experiment comparing embeddings with and
    without POS removal. That will be added as a contribution later.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    doc = nlp(text)
    tokens = [tok.text for tok in doc if tok.pos_ not in UNWANTED_POS]
    return " ".join(tokens)


def load_questions(
    path: Union[str, Path],
    column: str = "Questions",
) -> List[str]:
    """
    Read questions from a CSV file.

    Returns a cleaned list of question strings. If the given column name
    is not found, tries a case-insensitive match before failing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(
            f"CSV file is empty or has no header: {path}"
        ) from exc

    if column not in df.columns:
        match = [c for c in df.columns if c.lower() == column.lower()]
        if match:
            column = match[0]
        else:
            raise KeyError(
                f"Column '{column}' not in {path}. "
                f"Columns present: {list(df.columns)}"
            )

    questions = df[column].dropna().astype(str).tolist()
    questions = [clean_question_text(q) for q in questions]
    questions = [q for q in questions if q]

    logger.info("Loaded %d questions from %s", len(questions), path)
    return questions


def load_questions_dataframe(
    path: Union[str, Path],
    column: str = "Questions",
) -> pd.DataFrame:
    """
    Same as load_questions, but returns the full DataFrame with a new
    'Clean_Question' column. Use this when you need the other columns
    (e.g. sub-category labels) to be preserved.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(
                f"CSV file is empty or has no header: {path}"
            ) from exc

    if column not in df.columns:
        match = [c for c in df.columns if c.lower() == column.lower()]
        if match:
            column = match[0]
        else:
            raise KeyError(
                f"Column '{column}' not in {path}. "
                f"Columns present: {list(df.columns)}"
            )

    df = df.copy()
    df["Clean_Question"] = df[column].apply(clean_question_text)
    df = df[df["Clean_Question"] != ""].reset_index(drop=True)

    logger.info("Loaded %d rows from %s", len(df), path)
    return df


def preprocess_questions(
    questions: List[str],
    apply_pos_filter: bool = False,
    spacy_model: str = "en_core_web_sm",
) -> List[str]:
    """
    Main entry point. Always cleans. Optionally applies POS filtering.

    Parameters
    ----------
    questions : list of str
    apply_pos_filter : bool, default False
        Set to True to reproduce the thesis preprocessing.
    spacy_model : str
        Only used if apply_pos_filter is True.
    """
    cleaned = [clean_question_text(q) for q in questions]
    cleaned = [q for q in cleaned if q]

    if not apply_pos_filter:
        logger.info("Preprocessed %d questions (cleaning only).", len(cleaned))
        return cleaned

    nlp = load_spacy_model(spacy_model)
    filtered = [remove_unwanted_pos(q, nlp) for q in cleaned]
    filtered = [q for q in filtered if q]

    logger.info("Preprocessed %d questions (with POS filter).", len(filtered))
    return filtered


if __name__ == "__main__":
    # Quick smoke test. Run: python src/preprocessing.py
    logging.basicConfig(level=logging.INFO)

    samples = [
        "1. Does your organization have a formalized maintenance strategy?",
        "   How is data used in your maintenance workflow?   ",
        "• Is predictive maintenance implemented?",
        "Q3: Do you use real-time sensor data?",
        "",
    ]

    print("--- cleaning only ---")
    for q in preprocess_questions(samples):
        print(" ", q)

    print("\n--- cleaning + POS filter (thesis version) ---")
    try:
        for q in preprocess_questions(samples, apply_pos_filter=True):
            print(" ", q)
    except (OSError, ImportError) as exc:
        print("  [skipped]", exc)