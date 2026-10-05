"""
Evaluation metrics for AssessAI.

Three metrics: specificity (1-5), repetition (0-1), coverage (0-1).
Specificity supports three implementations so we can compare them:
    rules       - regex heuristics based on the CLEAR framework
    anchors     - embedding similarity to fixed specific/vague anchors
    classifier  - trained sklearn classifier on TF-IDF or SBERT features

The classifier path is the one from the thesis. The notebook in
notebooks/01_validation.ipynb compares all three against hand labels.

Author: Aditya Gaur
"""

import logging
import re
from typing import Dict, Optional, Sequence

import numpy as np

from .embeddings import Embedder, cosine_similarity_matrix, cross_similarity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Specificity: rules
# ---------------------------------------------------------------------------
# Patterns chosen to match the CLEAR framework. Kept short on purpose so
# they are easy to read and easy to override per domain.

_NUMBER_RE = re.compile(r"\b\d+\b")
_TIMEFRAME_RE = re.compile(
    r"\b(per|every|each|weekly|monthly|daily|hourly|annually|yearly|quarterly)\b"
)
_UNIT_RE = re.compile(
    r"\b(percent|%|hours?|days?|weeks?|months?|years?|minutes?|seconds?)\b"
)
_MEASURE_VERB_RE = re.compile(
    r"\b(measured|recorded|logged|documented|tracked|monitored|"
    r"scheduled|inspected|audited|reported|analysed|analyzed)\b"
)
_NAMED_TOOL_RE = re.compile(
    r"\b(ai|ml|iot|iiot|cmms|erp|mes|scada|api|sensor|dashboard|"
    r"analytics|algorithm|kpi|mttr|mtbf|oee|digital[\s-]?twin|"
    r"predictive|real[\s-]?time)\b"
)
_VAGUE_QUANTIFIER_RE = re.compile(
    r"\b(some|many|few|several|often|sometimes|occasionally|various|lots?)\b"
)
_SUBJECTIVE_RE = re.compile(
    r"\b(good|bad|effective|efficient|useful|better|worse|"
    r"appropriate|adequate|proper|mature)\b"
)
_PLACEHOLDER_RE = re.compile(r"\b(things?|stuff|aspects?|areas?|systems?)\b")


def _specificity_rules(questions: Sequence[str]) -> np.ndarray:
    """
    Score questions 1-5 from regex signals.

    Start at 3. Positive signals add 0.5 each, negative signals subtract.
    Clip to [1, 5].
    """
    scores = []
    for q in questions:
        t = q.lower()
        n_words = len(t.split())

        pos = 0
        if _NUMBER_RE.search(t):
            pos += 1
        if _TIMEFRAME_RE.search(t):
            pos += 1
        if _UNIT_RE.search(t):
            pos += 1
        if _MEASURE_VERB_RE.search(t):
            pos += 1
        if _NAMED_TOOL_RE.search(t):
            pos += 1
        if n_words >= 10:
            pos += 1

        neg = 0.0
        if _VAGUE_QUANTIFIER_RE.search(t):
            neg += 1
        if _SUBJECTIVE_RE.search(t):
            neg += 1.5
        if _PLACEHOLDER_RE.search(t):
            neg += 1
        if n_words < 6:
            neg += 1

        raw = 3.0 + 0.5 * pos - 0.5 * neg
        scores.append(max(1.0, min(5.0, raw)))

    return np.array(scores, dtype=float)


# ---------------------------------------------------------------------------
# Specificity: anchors
# ---------------------------------------------------------------------------
# Kept for the validation study only. On our data this method scored a vague
# question higher than a moderate one, because embeddings encode topic not
# specificity. Reported in the notebook as a negative result.

DEFAULT_SPECIFIC_ANCHORS = [
    "How many hours did the machine run last week?",
    "What percentage of failures were logged with a timestamp?",
    "Is there a documented procedure for handling equipment anomalies?",
    "How often per month are assets inspected against the schedule?",
    "Does the sensor report values every second?",
    "What is the average time between failures for the critical pump?",
    "Is a root cause recorded for every unplanned downtime event?",
]

DEFAULT_VAGUE_ANCHORS = [
    "Is the system good?",
    "Is the process effective?",
    "Is the team efficient?",
    "Is the strategy aligned?",
    "Is the data useful?",
    "Is the approach working?",
    "Are things going well?",
]

DEFAULT_STRETCH = 4.0


def _specificity_anchors(
    questions: Sequence[str],
    embedder: Embedder,
    specific_anchors: Sequence[str],
    vague_anchors: Sequence[str],
    stretch: float,
) -> np.ndarray:
    q_vecs = embedder.encode(list(questions))
    s_vecs = embedder.encode(list(specific_anchors))
    v_vecs = embedder.encode(list(vague_anchors))

    sim_s = cross_similarity(q_vecs, s_vecs).max(axis=1)
    sim_v = cross_similarity(q_vecs, v_vecs).max(axis=1)

    diff = np.clip((sim_s - sim_v) * stretch, -1.0, 1.0)
    return 3.0 + 2.0 * diff


# ---------------------------------------------------------------------------
# Specificity: classifier
# ---------------------------------------------------------------------------

class SpecificityClassifier:
    """
    Trained specificity scorer on a 1-5 scale.

    features: "tfidf" or "sbert"
    model:    "logreg", "rf", or "xgb"

    XGBoost needs 0-indexed labels, so we map internally and invert on
    prediction. Users keep calling 1-5.
    """

    def __init__(
        self,
        features: str = "sbert",
        model: str = "xgb",
        embedder: Optional[Embedder] = None,
        random_state: int = 42,
    ):
        if features not in ("tfidf", "sbert"):
            raise ValueError(f"features must be 'tfidf' or 'sbert', got {features!r}")
        if model not in ("logreg", "rf", "xgb"):
            raise ValueError(f"model must be 'logreg', 'rf', or 'xgb', got {model!r}")

        self.features = features
        self.model_name = model
        self.embedder = embedder or Embedder()
        self.random_state = random_state

        self._vectorizer = None
        self._clf = None
        self._label_map: Dict[int, int] = {}
        self._inverse_map: Dict[int, int] = {}

    def _fit_features(self, questions):
        if self.features == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words="english",
            )
            return self._vectorizer.fit_transform(questions)
        return self.embedder.encode(list(questions))

    def _transform(self, questions):
        if self.features == "tfidf":
            return self._vectorizer.transform(questions)
        return self.embedder.encode(list(questions))

    def fit(self, questions: Sequence[str], scores: Sequence[int]) -> "SpecificityClassifier":
        X = self._fit_features(list(questions))
        y_raw = np.array(scores, dtype=int)

        unique = sorted(set(y_raw.tolist()))
        self._label_map = {label: i for i, label in enumerate(unique)}
        self._inverse_map = {i: label for label, i in self._label_map.items()}
        y = np.array([self._label_map[s] for s in y_raw])

        if self.model_name == "logreg":
            from sklearn.linear_model import LogisticRegression
            self._clf = LogisticRegression(max_iter=1000, random_state=self.random_state)
        elif self.model_name == "rf":
            from sklearn.ensemble import RandomForestClassifier
            self._clf = RandomForestClassifier(
                n_estimators=200, random_state=self.random_state
            )
        else:
            try:
                from xgboost import XGBClassifier
            except ImportError as exc:
                raise ImportError(
                    "xgboost is not installed. Run: pip install xgboost"
                ) from exc
            self._clf = XGBClassifier(
                n_estimators=200,
                eval_metric="mlogloss",
                random_state=self.random_state,
                verbosity=0,
            )

        self._clf.fit(X, y)
        return self

    def predict(self, questions: Sequence[str]) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("Classifier not fitted. Call fit() first.")
        X = self._transform(list(questions))
        y = self._clf.predict(X)
        return np.array([self._inverse_map[int(v)] for v in y], dtype=float)


# ---------------------------------------------------------------------------
# Specificity: dispatcher
# ---------------------------------------------------------------------------

def specificity(
    questions: Sequence[str],
    method: str = "rules",
    embedder: Optional[Embedder] = None,
    classifier: Optional[SpecificityClassifier] = None,
    specific_anchors: Optional[Sequence[str]] = None,
    vague_anchors: Optional[Sequence[str]] = None,
    stretch: float = DEFAULT_STRETCH,
) -> Dict[str, object]:
    """
    Score each question on specificity from 1 (vague) to 5 (specific).

    method:
        "rules"      regex heuristics, no training needed (default)
        "anchors"    embedding similarity to fixed anchors
        "classifier" needs a fitted SpecificityClassifier passed in
    """
    if not questions:
        return {"scores": np.array([]), "mean": 0.0, "min": 0.0, "max": 0.0, "method": method}

    if method == "rules":
        scores = _specificity_rules(questions)
    elif method == "anchors":
        emb = embedder or Embedder()
        scores = _specificity_anchors(
            questions,
            emb,
            specific_anchors or DEFAULT_SPECIFIC_ANCHORS,
            vague_anchors or DEFAULT_VAGUE_ANCHORS,
            stretch,
        )
    elif method == "classifier":
        if classifier is None:
            raise ValueError("method='classifier' requires a fitted SpecificityClassifier")
        scores = classifier.predict(questions)
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'rules', 'anchors', or 'classifier'.")

    return {
        "scores": scores,
        "mean": float(scores.mean()),
        "min": float(scores.min()),
        "max": float(scores.max()),
        "method": method,
    }


# ---------------------------------------------------------------------------
# Repetition
# ---------------------------------------------------------------------------

def repetition(
    questions: Sequence[str],
    embedder: Optional[Embedder] = None,
    duplicate_threshold: float = 0.85,
) -> Dict[str, float]:
    """
    Report several redundancy indicators rather than one number.

    mean_similarity      average off-diagonal cosine similarity
    duplicate_fraction   fraction of pairs above threshold
    max_similarity       the most similar pair in the set
    """
    if len(questions) < 2:
        return {
            "mean_similarity": 0.0,
            "duplicate_fraction": 0.0,
            "max_similarity": 0.0,
            "duplicate_threshold": float(duplicate_threshold),
        }

    emb = embedder or Embedder()
    vecs = emb.encode(list(questions))
    sim = cosine_similarity_matrix(vecs)

    n = sim.shape[0]
    off_diag = sim[~np.eye(n, dtype=bool)]

    return {
        "mean_similarity": float(off_diag.mean()),
        "duplicate_fraction": float((off_diag >= duplicate_threshold).mean()),
        "max_similarity": float(off_diag.max()),
        "duplicate_threshold": float(duplicate_threshold),
    }


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def coverage(
    questions: Sequence[str],
    sub_categories: Sequence[str],
    embedder: Optional[Embedder] = None,
    match_threshold: float = 0.35,
) -> Dict[str, object]:
    """
    How well the question set covers the given sub-categories.

    Two sub-scores:
        presence - fraction of sub-categories with a question above threshold
        balance  - evenness of questions across covered sub-categories

    score = presence * balance.

    Threshold default of 0.35 is tuned for MiniLM. Other embedding models
    will need different values; pass match_threshold to override.
    """
    if not questions or not sub_categories:
        return {
            "score": 0.0,
            "presence": 0.0,
            "balance": 0.0,
            "sub_categories": [],
            "match_threshold": float(match_threshold),
        }

    emb = embedder or Embedder()
    q_vecs = emb.encode(list(questions))
    sc_vecs = emb.encode(list(sub_categories))
    sim = cross_similarity(q_vecs, sc_vecs)  # (n_q, n_sc)

    breakdown = []
    for j, sc in enumerate(sub_categories):
        best_q_idx = int(sim[:, j].argmax())
        best_score = float(sim[best_q_idx, j])
        breakdown.append({
            "name": sc,
            "best_match": list(questions)[best_q_idx],
            "similarity": best_score,
            "covered": best_score >= match_threshold,
        })

    presence = sum(1 for b in breakdown if b["covered"]) / len(sub_categories)

    best_sc_idx = sim.argmax(axis=1)
    best_sc_score = sim.max(axis=1)
    counts = [0] * len(sub_categories)
    for i in range(len(questions)):
        if best_sc_score[i] >= match_threshold:
            counts[best_sc_idx[i]] += 1

    covered_idx = [j for j, b in enumerate(breakdown) if b["covered"]]
    if len(covered_idx) <= 1:
        balance = 1.0
    else:
        counts_covered = [counts[j] for j in covered_idx]
        total = sum(counts_covered)
        if total == 0:
            balance = 0.0
        else:
            p = np.array(counts_covered, dtype=float) / total
            entropy = -(p * np.log(p + 1e-12)).sum()
            balance = float(entropy / np.log(len(covered_idx)))

    return {
        "score": float(presence * balance),
        "presence": float(presence),
        "balance": float(balance),
        "sub_categories": breakdown,
        "match_threshold": float(match_threshold),
    }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Holds one Embedder so the same question isn't re-embedded for each metric.
    """

    def __init__(self, embedder: Optional[Embedder] = None):
        self.embedder = embedder or Embedder()

    def specificity(self, questions, **kwargs):
        return specificity(questions, embedder=self.embedder, **kwargs)

    def repetition(self, questions, **kwargs):
        return repetition(questions, embedder=self.embedder, **kwargs)

    def coverage(self, questions, sub_categories, **kwargs):
        return coverage(questions, sub_categories, embedder=self.embedder, **kwargs)

    def evaluate(self, questions, sub_categories=None, **kwargs):
        report = {
            "specificity": self.specificity(questions, **kwargs),
            "repetition": self.repetition(questions, **kwargs),
        }
        if sub_categories:
            report["coverage"] = self.coverage(questions, sub_categories, **kwargs)
        return report


def evaluate(questions, sub_categories=None, **kwargs):
    """One-shot evaluation. Builds a temporary Evaluator internally."""
    return Evaluator().evaluate(questions, sub_categories, **kwargs)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    samples = [
        "Does your maintenance department use AI-based anomaly detection on critical rotating equipment?",
        "Is maintenance good?",
        "Do you use data analytics in maintenance operations?",
        "Does your maintenance team use AI-based anomaly detection on critical rotating equipment?",
    ]
    subcats = [
        "Identification", "Reporting", "Execution",
        "Resource Management", "Planning", "Documentation",
    ]

    ev = Evaluator()

    print("--- specificity (rules) ---")
    for q, s in zip(samples, ev.specificity(samples, method="rules")["scores"]):
        print(f"  {s:.2f}  {q[:60]}")

    print("\n--- specificity (anchors) ---")
    for q, s in zip(samples, ev.specificity(samples, method="anchors")["scores"]):
        print(f"  {s:.2f}  {q[:60]}")

    print("\n--- repetition ---")
    for k, v in ev.repetition(samples).items():
        print(f"  {k}: {v}")

    print("\n--- coverage ---")
    cov = ev.coverage(samples, subcats)
    print(f"  score={cov['score']:.2f}  presence={cov['presence']:.2f}  balance={cov['balance']:.2f}")