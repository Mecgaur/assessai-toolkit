"""
Prompt tuning for AssessAI.

Clusters a set of questions, extracts keywords from each cluster, and
suggests additions to a base prompt. Follows the thesis pipeline:
    spaCy preprocessing -> SBERT embeddings -> UMAP -> K-Means -> keywords

Author: Aditya Gaur
"""

import logging
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from .embeddings import Embedder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def cluster_questions(
    questions: Sequence[str],
    embedder: Optional[Embedder] = None,
    n_clusters: Union[int, str] = "auto",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    n_components: int = 5,
) -> Dict[str, object]:
    """
    Cluster questions by semantic similarity.

    The pipeline: embed -> UMAP to a low-dim space -> K-Means.

    n_clusters:
        "auto"  pick k by silhouette score across 2..8
        int     use this k directly

    Returns a dict with labels, centroids, and the reduced coordinates.
    """
    if len(questions) < 3:
        raise ValueError("Need at least 3 questions to cluster.")

    emb = embedder or Embedder()
    vecs = emb.encode(list(questions))

    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "umap-learn is not installed. Run: pip install umap-learn"
        ) from exc

    reducer = umap.UMAP(
        n_components=min(n_components, max(2, len(questions) - 2)),
        n_neighbors=min(n_neighbors, max(2, len(questions) - 1)),
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
    )
    reduced = reducer.fit_transform(vecs)

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    if n_clusters == "auto":
        max_k = min(8, len(questions) - 1)
        best_k, best_score = 2, -1.0
        for k in range(2, max_k + 1):
            labels = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(reduced)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(reduced, labels)
            if score > best_score:
                best_k, best_score = k, score
        n_clusters = best_k
        logger.info("Auto-selected k=%d (silhouette=%.3f)", best_k, best_score)

    kmeans = KMeans(n_clusters=int(n_clusters), random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(reduced)

    return {
        "labels": labels.tolist(),
        "n_clusters": int(n_clusters),
        "reduced": reduced,
        "centroids": kmeans.cluster_centers_,
        "questions": list(questions),
    }


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------

def extract_keywords(
    questions: Sequence[str],
    labels: Sequence[int],
    top_n: int = 5,
    stop_words: str = "english",
) -> Dict[int, List[str]]:
    """
    Return top_n keywords per cluster using TF-IDF.

    Cluster 0's keywords come from questions assigned label 0, and so on.
    A cluster with fewer than 2 questions is skipped (TF-IDF needs >=1 doc
    to compute a document frequency; we just take the highest-idf terms).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    questions = list(questions)
    labels = list(labels)

    # Build per-cluster document strings so TF-IDF sees clusters as documents.
    clusters: Dict[int, List[str]] = {}
    for q, lab in zip(questions, labels):
        clusters.setdefault(int(lab), []).append(q)

    keywords: Dict[int, List[str]] = {}
    for lab, docs in clusters.items():
        if len(docs) < 2:
            # Too few docs to rank well; fall back to most common non-stop words.
            words = " ".join(docs).lower().split()
            stop = _basic_stopwords()
            counts: Dict[str, int] = {}
            for w in words:
                w = "".join(c for c in w if c.isalpha())
                if len(w) > 3 and w not in stop:
                    counts[w] = counts.get(w, 0) + 1
            keywords[lab] = sorted(counts, key=counts.get, reverse=True)[:top_n]
            continue

        vec = TfidfVectorizer(
            stop_words=stop_words,
            ngram_range=(1, 2),
            max_features=500,
        )
        try:
            matrix = vec.fit_transform(docs)
        except ValueError:
            keywords[lab] = []
            continue

        scores = np.asarray(matrix.sum(axis=0)).ravel()
        terms = vec.get_feature_names_out()
        order = scores.argsort()[::-1]
        keywords[lab] = [terms[i] for i in order[:top_n] if scores[i] > 0]

    return keywords


def _basic_stopwords():
    return {
        "the", "and", "for", "with", "your", "have", "does", "are", "you",
        "how", "what", "which", "that", "this", "from", "into", "within",
        "about", "their", "there", "these", "those", "will", "would",
        "should", "could", "when", "where", "been", "being", "than",
        "them", "they", "also", "such", "into", "over", "under",
    }


# ---------------------------------------------------------------------------
# Prompt suggestion
# ---------------------------------------------------------------------------

def suggest_prompt(
    base_prompt: str,
    questions: Sequence[str],
    embedder: Optional[Embedder] = None,
    n_clusters: Union[int, str] = "auto",
    top_n_keywords: int = 5,
) -> Dict[str, object]:
    """
    Suggest additions to a base prompt from cluster keywords.

    Returns the refined prompt and the per-cluster breakdown for transparency.
    """
    clustered = cluster_questions(questions, embedder=embedder, n_clusters=n_clusters)
    labels = clustered["labels"]
    keywords = extract_keywords(list(questions), labels, top_n=top_n_keywords)

    additions = []
    for lab in sorted(keywords):
        kws = keywords[lab]
        if not kws:
            continue
        additions.append(f"terms related to {', '.join(kws[:top_n_keywords])}")

    if additions:
        refined = (
            base_prompt.rstrip(". ")
            + ". Include questions covering: "
            + "; ".join(additions)
            + "."
        )
    else:
        refined = base_prompt

    return {
        "base_prompt": base_prompt,
        "refined_prompt": refined,
        "keywords": keywords,
        "labels": labels,
        "n_clusters": clustered["n_clusters"],
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    samples = [
        "Does your organization have a predictive maintenance strategy?",
        "Is predictive maintenance used on critical equipment?",
        "Do you use real-time sensor data for predictive maintenance?",
        "Are maintenance schedules planned using historical data?",
        "How are maintenance tasks prioritized and scheduled?",
        "Is there a formal maintenance planning process?",
        "Do technicians receive digital skills training?",
        "Are workers trained on Industry 4.0 technologies?",
        "Is there a continuous learning programme for staff?",
        "Is maintenance data collected automatically?",
        "Are KPIs such as MTTR and OEE tracked centrally?",
        "Is maintenance documentation digitized and searchable?",
    ]

    result = suggest_prompt(
        base_prompt="Generate questions for Industry 4.0 maintenance readiness",
        questions=samples,
        n_clusters="auto",
        top_n_keywords=4,
    )

    print("--- clusters ---")
    print(f"k = {result['n_clusters']}")
    print("\n--- keywords per cluster ---")
    for lab, kws in result["keywords"].items():
        print(f"  cluster {lab}: {kws}")

    print("\n--- refined prompt ---")
    print(result["refined_prompt"])