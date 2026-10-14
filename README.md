# AssessAI Toolkit 

A Python toolkit for evaluating and improving the quality of
LLM-generated assessment questions. Built during M.Tech research at
IIT Indore, then extended into a small comparative study of specificity
methods. 

The toolkit exposes three metrics — **specificity**, **repetition**,
**coverage** — and a **prompt-tuning** loop that clusters questions,
extracts keywords, and suggests refinements. The study measures which
specificity method best correlates with human judgment on 60 hand-labeled
questions. 

The full project is still under works.

## Headline finding

Eight specificity methods were compared against 60 hand-labeled examples
on a 1–5 scale, using Spearman correlation:

| Method          | Spearman rho | p-value | Notes                             |
|-----------------|--------------|---------|-----------------------------------|
| **sbert+logreg**| **+0.342**   | 0.008   | The one method with a signal      |
| sbert+xgb       | +0.168       | 0.203   | Thesis method; no significant result |
| sbert+rf        | +0.145       | 0.272   |                                   |
| tfidf+xgb       | +0.109       | 0.409   |                                   |
| anchors         | +0.083       | 0.527   | Documented negative result        |
| rules           | +0.019       | 0.883   | Transparent but weak              |
| tfidf+rf        | +0.000       | 0.999   |                                   |
| tfidf+logreg    | −0.069       | 0.601   |                                   |

**The thesis assumed an XGBoost classifier trained on SBERT embeddings
would work best. On this dataset, logistic regression on the same
embeddings performs substantially better.** That is a modest but honest
finding.

## What is in this repository

- `src/evaluation_metrics.py` — specificity (rules, anchors, classifier),
  repetition, coverage
- `src/prompt_tuning.py` — UMAP + K-Means clustering, TF-IDF keyword
  extraction, prompt suggestion
- `src/pipeline.py` — closed-loop orchestrator (generate → evaluate →
  tune → regenerate)
- `notebooks/01_validation.ipynb` — usage demo plus the specificity
  validation study
- `notebooks/02_prompt_tuning_demo.ipynb` — clusters 100 maintenance
  questions and generates a refined prompt
- `notebooks/03_full_pipeline_demo.ipynb` — end-to-end pipeline with a
  mock LLM
- `docs/validation.png`, `docs/clustering.png` — plots produced by the
  notebooks
- `docs/validation_results.csv` — full results table

## Installation

```bash
git clone https://github.com/Mecgour/assessai-toolkit.git
cd assessai-toolkit
python -m venv venv
source venv/Scripts/activate   # Windows Git Bash
# or: source venv/bin/activate   # Linux / macOS
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Quickstart

Evaluate a set of questions

```bash
from src import evaluate

report = evaluate(
    questions=my_questions,
    sub_categories=["Identification", "Reporting", "Execution"],
)
print(report["specificity"]["mean"])
print(report["repetition"]["mean_similarity"])
print(report["coverage"]["score"])
```

Suggest a refined prompt:

```bash
from src import suggest_prompt

result = suggest_prompt(
    base_prompt="Generate questions for maintenance readiness",
    questions=my_questions,
)
print(result["refined_prompt"])
```
Run the full closed-loop pipeline (requires an LLM callable):

```bash
from src.pipeline import run

def my_llm(prompt, sub_categories):
    # call your LLM here; return a list of question strings
    ...

result = run(
    base_prompt="Generate questions for maintenance readiness",
    llm=my_llm,
    sub_categories=["Identification", "Reporting", "Execution"],
    max_iterations=3,
)
```

## The Three Metrics

### Specificity
How concrete and actionable a question is, on a 1–5 scale. Three
implementations:

- `rules` — regex heuristics based on the CLEAR framework. No training
data, no embeddings, instant. The default because it is transparent.
- `anchors` — embedding similarity to fixed specific and vague anchor
questions. Kept for the study; performed poorly on our data.
- `classifier` — trained sklearn classifier. Supports TF-IDF or SBERT
features, and logistic regression, random forest, or XGBoost models.
The thesis used SBERT+XGBoost.

The validation notebook tests all eight combinations.

### Repetition
Returns three numbers rather than one, because repetition is not a single
thing:

- `mean_similarity` — average cosine similarity across all pairs
- `duplicate_fraction` — fraction of pairs above a threshold (default 0.85)
- `max_similarity` — the most similar pair in the set

### Coverage
Returns two sub-scores plus a per-sub-category breakdown:

- `presence` — fraction of sub-categories with a question above threshold
- `balance` — evenness of the question distribution across covered
sub-categories (entropy-based)
- `score` — presence × balance

The per-sub-category breakdown tells the user which sub-categories
are missing.

## The validation study in one paragraph

60 hand-labeled questions were collected (30 general + 30 sub-category,
scored 1–4 using the CLEAR framework). Eight
specificity methods were compared against these labels using Spearman
correlation. The classifiers used 5-fold stratified cross-validation on
59 questions (one example with label 1 was excluded because it could not
form a stratified fold). Of the eight methods, only SBERT+Logistic
Regression produced a statistically significant correlation (ρ = 0.342,
p = 0.008). All other methods were indistinguishable from chance on this
sample. The full table, plot, and CSV are in `docs/`.

## Limitations

- **Small validation set**. 60 examples. Correlation estimates have wide
confidence intervals. This is a proof of concept, not a benchmark.

- **Single annotator**. The labels come from one person (the thesis
author). Inter-rater reliability is unknown.

- **No live LLM validation**. The pipeline's llm argument is pluggable,
but no real LLM was called during this study. Notebook 03 uses a mock.

- **The anchors method failed**. It conflates topic with specificity:
a vague question on the same topic as a specific anchor scores high.
Documented in the notebook as a negative result.

- **Rules method is domain-specific**. The regex patterns in
`evaluation_metrics.py` were tuned for maintenance-domain questions.
Users in other domains should override them.

- **TF-IDF keyword extraction on small sets is noisy**. On 100 questions
it produces meaningful themes; on 4 questions it produces awkward
bigrams like "kind maintenance". Documented in notebook 02.

- **English-only**. No multilingual support.

- **Composite quality is not monotonic**. Adding questions can lower
average specificity even as coverage rises. Users can override the
weighting via the `weights` argument.

## What this project is not

Not a replacement for human review. It is a filter: it tells you which
of your 100 LLM-generated questions are worth a closer look, and which
sub-categories are missing. It does not tell you whether a question is
appropriate for your specific audience.

## Citation

If you use this toolkit, please cite the M.Tech thesis (see
`CITATION.cff`).

## Acknowledgements

Developed during M.Tech research at the Department of Mechanical
Engineering, IIT Indore, under the supervision of Prof. Bhupesh Kumar
Lad.

## License

The code and documentation are currently released for research and non-commercial use only.

