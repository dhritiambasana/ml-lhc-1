# Duplicate Question Retrieval Model

## Overview

This folder contains the first machine-learning baseline for retrieving likely duplicate Stack Overflow questions. Given a question, the system ranks existing questions by text similarity.

The current baseline uses **TF-IDF with word unigrams and bigrams**, followed by cosine similarity. The evaluation script measures whether the known master question for a duplicate link appears among the top-ranked results.

This is a retrieval baseline, not a classifier that predicts whether Stack Overflow will eventually close a question as a duplicate.

## Files

- `prepare-data.py`: reads the raw question and post-link CSV files, cleans question text, and creates prepared question and duplicate-pair datasets.
- `tfidf-baseline.py`: builds a TF-IDF index and retrieves the most similar questions for a supplied query.
- `evaluate.py`: evaluates retrieval against known duplicate links using Recall@K. It defaults to a sample of 200 pairs to keep runtime and memory use manageable.

## Input datasets

The scripts expect the following files relative to the repository root:

- `data/dataset/posts_long.csv`
- `data/dataset/postLinks.csv`

`posts_long.csv` must contain `id`, `post_type_id`, `title`, and `body`. Only questions (`post_type_id == 1`) are retained. The text field is made from the cleaned title and body.

`postLinks.csv` must contain `post_id`, `related_post_id`, and `link_type_id`. Rows with `link_type_id == 3` are treated as duplicate links. Only pairs for which both question IDs are present in the prepared question collection are retained.

## Setup

From the repository root, install the dependencies:

```bash
pip install pandas scikit-learn joblib numpy
```

## Run order

Run all commands from the repository root:

```bash
python model/prepare-data.py
python model/tfidf-baseline.py --rebuild
python model/tfidf-baseline.py --query "How do I sort a list in Python?" --top-k 10
python model/evaluate.py
```

The baseline index is saved under `model/results/`. The prepared datasets are saved under `model/data/`.

### Evaluation options

Evaluate a larger sample after the default run succeeds:

```bash
python model/evaluate.py --limit 1000
```

The evaluator accepts custom Recall@K cutoffs:

```bash
python model/evaluate.py --limit 200 --top-k 1 5 10
```

## Outputs

Generated files are written to `model/results/`:

- `tfidf-index.joblib`: vectorizer and sparse TF-IDF matrix built by `tfidf-baseline.py`.
- `tfidf-metadata.json`: basic metadata about the baseline index.
- `tfidf-index-evaluation.joblib`: evaluation-specific TF-IDF index created by `evaluate.py`.
- `retrieval-predictions.csv`: each evaluated duplicate pair and the rank of its known master question.
- `metrics.json`: evaluation settings and Recall@K metrics.

Prepared intermediate CSVs are written to `model/data/`:

- `questions-prepared.csv`
- `duplicate-pairs.csv`

These are generated artifacts, not replacements for the original source datasets.

## Metric

**Recall@K** is the proportion of evaluated known duplicate links for which the linked master question appears within the top K retrieved results.

For example, Recall@10 = 0.70 means that 70% of the evaluated known duplicate links had their linked master in the top 10 results.

## Limitations and interpretation

1. The current evaluator defaults to the first 200 duplicate pairs after filtering and deduplication. This is a quick initial sample, not necessarily representative of all questions or duplicate links.
2. The index contains the evaluated questions. The evaluator excludes each query question from its own results, but this is not a fully held-out evaluation.
3. Questions without a recorded duplicate link are not necessarily true negatives.
4. TF-IDF measures lexical similarity. It can miss semantically similar questions with different wording and can over-rank questions that share terms but ask different things.
5. `prepare-data.py` uses basic regular-expression HTML cleanup. It is not a full HTML parser and may not preserve all code or markup semantics.
6. The word-only evaluator uses a 100,000-feature cap and float32 values to reduce memory use. Its metrics evaluate this lighter configuration, not the earlier word-plus-character configuration in `tfidf-baseline.py`.
7. Results depend on which questions and links are available in the supplied dataset files.

## Next steps

1. Inspect `model/results/metrics.json` and confirm that the metrics are finite and the evaluated pair count is as expected.
2. Run the evaluator on a larger sample, such as 1,000 pairs, and compare the Recall@K values.
3. Add a stronger evaluation split, preferably time-based or held-out by question, to reduce optimistic results from evaluating queries against a corpus containing the same questions.
4. Compare the TF-IDF baseline with a semantic embedding model, such as a Sentence Transformer, using the same evaluation pairs and metrics.
5. Keep the model configuration and dataset size with every recorded result so comparisons remain reproducible.
