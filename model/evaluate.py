
"""Memory-conscious evaluation of Stack Overflow duplicate retrieval."""

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

MODEL_DIR = Path(__file__).resolve().parent
DATA_DIR = MODEL_DIR / "data"
RESULTS_DIR = MODEL_DIR / "results"

QUESTIONS_PATH = DATA_DIR / "questions-prepared.csv"
PAIRS_PATH = DATA_DIR / "duplicate-pairs.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=200,
        help="Number of duplicate pairs to evaluate; 0 means all"
    )
    parser.add_argument(
        "--top-k", type=int, nargs="+", default=[1, 5, 10]
    )
    args = parser.parse_args()

    if args.limit < 0 or not args.top_k or min(args.top_k) < 1:
        raise ValueError("--limit must be >= 0 and --top-k values must be positive")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for path in (QUESTIONS_PATH, PAIRS_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run model/prepare-data.py first."
            )

    print("Loading prepared data...", flush=True)
    questions = pd.read_csv(QUESTIONS_PATH)
    pairs = pd.read_csv(PAIRS_PATH)

    questions["id"] = pd.to_numeric(
        questions["id"], errors="raise"
    ).astype("int64")
    pairs["post_id"] = pd.to_numeric(
        pairs["post_id"], errors="coerce"
    )
    pairs["related_post_id"] = pd.to_numeric(
        pairs["related_post_id"], errors="coerce"
    )
    pairs = pairs.dropna(
        subset=["post_id", "related_post_id"]
    ).copy()
    pairs["post_id"] = pairs["post_id"].astype("int64")
    pairs["related_post_id"] = pairs["related_post_id"].astype("int64")

    id_to_index = {
        qid: idx for idx, qid in enumerate(questions["id"])
    }

    pairs = pairs[
        pairs["post_id"].isin(id_to_index)
        & pairs["related_post_id"].isin(id_to_index)
    ].drop_duplicates()

    if args.limit:
        pairs = pairs.head(args.limit)

    if pairs.empty:
        raise ValueError("No usable duplicate pairs were found.")

    print(f"Questions: {len(questions):,}", flush=True)
    print(f"Duplicate pairs to evaluate: {len(pairs):,}", flush=True)
    print("Building word TF-IDF matrix...", flush=True)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=100_000,
        sublinear_tf=True,
        dtype=np.float32
    )

    matrix = vectorizer.fit_transform(
        questions["text"].fillna("").astype(str)
    )

    print(
        f"Matrix ready: {matrix.shape[0]:,} questions, "
        f"{matrix.shape[1]:,} features.",
        flush=True
    )

    ks = sorted(set(args.top_k))
    max_k = min(max(ks), max(1, len(questions) - 1))
    hits = {k: 0 for k in ks}
    rows = []

    # Each iteration creates only one question-to-corpus score vector.
    for number, pair in enumerate(pairs.itertuples(index=False), start=1):
        query_id = int(pair.post_id)
        target_id = int(pair.related_post_id)
        query_idx = id_to_index[query_id]
        target_idx = id_to_index[target_id]

        # Sparse multiplication; only convert this one score row to dense.
        scores = (matrix[query_idx] @ matrix.T).toarray().ravel()

        # Do not recommend the query question as its own duplicate.
        scores[query_idx] = -1.0

        # Stable descending ranking; self has already been excluded.
        ranked = np.argsort(-scores, kind="stable")[:max_k]
        ranked_ids = questions.iloc[ranked]["id"].to_numpy()

        target_positions = np.flatnonzero(ranked_ids == target_id)
        rank = int(target_positions[0] + 1) if len(target_positions) else None

        for k in ks:
            if rank is not None and rank <= k:
                hits[k] += 1

        rows.append({
            "duplicate_question_id": query_id,
            "master_question_id": target_id,
            "master_rank": rank
        })

        if number % 25 == 0 or number == len(pairs):
            print(
                f"Evaluated {number:,}/{len(pairs):,} pairs",
                flush=True
            )

    total = len(rows)
    metrics = {
        "evaluated_duplicate_pairs": total,
        "question_count": int(len(questions)),
        "feature_count": int(matrix.shape[1]),
        "method": "Word TF-IDF, 1-2 grams, max 100000 features",
        "recall_at_k": {
            str(k): round(hits[k] / total, 4) for k in ks
        },
        "limitation": (
            "This is a sampled retrieval evaluation. The corpus includes "
            "the evaluated questions, although each query is excluded from "
            "its own results. Unlinked pairs are not assumed to be negatives."
        )
    }

    pd.DataFrame(rows).to_csv(
        RESULTS_DIR / "retrieval-predictions.csv", index=False
    )
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print("\nEvaluation results:")
    print(json.dumps(metrics, indent=2))
    print(f"Results saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
