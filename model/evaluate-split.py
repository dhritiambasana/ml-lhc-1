"""Held-out evaluation for duplicate-question retrieval.

Run from the repository root:
    python model/evaluate-split.py
    python model/evaluate-split.py --limit 1000 --top-k 1 5 10

This creates a deterministic train/test split of known duplicate pairs. Questions
used as test queries are removed from the retrieval corpus to reduce leakage.
The linked master must still exist in the searchable corpus to count as a hit.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

MODEL_DIR = Path(__file__).resolve().parent
DATA_DIR = MODEL_DIR / "data"
RESULTS_DIR = MODEL_DIR / "results"
QUESTIONS_PATH = DATA_DIR / "questions-prepared.csv"
PAIRS_PATH = DATA_DIR / "duplicate-pairs.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=1000,
        help="Maximum duplicate pairs to use; 0 means all available pairs"
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 5, 10])
    args = parser.parse_args()

    if args.limit < 0:
        raise ValueError("--limit must be >= 0")
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1")
    if not args.top_k or min(args.top_k) < 1:
        raise ValueError("All --top-k values must be positive")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for path in (QUESTIONS_PATH, PAIRS_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python model/prepare-data.py` first."
            )

    print("Loading prepared data...", flush=True)
    questions = pd.read_csv(QUESTIONS_PATH, low_memory=False)
    pairs = pd.read_csv(PAIRS_PATH, low_memory=False)

    questions["id"] = pd.to_numeric(questions["id"], errors="raise").astype("int64")
    pairs["post_id"] = pd.to_numeric(pairs["post_id"], errors="coerce")
    pairs["related_post_id"] = pd.to_numeric(pairs["related_post_id"], errors="coerce")
    pairs = pairs.dropna(subset=["post_id", "related_post_id"]).copy()
    pairs["post_id"] = pairs["post_id"].astype("int64")
    pairs["related_post_id"] = pairs["related_post_id"].astype("int64")
    pairs = pairs.drop_duplicates()

    id_to_index = {int(qid): idx for idx, qid in enumerate(questions["id"])}
    pairs = pairs[
        pairs["post_id"].isin(id_to_index)
        & pairs["related_post_id"].isin(id_to_index)
        & (pairs["post_id"] != pairs["related_post_id"])
    ].copy()

    # If several duplicate questions link to the same master, keep the rows.
    # Split by query ID so the same query cannot occur in both sets.
    pairs = pairs.drop_duplicates(subset=["post_id"], keep="first")
    if args.limit:
        pairs = pairs.head(args.limit)

    if len(pairs) < 2:
        raise ValueError("Need at least two usable duplicate pairs for a held-out split.")

    train_pairs, test_pairs = train_test_split(
        pairs,
        test_size=args.test_size,
        random_state=args.random_state,
        shuffle=True
    )

    # Keep every question except the held-out duplicate queries in the index.
    test_query_ids = set(test_pairs["post_id"].astype(int))
    corpus = questions[~questions["id"].isin(test_query_ids)].reset_index(drop=True)
    corpus_ids = corpus["id"].to_numpy(dtype=np.int64)
    corpus_id_to_index = {int(qid): idx for idx, qid in enumerate(corpus_ids)}
    test_pairs = test_pairs[
        test_pairs["related_post_id"].isin(corpus_id_to_index)
    ].reset_index(drop=True)

    if test_pairs.empty:
        raise ValueError(
            "No test master questions remain in the corpus. Try a different split or more data."
        )

    print(f"All questions: {len(questions):,}", flush=True)
    print(f"Training duplicate pairs: {len(train_pairs):,}", flush=True)
    print(f"Test duplicate pairs with searchable masters: {len(test_pairs):,}", flush=True)
    print(f"Search corpus after removing test queries: {len(corpus):,}", flush=True)
    print("Fitting word TF-IDF index...", flush=True)

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
    matrix = vectorizer.fit_transform(corpus["text"].fillna("").astype(str))

    ks = sorted(set(args.top_k))
    max_k = min(max(ks), len(corpus))
    hits = {k: 0 for k in ks}
    rows = []

    for number, pair in enumerate(test_pairs.itertuples(index=False), start=1):
        query_id = int(pair.post_id)
        master_id = int(pair.related_post_id)
        query_text = questions.loc[
            questions["id"] == query_id, "text"
        ].iloc[0]

        query_vector = vectorizer.transform([str(query_text)])
        scores = (query_vector @ matrix.T).toarray().ravel()
        ranked_indices = np.argsort(-scores, kind="stable")[:max_k]
        ranked_ids = corpus_ids[ranked_indices]
        positions = np.flatnonzero(ranked_ids == master_id)
        rank = int(positions[0] + 1) if len(positions) else None

        for k in ks:
            if rank is not None and rank <= k:
                hits[k] += 1

        rows.append({
            "duplicate_question_id": query_id,
            "master_question_id": master_id,
            "master_rank": rank,
            "reciprocal_rank": 1.0 / rank if rank is not None else 0.0
        })

        if number % 25 == 0 or number == len(test_pairs):
            print(f"Evaluated {number:,}/{len(test_pairs):,} test pairs", flush=True)

    total = len(rows)
    metrics = {
        "question_count_total": int(len(questions)),
        "search_corpus_count": int(len(corpus)),
        "training_duplicate_pairs": int(len(train_pairs)),
        "test_duplicate_pairs_requested": int(round(len(pairs) * args.test_size)),
        "test_duplicate_pairs_evaluated": total,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "max_duplicate_pairs": args.limit,
        "method": "Word TF-IDF (1-2 grams), max 100000 features, float32",
        "recall_at_k": {str(k): round(hits[k] / total, 4) for k in ks},
        "mean_reciprocal_rank": round(
            float(np.mean([row["reciprocal_rank"] for row in rows])), 4
        ),
        "interpretation": (
            "Test duplicate query questions are excluded from the search corpus. "
            "Only pairs whose linked master remains in the corpus are evaluated. "
            "This is a held-out-query split, not a fully time-based or entity-independent "
            "benchmark; related or near-identical questions can still appear in the corpus."
        )
    }

    predictions_path = RESULTS_DIR / "heldout-retrieval-predictions.csv"
    metrics_path = RESULTS_DIR / "heldout-metrics.json"
    pd.DataFrame(rows).to_csv(predictions_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nHeld-out evaluation metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"\nPredictions: {predictions_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
