"""Build a TF-IDF question retrieval index and retrieve likely duplicates.

Run from the repository root:
    python model/tfidf-baseline.py
    python model/tfidf-baseline.py --query "How do I sort a list in Python?"
"""
from pathlib import Path
import numpy as np
import argparse
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.pipeline import FeatureUnion
from joblib import dump, load

MODEL_DIR = Path(__file__).resolve().parent
DATA_DIR = MODEL_DIR / "data"
RESULTS_DIR = MODEL_DIR / "results"
QUESTIONS_PATH = DATA_DIR / "questions-prepared.csv"
INDEX_PATH = RESULTS_DIR / "tfidf-index.joblib"
META_PATH = RESULTS_DIR / "tfidf-metadata.json"


def build_index(questions):
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

    return vectorizer, matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", help="A question title or text to search for")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to display")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the TF-IDF index")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(
            f"{QUESTIONS_PATH} not found. Run `python model/prepare-data.py` first."
        )

    questions = pd.read_csv(QUESTIONS_PATH, low_memory=False)
    questions["id"] = pd.to_numeric(questions["id"], errors="raise").astype("int64")

    if args.rebuild or not INDEX_PATH.exists():
        vectorizer, matrix = build_index(questions)
        dump({"vectorizer": vectorizer, "matrix": matrix}, INDEX_PATH)
        META_PATH.write_text(json.dumps({
            "questions_path": str(QUESTIONS_PATH),
            "question_count": int(len(questions)),
            "features": "word 1-2 grams + character 3-5 grams",
            "similarity": "cosine similarity (computed by linear_kernel for normalized TF-IDF vectors)"
        }, indent=2), encoding="utf-8")
        print(f"Built TF-IDF index for {len(questions):,} questions: {INDEX_PATH}")
    else:
        saved = load(INDEX_PATH)
        vectorizer, matrix = saved["vectorizer"], saved["matrix"]
        print(f"Loaded TF-IDF index: {INDEX_PATH}")

    if not args.query:
        print('No query supplied. Example: python model/tfidf-baseline.py --query "Python list sorting"')
        return

    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")

    query_vector = vectorizer.transform([args.query])
    scores = linear_kernel(query_vector, matrix).ravel()
    top_indices = scores.argsort()[::-1][:args.top_k]
    results = questions.iloc[top_indices][["id", "title"]].copy()
    results["similarity"] = scores[top_indices]
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
