"""Prepare Stack Overflow questions and known duplicate links.

Run from the repository root:
    python model/prepare-data.py
"""
from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "dataset"
OUT_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def clean_html(text) -> str:
    """Basic HTML cleanup; this is not a full HTML parser."""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"<code\b[^>]*>.*?</code>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&lt;|&#60;", "<", text, flags=re.I)
    text = re.sub(r"&gt;|&#62;", ">", text, flags=re.I)
    text = re.sub(r"&quot;|&#34;", '"', text, flags=re.I)
    text = re.sub(r"&#39;|&apos;", "'", text, flags=re.I)
    text = re.sub(r"&amp;", "&", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required dataset not found: {path}")
    return normalize_columns(pd.read_csv(path, low_memory=False))


def main():
    questions_path = DATA_DIR / "posts_long.csv"
    links_path = DATA_DIR / "postLinks.csv"

    questions = read_csv_required(questions_path)
    links = read_csv_required(links_path)

    required_question_cols = {"id", "post_type_id", "title", "body"}
    missing = required_question_cols - set(questions.columns)
    if missing:
        raise ValueError(f"posts_long.csv is missing columns: {sorted(missing)}")

    questions["id"] = pd.to_numeric(questions["id"], errors="coerce")
    questions["post_type_id"] = pd.to_numeric(questions["post_type_id"], errors="coerce")
    questions = questions[questions["post_type_id"] == 1].copy()
    questions = questions.dropna(subset=["id"]).drop_duplicates(subset=["id"])
    questions["id"] = questions["id"].astype("int64")
    questions["title"] = questions["title"].fillna("").astype(str).map(clean_html)
    questions["body"] = questions["body"].fillna("").astype(str).map(clean_html)
    questions["text"] = (questions["title"] + " " + questions["body"]).str.strip()
    questions = questions[questions["text"].str.len() > 0].copy()

    required_link_cols = {"post_id", "related_post_id", "link_type_id"}
    missing = required_link_cols - set(links.columns)
    if missing:
        raise ValueError(f"postLinks.csv is missing columns: {sorted(missing)}")

    for col in ["post_id", "related_post_id", "link_type_id"]:
        links[col] = pd.to_numeric(links[col], errors="coerce")
    # Stack Overflow link_type_id == 3 denotes a duplicate link.
    duplicate_pairs = links[links["link_type_id"] == 3][
        ["post_id", "related_post_id"]
    ].dropna().drop_duplicates().copy()
    duplicate_pairs["post_id"] = duplicate_pairs["post_id"].astype("int64")
    duplicate_pairs["related_post_id"] = duplicate_pairs["related_post_id"].astype("int64")

    question_ids = set(questions["id"])
    duplicate_pairs = duplicate_pairs[
        duplicate_pairs["post_id"].isin(question_ids)
        & duplicate_pairs["related_post_id"].isin(question_ids)
    ].copy()

    questions_out = OUT_DIR / "questions-prepared.csv"
    pairs_out = OUT_DIR / "duplicate-pairs.csv"
    questions[["id", "title", "body", "text"]].to_csv(questions_out, index=False)
    duplicate_pairs.to_csv(pairs_out, index=False)

    print(f"Prepared questions: {len(questions):,} -> {questions_out}")
    print(f"Known duplicate pairs: {len(duplicate_pairs):,} -> {pairs_out}")
    print("Note: only duplicate links whose two questions exist in posts_long.csv are retained.")


if __name__ == "__main__":
    main()
