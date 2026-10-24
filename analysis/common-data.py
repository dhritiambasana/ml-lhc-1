from pathlib import Path
import re

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "dataset"


def normalize_columns(df):
    """Standardize CSV column names to lowercase, trimmed names."""
    df.columns = df.columns.str.strip().str.lower()
    return df


def extract_tags(value):
    """Extract tags from strings such as <python><pandas><numpy>."""
    if pd.isna(value):
        return []

    value = str(value)

    if "<" in value and ">" in value:
        return re.findall(r"<([^<>]+)>", value)

    return [
        tag.strip()
        for tag in re.split(r"[;\s]+", value.strip())
        if tag.strip()
    ]


def load_questions(filename="posts_short.csv"):
    """Load questions with the columns used by the analyses."""
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Posts file not found: {path}")

    posts = normalize_columns(pd.read_csv(path, low_memory=False))

    required = {"id", "post_type_id", "creation_date", "tags"}
    missing = required - set(posts.columns)
    if missing:
        raise ValueError(
            f"{path.name} is missing columns {sorted(missing)}. "
            f"Available columns: {posts.columns.tolist()}"
        )

    posts = posts[posts["post_type_id"] == 1].copy()
    posts = posts.drop_duplicates(subset="id")
    posts["id"] = pd.to_numeric(posts["id"], errors="coerce")
    posts = posts.dropna(subset=["id"])
    posts["id"] = posts["id"].astype("int64")
    posts["creation_date"] = pd.to_datetime(
        posts["creation_date"], errors="coerce", utc=True
    )
    posts["tag_list"] = posts["tags"].apply(extract_tags)

    return posts


def load_duplicate_question_ids(filename="postLinks.csv"):
    """Return duplicate question IDs from PostLinks rows with LinkTypeId=3."""
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"PostLinks file not found: {path}")

    links = normalize_columns(pd.read_csv(path, low_memory=False))

    required = {"post_id", "related_post_id", "link_type_id"}
    missing = required - set(links.columns)
    if missing:
        raise ValueError(
            f"{path.name} is missing columns {sorted(missing)}. "
            f"Available columns: {links.columns.tolist()}"
        )

    link_type = pd.to_numeric(links["link_type_id"], errors="coerce")
    source_ids = pd.to_numeric(
        links.loc[link_type == 3, "post_id"], errors="coerce"
    ).dropna()

    return set(source_ids.astype("int64").unique())
