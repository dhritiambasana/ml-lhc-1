
from pathlib import Path
from collections import Counter
import re

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "dataset"
OUTPUT_DIR = BASE_DIR / "analysis_results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

POST_FILE = DATA_DIR / "posts_short.csv"
LINKS_FILE = DATA_DIR / "postLinks.csv"

sns.set_theme(style="whitegrid")


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def extract_tags(value):
    """Extract tags from <python><pandas><numpy> format."""
    if pd.isna(value):
        return []

    return re.findall(r"<([^<>]+)>", str(value))


def normalize_columns(df):
    """Normalize column names to lowercase."""
    df.columns = df.columns.str.strip().str.lower()
    return df


# =========================================================
# LOAD DATA
# =========================================================

posts = pd.read_csv(POST_FILE, low_memory=False)
posts = normalize_columns(posts)

required_post_columns = {
    "id", "post_type_id", "creation_date", "tags"
}

if not required_post_columns.issubset(posts.columns):
    raise ValueError(
        f"Missing columns in posts_short.csv: "
        f"{required_post_columns - set(posts.columns)}"
    )

# Retain questions only.
posts = posts[posts["post_type_id"] == 1].copy()

# Remove duplicate records, if any.
posts = posts.drop_duplicates(subset="id")

posts["creation_date"] = pd.to_datetime(
    posts["creation_date"], errors="coerce", utc=True
)

posts["tag_list"] = posts["tags"].apply(extract_tags)

print("Total questions:", len(posts))


# =========================================================
# Q1. NUMBER OF TAGS PER QUESTION
# =========================================================

print("\n========== Q1: TAGS PER QUESTION ==========")

posts["tag_count"] = posts["tag_list"].apply(len)

tag_count_distribution = (
    posts["tag_count"].value_counts().sort_index()
)

print(tag_count_distribution)
print("\nSummary statistics:")
print(posts["tag_count"].describe())

posts[["id", "tag_count"]].to_csv(
    OUTPUT_DIR / "1-tags-per-question.csv",
    index=False
)

tag_count_distribution.to_csv(
    OUTPUT_DIR / "1-tag-count-distribution.csv",
    header=["number_of_questions"]
)

plt.figure(figsize=(9, 5))
tag_count_distribution.plot(kind="bar")
plt.title("Number of Tags Per Question")
plt.xlabel("Number of tags")
plt.ylabel("Number of questions")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "1-tags-per-question.png", dpi=300)
plt.show()


# =========================================================
# Q2. TOTAL NUMBER OF UNIQUE TAGS
# =========================================================

print("\n========== Q2: UNIQUE TAGS ==========")

all_tags = [
    tag
    for tag_list in posts["tag_list"]
    for tag in tag_list
]

tag_frequency = Counter(all_tags)
unique_tags = set(all_tags)

print("Total tag occurrences:", len(all_tags))
print("Total unique tags:", len(unique_tags))

pd.DataFrame({
    "tag": sorted(unique_tags)
}).to_csv(
    OUTPUT_DIR / "2-unique-tags.csv",
    index=False
)


# =========================================================
# Q3. TOP 25 MOST FREQUENT TAGS
# =========================================================

print("\n========== Q3: TOP 25 TAGS ==========")

top_25_df = pd.DataFrame(
    tag_frequency.most_common(25),
    columns=["tag", "frequency"]
)

print(top_25_df.to_string(index=False))

top_25_df.to_csv(
    OUTPUT_DIR / "3-top-25-tags.csv",
    index=False
)

plt.figure(figsize=(12, 8))
sns.barplot(
    data=top_25_df,
    x="frequency",
    y="tag"
)

plt.title("Top 25 Most Frequently Used Tags")
plt.xlabel("Frequency")
plt.ylabel("Tag")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "3-top-25-tags.png", dpi=300)
plt.show()


# =========================================================
# Q4. DISTRIBUTION OF TOP 500 TAGS
# =========================================================

print("\n========== Q4: TOP 500 TAG DISTRIBUTION ==========")

top_500_df = pd.DataFrame(
    tag_frequency.most_common(500),
    columns=["tag", "frequency"]
)

top_500_df["rank"] = range(1, len(top_500_df) + 1)

top_500_df.to_csv(
    OUTPUT_DIR / "4-top-500-tags.csv",
    index=False
)

# Frequency histogram
plt.figure(figsize=(10, 6))
sns.histplot(data=top_500_df, x="frequency", bins=30)
plt.title("Frequency Distribution of Top 500 Tags")
plt.xlabel("Tag frequency")
plt.ylabel("Number of tags")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "4-frequency-distribution.png", dpi=300)
plt.show()

# Rank-frequency plot
plt.figure(figsize=(10, 6))
plt.plot(
    top_500_df["rank"],
    top_500_df["frequency"],
    marker=".",
    linestyle="none"
)
plt.title("Rank-Frequency Distribution of Top 500 Tags")
plt.xlabel("Rank")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "4-rank-frequency.png", dpi=300)
plt.show()

# Log-log plot
plt.figure(figsize=(10, 6))
plt.loglog(
    top_500_df["rank"],
    top_500_df["frequency"],
    marker=".",
    linestyle="none"
)
plt.title("Log-Log Rank-Frequency Distribution")
plt.xlabel("Rank (log scale)")
plt.ylabel("Frequency (log scale)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "4-log-log-distribution.png", dpi=300)
plt.show()


# =========================================================
# Q5. MONTHLY DUPLICATE QUESTION RATIO, 2008-2018
# =========================================================

print("\n========== Q5: MONTHLY DUPLICATE RATIO ==========")

links = pd.read_csv(LINKS_FILE, low_memory=False)
links = normalize_columns(links)

required_link_columns = {
    "post_id", "related_post_id", "link_type_id"
}

if not required_link_columns.issubset(links.columns):
    raise ValueError(
        f"Missing columns in postLinks.csv: "
        f"{required_link_columns - set(links.columns)}. "
        f"Available columns: {links.columns.tolist()}"
    )

# LinkTypeId = 3 means PostId is a duplicate
# of RelatedPostId.
duplicate_links = links[links["link_type_id"] == 3]

# Use IDs from the source/duplicate side.
duplicate_ids = set(
    pd.to_numeric(
        duplicate_links["post_id"], errors="coerce"
    ).dropna().astype("int64")
)

posts["id"] = pd.to_numeric(posts["id"], errors="coerce")

# Only consider questions created from 2008 through 2018.
start_date = pd.Timestamp("2008-01-01", tz="UTC")
end_date = pd.Timestamp("2019-01-01", tz="UTC")

monthly_posts = posts[
    posts["creation_date"].ge(start_date)
    & posts["creation_date"].lt(end_date)
].copy()

monthly_posts["is_duplicate"] = (
    monthly_posts["id"].isin(duplicate_ids)
)

monthly_posts["month"] = (
    monthly_posts["creation_date"].dt.tz_localize(None).dt.to_period("M")
)

monthly_stats = monthly_posts.groupby("month").agg(
    total_questions=("id", "nunique"),
    duplicate_questions=("is_duplicate", "sum")
)

# Include all months from January 2008 to December 2018.
all_months = pd.period_range("2008-01", "2018-12", freq="M")
monthly_stats = monthly_stats.reindex(all_months)
monthly_stats.index.name = "month"

monthly_stats["total_questions"] = (
    monthly_stats["total_questions"].fillna(0).astype(int)
)

monthly_stats["duplicate_questions"] = (
    monthly_stats["duplicate_questions"].fillna(0).astype(int)
)

# Avoid division by zero for months with no records.
monthly_stats["duplicate_ratio_percent"] = (
    monthly_stats["duplicate_questions"]
    .div(monthly_stats["total_questions"].replace(0, float("nan")))
    * 100
)

print(monthly_stats.to_string())

monthly_stats.to_csv(
    OUTPUT_DIR / "5-monthly-duplicate-ratio.csv"
)

plt.figure(figsize=(16, 6))
plt.plot(
    monthly_stats.index.astype(str),
    monthly_stats["duplicate_ratio_percent"],
    linewidth=1.5
)

plt.title("Monthly Ratio of Duplicate Questions (2008-2018)")
plt.xlabel("Month")
plt.ylabel("Duplicate questions (%)")

tick_positions = list(range(0, len(all_months), 12))
plt.xticks(
    tick_positions,
    [str(all_months[i]) for i in tick_positions],
    rotation=45
)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "5-monthly-duplicate-ratio.png", dpi=300)
plt.show()


# =========================================================
# COMPLETE
# =========================================================

print("\nAll five analyses completed.")
print("Results directory:", OUTPUT_DIR.resolve())
