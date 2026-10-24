from pathlib import Path
import importlib.util

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "dataset"
OUTPUT_DIR = BASE_DIR / "analysis-results" / "question-8"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Reuse the shared data-loading helpers created for earlier questions.
COMMON_FILE = BASE_DIR / "analysis" / "common-data.py"
spec = importlib.util.spec_from_file_location("common_data", COMMON_FILE)
common_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common_data)

load_questions = common_data.load_questions
load_duplicate_question_ids = common_data.load_duplicate_question_ids
normalize_columns = common_data.normalize_columns

sns.set_theme(style="whitegrid")


def main():
    print("========== QUESTION 8 ==========")
    print("Goal: examine the reputation of users who asked duplicate questions.")

    # Load questions marked as duplicates.
    questions = load_questions()
    duplicate_ids = load_duplicate_question_ids()

    required_question_columns = {"owner_user_id"}
    missing = required_question_columns - set(questions.columns)
    if missing:
        raise ValueError(
            f"Question data is missing columns: {sorted(missing)}"
        )

    duplicate_questions = questions[
        questions["id"].isin(duplicate_ids)
    ].copy()

    duplicate_questions["owner_user_id"] = pd.to_numeric(
        duplicate_questions["owner_user_id"], errors="coerce"
    )
    duplicate_questions = duplicate_questions.dropna(
        subset=["owner_user_id"]
    )
    duplicate_questions["owner_user_id"] = (
        duplicate_questions["owner_user_id"].astype("int64")
    )

    # Load user reputation.
    users_path = DATA_DIR / "users.csv"
    if not users_path.exists():
        raise FileNotFoundError(
            f"Cannot find {users_path}. Question 8 requires users.csv."
        )

    users = normalize_columns(pd.read_csv(users_path, low_memory=False))

    required_user_columns = {"id", "reputation"}
    missing = required_user_columns - set(users.columns)
    if missing:
        raise ValueError(
            f"users.csv is missing columns: {sorted(missing)}. "
            f"Available columns: {users.columns.tolist()}"
        )

    users["id"] = pd.to_numeric(users["id"], errors="coerce")
    users["reputation"] = pd.to_numeric(
        users["reputation"], errors="coerce"
    )
    users = users.dropna(subset=["id", "reputation"])
    users["id"] = users["id"].astype("int64")
    users = users.drop_duplicates(subset="id", keep="first")

    # Match duplicate-question authors to their user reputation.
    # Count each user once, even if they asked several duplicate questions.
    duplicate_authors = (
        duplicate_questions[["owner_user_id"]]
        .drop_duplicates()
        .merge(
            users[["id", "reputation"]],
            left_on="owner_user_id",
            right_on="id",
            how="inner",
        )
        .drop_duplicates(subset="owner_user_id")
    )

    if duplicate_authors.empty:
        raise ValueError(
            "No duplicate-question authors could be matched to users.csv. "
            "Check whether owner_user_id and users.id use matching IDs."
        )

    # Reputation categories. These are analysis choices, not official
    # Stack Overflow reputation levels.
    bins = [-1, 1, 10, 100, 200, 500, 1000, 5000, 10000, float("inf")]
    labels = [
        "0–1",
        "2–10",
        "11–100",
        "101–200",
        "201–500",
        "501–1,000",
        "1,001–5,000",
        "5,001–10,000",
        "Over 10,000",
    ]

    duplicate_authors["reputation_category"] = pd.cut(
        duplicate_authors["reputation"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    # Keep categories in the intended order, including categories with zero users.
    distribution = (
        duplicate_authors["reputation_category"]
        .value_counts(sort=False)
        .reindex(labels, fill_value=0)
        .rename_axis("reputation_category")
        .reset_index(name="number_of_users")
    )

    print(f"Duplicate questions in posts data: {len(duplicate_questions):,}")
    print(f"Distinct authors matched to users.csv: {len(duplicate_authors):,}")
    print("\nUsers by reputation category:")
    print(distribution.to_string(index=False))

    # Save detailed and summarized results.
    duplicate_authors[
        ["owner_user_id", "reputation", "reputation_category"]
    ].to_csv(
        OUTPUT_DIR / "duplicate-authors-by-reputation.csv",
        index=False,
    )
    distribution.to_csv(
        OUTPUT_DIR / "reputation-distribution.csv",
        index=False,
    )

    # Plot number of distinct users in each reputation range.
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=distribution,
        x="reputation_category",
        y="number_of_users",
        color="steelblue",
    )
    ax.set_title("Reputation Distribution of Users Who Asked Duplicate Questions")
    ax.set_xlabel("User reputation")
    ax.set_ylabel("Number of distinct users")

    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "reputation-distribution.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

