from pathlib import Path
import importlib.util

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent.parent
COMMON_FILE = BASE_DIR / "analysis" / "common-data.py"

spec = importlib.util.spec_from_file_location("common_data", COMMON_FILE)
common_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common_data)

load_questions = common_data.load_questions
load_duplicate_question_ids = common_data.load_duplicate_question_ids
normalize_columns = common_data.normalize_columns

DATA_DIR = BASE_DIR / "data" / "dataset"
OUTPUT_DIR = BASE_DIR / "analysis-results" / "question-7"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


def main():
    print("========== QUESTION 7 ==========")
    print("Goal: estimate how long duplicate questions take to be closed.")

    questions = load_questions()
    duplicate_ids = load_duplicate_question_ids()

    history_path = DATA_DIR / "post_history.csv"
    if not history_path.exists():
        raise FileNotFoundError(
            f"Cannot find {history_path}. Question 7 needs post_history.csv "
            "to obtain closure timestamps."
        )

    history = normalize_columns(
        pd.read_csv(history_path, low_memory=False)
    )

    required = {"post_id", "ph_type_id", "creation_date"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(
            f"post_history.csv is missing columns {sorted(missing)}. "
            f"Available columns: {history.columns.tolist()}"
        )

    history["post_id"] = pd.to_numeric(history["post_id"], errors="coerce")
    history["ph_type_id"] = pd.to_numeric(
        history["ph_type_id"], errors="coerce"
    )
    history["creation_date"] = pd.to_datetime(
        history["creation_date"], errors="coerce", utc=True
    )

    # Stack Exchange PostHistoryTypeId=10 represents a post being closed.
    close_events = history[
        (history["ph_type_id"] == 10)
        & (history["post_id"].isin(duplicate_ids))
    ].copy()

    # The first close event is used if a question has multiple close records.
    close_events = (
        close_events.dropna(subset=["post_id", "creation_date"])
        .sort_values("creation_date")
        .drop_duplicates(subset="post_id", keep="first")
    )

    # Match duplicate questions to their original creation timestamps.
    duplicate_questions = questions[
        questions["id"].isin(duplicate_ids)
    ][["id", "creation_date"]].rename(
        columns={"id": "post_id", "creation_date": "question_created_at"}
    )

    durations = duplicate_questions.merge(
        close_events[["post_id", "creation_date"]].rename(
            columns={"creation_date": "closed_at"}
        ),
        on="post_id",
        how="inner",
    )

    durations = durations.dropna(
        subset=["question_created_at", "closed_at"]
    )

    durations["hours_to_close"] = (
        durations["closed_at"] - durations["question_created_at"]
    ).dt.total_seconds() / 3600

    # Exclude negative durations caused by inconsistent or invalid timestamps.
    durations = durations[durations["hours_to_close"] >= 0].copy()

    if durations.empty:
        raise ValueError(
            "No valid closure durations were found. Check that post_history.csv "
            "contains PostHistoryTypeId=10 records and that the IDs match "
            "the duplicate IDs in postLinks.csv."
        )

    durations["days_to_close"] = durations["hours_to_close"] / 24

    print(f"Duplicate IDs from postLinks: {len(duplicate_ids):,}")
    print(f"Duplicate questions with matched close events: {len(durations):,}")
    print("\nTime-to-close summary (hours):")
    print(durations["hours_to_close"].describe())

    # Save per-question durations.
    durations.rename(columns={"post_id": "question_id"}).to_csv(
        OUTPUT_DIR / "duplicate-question-close-times.csv",
        index=False,
    )

    # Group durations into readable time buckets for the requested bar chart.
    bins = [-0.000001, 1, 6, 24, 72, 168, 720, float("inf")]
    labels = [
        "Under 1 hour",
        "1–6 hours",
        "6–24 hours",
        "1–3 days",
        "3–7 days",
        "7–30 days",
        "Over 30 days",
    ]

    durations["time_to_close_bucket"] = pd.cut(
        durations["hours_to_close"],
        bins=bins,
        labels=labels,
        right=True,
        include_lowest=True,
    )

    bucket_counts = (
        durations["time_to_close_bucket"]
        .value_counts(sort=False)
        .rename_axis("time_to_close")
        .reset_index(name="number_of_questions")
    )

    bucket_counts.to_csv(
        OUTPUT_DIR / "close-time-distribution.csv",
        index=False,
    )

    plt.figure(figsize=(11, 6))
    ax = sns.barplot(
        data=bucket_counts,
        x="time_to_close",
        y="number_of_questions",
        color="steelblue",
    )
    ax.set_title("Time Required to Close Duplicate Questions")
    ax.set_xlabel("Time between question creation and closure")
    ax.set_ylabel("Number of duplicate questions")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "time-to-close-distribution.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
