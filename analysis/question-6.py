from pathlib import Path
import importlib.util

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# This project uses hyphens in filenames. Python cannot use a hyphenated
# filename in a normal "import common-data", so load it with importlib.
BASE_DIR = Path(__file__).resolve().parent.parent
COMMON_FILE = BASE_DIR / "analysis" / "common-data.py"

spec = importlib.util.spec_from_file_location("common_data", COMMON_FILE)
common_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common_data)

load_questions = common_data.load_questions
load_duplicate_question_ids = common_data.load_duplicate_question_ids

OUTPUT_DIR = BASE_DIR / "analysis-results" / "question-6"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


def main():
    print("========== QUESTION 6 ==========")

    # Load questions and IDs marked as duplicates.
    questions = load_questions()
    duplicate_ids = load_duplicate_question_ids()

    questions["is_duplicate"] = questions["id"].isin(duplicate_ids)

    # One row per question-tag pair. Exploding makes it possible to
    # count each question under every tag it has.
    question_tags = (
        questions[["id", "is_duplicate", "tag_list"]]
        .explode("tag_list")
        .rename(columns={"tag_list": "tag"})
    )

    question_tags = question_tags.dropna(subset=["tag"])
    question_tags = question_tags[question_tags["tag"].astype(str).str.len() > 0]

    # For each tag:
    # duplicate percentage = duplicate questions with tag /
    #                        all questions with tag * 100.
    tag_stats = question_tags.groupby("tag").agg(
        total_questions=("id", "nunique"),
        duplicate_questions=("is_duplicate", "sum"),
    )

    tag_stats["duplicate_percentage"] = (
        tag_stats["duplicate_questions"]
        / tag_stats["total_questions"]
        * 100
    )

    tag_stats = tag_stats.sort_values(
        ["duplicate_percentage", "duplicate_questions", "total_questions"],
        ascending=[False, False, False],
    )

    tag_stats = tag_stats.reset_index()

    print(f"Questions loaded: {len(questions):,}")
    print(f"Distinct duplicate IDs in PostLinks: {len(duplicate_ids):,}")
    print(f"Unique tags analysed: {len(tag_stats):,}")
    print("\nTop 20 tags by duplicate-question percentage:")
    print(tag_stats.head(20).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # Save all tag statistics and the top-20 subset.
    tag_stats.to_csv(OUTPUT_DIR / "tag-duplicate-percentages.csv", index=False)

    top_20 = tag_stats.head(20).copy()
    top_20.to_csv(OUTPUT_DIR / "top-20-duplicate-tags.csv", index=False)

    # Bar chart: highest duplicate percentages first.
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=top_20,
        x="duplicate_percentage",
        y="tag",
    )
    plt.title("Top 20 Tags by Duplicate-Question Percentage")
    plt.xlabel("Duplicate questions (% of questions with that tag)")
    plt.ylabel("Tag")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "top-20-duplicate-tags.png", dpi=300)
    plt.show()

    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
