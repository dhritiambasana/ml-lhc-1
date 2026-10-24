from pathlib import Path
import importlib.util
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "analysis-results" / "question-9"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COMMON_FILE = BASE_DIR / "analysis" / "common-data.py"
spec = importlib.util.spec_from_file_location("common_data", COMMON_FILE)
common_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common_data)

load_questions = common_data.load_questions


def main():
    print("========== QUESTION 9 ==========")
    print("Goal: determine what percentage of questions are covered by the top 500 tags.")

    questions = load_questions()

    # Count each tag's occurrences across questions.
    tag_frequency = Counter(
        tag
        for tags in questions["tag_list"]
        for tag in tags
    )

    if not tag_frequency:
        raise ValueError("No tags were found in the loaded questions.")

    # Select the 500 most frequent tags.
    top_500_tags = {
        tag for tag, _ in tag_frequency.most_common(500)
    }

    # A question is covered if it has at least one of the top 500 tags.
    questions["covered_by_top_500"] = questions["tag_list"].apply(
        lambda tags: any(tag in top_500_tags for tag in tags)
    )

    total_questions = len(questions)
    covered_questions = int(questions["covered_by_top_500"].sum())
    uncovered_questions = total_questions - covered_questions

    covered_percentage = (
        covered_questions / total_questions * 100
        if total_questions else 0.0
    )

    results = pd.DataFrame([{
        "number_of_top_tags": len(top_500_tags),
        "total_questions": total_questions,
        "covered_questions": covered_questions,
        "uncovered_questions": uncovered_questions,
        "covered_percentage": covered_percentage,
    }])

    print(f"Questions analysed: {total_questions:,}")
    print(f"Top tags available: {len(top_500_tags):,}")
    print(f"Questions with at least one top-500 tag: {covered_questions:,}")
    print(f"Questions with none of the top-500 tags: {uncovered_questions:,}")
    print(f"Percentage covered: {covered_percentage:.2f}%")

    results.to_csv(
        OUTPUT_DIR / "top-500-tag-coverage.csv",
        index=False,
    )

    # Save the selected tags and their frequencies for traceability.
    top_500_df = pd.DataFrame(
        tag_frequency.most_common(500),
        columns=["tag", "frequency"],
    )
    top_500_df.to_csv(
        OUTPUT_DIR / "top-500-tags.csv",
        index=False,
    )

    # Simple bar chart comparing covered and uncovered questions.
    chart_data = pd.DataFrame({
        "coverage": ["Covered", "Not covered"],
        "number_of_questions": [covered_questions, uncovered_questions],
    })

    plt.figure(figsize=(8, 5))
    plt.bar(
        chart_data["coverage"],
        chart_data["number_of_questions"],
    )
    plt.title("Question Coverage by the Top 500 Tags")
    plt.xlabel("Coverage status")
    plt.ylabel("Number of questions")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "top-500-tag-coverage.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
