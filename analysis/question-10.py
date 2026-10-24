from pathlib import Path
import importlib.util

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "analysis-results" / "question-10"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COMMON_FILE = BASE_DIR / "analysis" / "common-data.py"
spec = importlib.util.spec_from_file_location("common_data", COMMON_FILE)
common_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common_data)

load_questions = common_data.load_questions


def main():
    print("========== QUESTION 10 ==========")
    print("Goal: determine the percentage of questions covered by the top 5,000 tags.")

    questions = load_questions()

    # Count each tag occurrence across all loaded questions.
    tag_frequency = {}
    for tags in questions["tag_list"]:
        for tag in tags:
            tag_frequency[tag] = tag_frequency.get(tag, 0) + 1

    if not tag_frequency:
        raise ValueError("No tags were found in the loaded questions.")

    # Take the 5,000 most frequently occurring tags, or all tags if fewer exist.
    top_5000 = sorted(
        tag_frequency.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:5000]
    top_5000_tags = {tag for tag, _ in top_5000}

    # A question is covered if at least one of its tags is in the top-5000 set.
    questions["covered_by_top_5000"] = questions["tag_list"].apply(
        lambda tags: any(tag in top_5000_tags for tag in tags)
    )

    total_questions = len(questions)
    covered_questions = int(questions["covered_by_top_5000"].sum())
    uncovered_questions = total_questions - covered_questions

    covered_percentage = (
        covered_questions / total_questions * 100
        if total_questions else 0.0
    )

    summary = pd.DataFrame([{
        "number_of_top_tags": len(top_5000_tags),
        "total_questions": total_questions,
        "covered_questions": covered_questions,
        "uncovered_questions": uncovered_questions,
        "covered_percentage": covered_percentage,
    }])

    print(f"Questions analysed: {total_questions:,}")
    print(f"Top tags available: {len(top_5000_tags):,}")
    print(f"Questions with at least one top-5000 tag: {covered_questions:,}")
    print(f"Questions with none of the top-5000 tags: {uncovered_questions:,}")
    print(f"Percentage covered: {covered_percentage:.2f}%")

    summary.to_csv(
        OUTPUT_DIR / "top-5000-tag-coverage.csv",
        index=False,
    )

    top_5000_df = pd.DataFrame(
        top_5000,
        columns=["tag", "frequency"],
    )
    top_5000_df.to_csv(
        OUTPUT_DIR / "top-5000-tags.csv",
        index=False,
    )

    chart_data = pd.DataFrame({
        "coverage": ["Covered", "Not covered"],
        "number_of_questions": [covered_questions, uncovered_questions],
    })

    plt.figure(figsize=(8, 5))
    plt.bar(
        chart_data["coverage"],
        chart_data["number_of_questions"],
    )
    plt.title("Question Coverage by the Top 5,000 Tags")
    plt.xlabel("Coverage status")
    plt.ylabel("Number of questions")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "top-5000-tag-coverage.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
