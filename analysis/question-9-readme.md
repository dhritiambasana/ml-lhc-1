# Question 9: Percentage of Questions Covered by the Top 500 Tags

## Problem / issue being solved

Determine what percentage of questions in the available dataset are covered by
the 500 most frequent tags. This measures how much of the question collection
can be represented by its most commonly used tags.

## Approach

1. Load question posts from `posts_short.csv` using the shared functions in
   `analysis/common-data.py`.
2. Extract each question's tags and count the total occurrences of every tag.
3. Select the 500 tags with the highest occurrence counts (or all available
   tags if fewer than 500 distinct tags exist).
4. Mark a question as covered if it has at least one tag in this top-500 set.
   A question with several top-500 tags is still counted only once.
5. Calculate:

   `coverage percentage = covered questions / total questions * 100`

6. Save the summary, the selected top tags, and a bar chart comparing covered
   and uncovered questions.

## Input files

Expected input:

- `data/dataset/posts_short.csv`
- `analysis/common-data.py` (shared helper file from earlier questions)

The posts file must include `id`, `post_type_id`, `creation_date`, and `tags`
columns, with column names normalized by the shared loader.

## Output files

Saved under `analysis-results/question-9/`:

- `top-500-tag-coverage.csv`: total, covered, and uncovered question counts,
  plus the coverage percentage.
- `top-500-tags.csv`: the selected tags and their frequencies.
- `top-500-tag-coverage.png`: bar chart comparing covered and uncovered questions.

## Run

From the project root:

```bash
python analysis/question-9.py
```

## Interpretation and limitations

A question is considered covered when it contains at least one of the top 500
tags. This does not mean every topic in the question is represented by those
tags, only that at least one of its tags belongs to the top-500 set.

The result describes only the questions present in `posts_short.csv`. If this
file is a sample, the reported percentage applies to that sample, not necessarily
to the full Stack Overflow dataset.
