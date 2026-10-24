# Question 10: Percentage of Questions Covered by the Top 5,000 Tags

## Problem / issue being solved

Determine what percentage of questions in the available dataset are covered by
the 5,000 most frequent tags. This extends Question 9, which performs the same
analysis using the top 500 tags, and allows the coverage values to be compared.

## Approach

1. Load question posts from `posts_short.csv` using the shared functions in
   `analysis/common-data.py`.
2. Extract the tags for each question and count how frequently each tag occurs.
3. Select the 5,000 most frequent tags, or all distinct tags if the dataset has
   fewer than 5,000.
4. Mark a question as covered if it has at least one tag in the top-5,000 set.
   Each question is counted only once, regardless of how many matching tags it has.
5. Calculate:

   `coverage percentage = covered questions / total questions * 100`

6. Save the summary, selected tags and frequencies, and a bar chart comparing
   covered and uncovered questions.

## Input files

Expected input:

- `data/dataset/posts_short.csv`
- `analysis/common-data.py` (shared helper file from earlier questions)

The posts file must contain the columns required by the shared loader:
`id`, `post_type_id`, `creation_date`, and `tags`.

## Output files

Saved under `analysis-results/question-10/`:

- `top-5000-tag-coverage.csv`: total, covered, and uncovered question counts,
  plus the coverage percentage.
- `top-5000-tags.csv`: selected tags and their frequencies.
- `top-5000-tag-coverage.png`: bar chart comparing covered and uncovered questions.

## Run

From the project root:

```bash
python analysis/question-10.py
```

## Interpretation and limitations

Coverage means a question has at least one of the top 5,000 tags. It does not
mean all topics in that question are represented by those tags.

Compare the percentage with Question 9. The top-5,000 coverage should be at
least as high as the top-500 coverage when both analyses use the same dataset
and the same tag-counting method.

The result describes only the questions present in `posts_short.csv`. If that
file is a sample, the reported percentage applies to that sample, not necessarily
to the full Stack Overflow dataset.
