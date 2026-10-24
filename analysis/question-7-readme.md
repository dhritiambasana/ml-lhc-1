# Question 7: Time Required to Close Duplicate Questions

## Problem / issue being solved

This analysis investigates how long duplicate Stack Overflow questions remain open
before they are closed. The goal is to understand the time between a question being
created and its recorded closure.

## Approach

1. Read question IDs and creation timestamps from `posts_short.csv`.
2. Identify duplicate-question IDs using `postLinks.csv` rows where `link_type_id`
   is `3`. In the Stack Exchange schema, the source `post_id` is the duplicate
   question and `related_post_id` points to the master question.
3. Read `post_history.csv` and select closure events with
   `post_history_type_id == 10` (post closed).
4. For each duplicate question with a matching closure event, calculate:
   `time to close = closure timestamp - question creation timestamp`.
5. Exclude invalid negative durations and save the per-question measurements.
6. Group the durations into time ranges and generate a bar chart showing the
   number of duplicate questions in each range.

## Input files

Expected under `data/dataset/`:

- `posts_short.csv`
- `postLinks.csv`
- `post_history.csv`

The columns used are `id`, `post_type_id`, `creation_date`, `post_id`,
`related_post_id`, `link_type_id`, and `post_history_type_id` (case-insensitive
after normalization).

## Output files

Results are saved under `analysis-results/question-7/`:

- `duplicate-question-close-times.csv`: per-question creation time, closure time,
  and elapsed time.
- `close-time-distribution.csv`: counts for each duration range.
- `time-to-close-distribution.png`: bar chart of the duration distribution.

## Run

From the project root:

```bash
python analysis/question-7.py
```

## Limitations

- Only questions present in `posts_short.csv` can be analysed. If it is a sample,
  results describe that sample rather than all Stack Overflow questions.
- A question must have a matching `post_history` closure event and a duplicate link
  to be included.
- This uses the first recorded close event. If questions were reopened and later
  closed again, this measures time until the first closure, not necessarily the
  final closure.
- The calculation assumes `post_history_type_id == 10` is represented in the
  supplied history file and that the dataset's timestamps use compatible formats.
