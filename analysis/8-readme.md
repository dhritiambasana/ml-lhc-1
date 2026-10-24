# Question 8: Reputation of Users Who Asked Duplicate Questions

## Problem / issue being solved
This analysis investigates whether duplicate questions are associated mainly
with low-reputation users, or whether users across a wider range of reputations
also ask duplicate questions. The result can help assess whether guidance for
new users might be useful.

## Approach
1. Load questions from posts_short.csv and keep question posts only.
2. Use postLinks.csv rows with link_type_id == 3 to identify duplicate
   question IDs. The source post_id is treated as the duplicate question.
3. Read owner_user_id for those duplicate questions.
4. Join the authors to users.csv using owner_user_id == users.id to obtain
   each author's reputation.
5. Count each distinct user once, even if they asked multiple duplicate
   questions. Group users into reputation ranges and plot the number of users
   in each range as a bar chart.
6. Save the matched author data, grouped counts, and chart under
   analysis-results/question-8/.
Reputation categories
The script uses these ranges: 0–1, 2–10, 11–100, 101–200, 201–500,
501–1,000, 1,001–5,000, 5,001–10,000, and over 10,000.

These are practical bins chosen for this analysis, not official reputation
levels. Users without a usable owner_user_id, or without a matching
reputation record in users.csv, are excluded.

## Input files

Expected under data/dataset/:
- posts_short.csv
- postLinks.csv
- users.csv
The script reuses analysis/common-data.py, created for the earlier questions.

## Output files
Saved under analysis-results/question-8/:
- duplicate-authors-by-reputation.csv: one row per matched author, with
  reputation and category.
- reputation-distribution.csv: number of distinct users in each category.
- reputation-distribution.png: bar chart.
Run
From the project root:
python analysis/question-8.py

## Limitations
- Results only describe questions available in posts_short.csv; if it is a
  sample, the results describe that sample rather than all Stack Overflow.
- This chart counts distinct users, not duplicate questions. A user who asked
  several duplicate questions is counted once.
- The analysis is descriptive. It does not establish that low reputation
  causes duplicate questions. To compare duplicate-question tendency by
  reputation, the number of all users or all questions in each reputation
  category would also be needed as a denominator.