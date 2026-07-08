# Experiment Summary

## User Request

Quantify how random seeds affect recommender system results using LensKit’s simple ImplicitMF on MovieLens100K `u.data`, with a fixed train/test split and multiple runs using different random seeds. The goal was to see whether metrics like NDCG or Recall vary across runs and whether reporting seeds is important.

## What Was Run

- Loaded MovieLens100K directly from:
  `C:\Users\nicol\AutoRecLab\workspace\u.data`
- Parsed the file with pandas as tab-separated columns:
  `user`, `item`, `rating`, `timestamp`
- Created a fixed split by selecting 20% of users as test users with split seed `42`
  - Train interactions: 81,550
  - Test interactions: 18,450
  - Test users: 189
- Trained LensKit `ImplicitMFScorer(features=20, epochs=20, use_ratings=True)`
- Repeated training and recommendation with seeds `[1, 2, 3, 4, 5]`
- Evaluated at cutoff 10 using per-user NDCG and Recall, then averaged across users

## Key Results

| Seed | NDCG | Recall | Users Evaluated |
|---|---:|---:|---:|
| 1 | 0.0 | 0.0 | 189 |
| 2 | 0.0 | 0.0 | 189 |
| 3 | 0.0 | 0.0 | 189 |
| 4 | 0.0 | 0.0 | 189 |
| 5 | 0.0 | 0.0 | 189 |

Across-seed variability:

| Metric | Mean | Std | Min | Max | Range |
|---|---:|---:|---:|---:|---:|
| NDCG | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Recall | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## Limitations

- The reported metrics are all exactly zero, so this run does not show any measurable seed sensitivity.
- The output does not include the actual recommended items, so it is not possible to determine from the provided materials why NDCG and Recall were zero.
- Only five seeds were tested.
- The experiment used a user-level split, not a per-interaction split, which may affect interpretation, but the output does not provide additional diagnostics.

## Conclusion

In this experiment, changing the random seed did not change NDCG or Recall at all: both metrics were 0.0 for every run, with zero variance across seeds. Based on these results alone, seed effects were not observable here, so this output does not provide evidence that seed reporting is important for this specific setup. However, because the metrics are uniformly zero, the experiment is inconclusive about broader seed sensitivity in general.