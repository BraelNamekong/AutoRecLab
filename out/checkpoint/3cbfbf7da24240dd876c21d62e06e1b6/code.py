import os
from pathlib import Path

import numpy as np
import pandas as pd

from omnirec.metrics.ranking import NDCG, Recall
from lenskit.als import ImplicitMFScorer
from lenskit.batch import recommend as batch_recommend
from lenskit.data import from_interactions_df
from lenskit.pipeline import topn_pipeline
from lenskit.splitting import crossfold_records


DATA_PATH = r'C:\Users\nicol\AutoRecLab\workspace\u.data'
WORKING_DIR = os.path.join(os.getcwd(), 'working')
os.makedirs(WORKING_DIR, exist_ok=True)


def load_movielens_100k(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep='\t',
        header=None,
        names=['user', 'item', 'rating', 'timestamp'],
        engine='python',
    )
    df['user'] = df['user'].astype(int)
    df['item'] = df['item'].astype(int)
    df['rating'] = df['rating'].astype(float)
    df['timestamp'] = df['timestamp'].astype(int)
    return df


def make_split(dataset, seed: int):
    # Deterministic fixed protocol shared by all model seeds.
    # crossfold_records gives a reproducible record-level split when the RNG is fixed.
    splits = list(crossfold_records(dataset, 5, rng=seed))
    return splits[0]


def evaluate_run(train_ds, test_split, seed: int, n_recs: int = 10):
    model = ImplicitMFScorer(features=64, epochs=10, regularization=0.1, rng=seed)
    pipe = topn_pipeline(model, n=n_recs)
    pipe.train(train_ds)

    users = list(test_split.test.keys())
    recs = batch_recommend(pipe, users, n=n_recs, n_jobs=1)

    recs_df = pd.DataFrame(recs)

    ndcg_metric = NDCG(k=n_recs)
    recall_metric = Recall(k=n_recs)

    per_user = []
    for key, test_items in test_split.test.items():
        user_recs = recs_df[recs_df['user'] == key]
        if user_recs.empty:
            continue
        test_df = pd.DataFrame({'user': [key] * len(test_items), 'item': list(test_items)})
        ndcg = ndcg_metric.calculate(user_recs, test_df)
        recall = recall_metric.calculate(user_recs, test_df)
        per_user.append((ndcg, recall))

    per_user = np.asarray(per_user, dtype=float)
    ndcg_vals = per_user[:, 0]
    recall_vals = per_user[:, 1]

    return {
        'seed': seed,
        'ndcg_mean': float(np.nanmean(ndcg_vals)),
        'recall_mean': float(np.nanmean(recall_vals)),
        'n_users_eval': int(len(per_user)),
    }


def summarize_runs(results: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    summary = []
    for metric in ['ndcg_mean', 'recall_mean']:
        vals = df[metric].to_numpy(dtype=float)
        summary.append(
            {
                'metric': metric,
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                'min': float(np.min(vals)),
                'max': float(np.max(vals)),
                'range': float(np.max(vals) - np.min(vals)),
            }
        )
    return pd.DataFrame(summary)


def main():
    df = load_movielens_100k(DATA_PATH)
    data = from_interactions_df(df)

    fixed_split = make_split(data, seed=42)

    seeds = [1, 2, 3, 4, 5, 11, 22, 33, 44, 55]
    results = []
    for seed in seeds:
        run_res = evaluate_run(fixed_split.train, fixed_split, seed=seed, n_recs=10)
        results.append(run_res)
        print(f"seed={seed} ndcg={run_res['ndcg_mean']:.6f} recall={run_res['recall_mean']:.6f} users={run_res['n_users_eval']}")

    run_df = pd.DataFrame(results)
    summary_df = summarize_runs(results)

    print('\nPer-run results:')
    print(run_df.to_string(index=False))
    print('\nAcross-seed summary:')
    print(summary_df.to_string(index=False))

    run_df.to_csv(os.path.join(WORKING_DIR, 'seed_sensitivity_runs.csv'), index=False)
    summary_df.to_csv(os.path.join(WORKING_DIR, 'seed_sensitivity_summary.csv'), index=False)


if __name__ == '__main__':
    main()