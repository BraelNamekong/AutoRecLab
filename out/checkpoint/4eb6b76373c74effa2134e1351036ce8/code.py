import os
from typing import Any

import numpy as np
import pandas as pd

from omnirec.metrics.ranking import NDCG, Recall

ImplicitMFScorer = Any
batch_recommend = None
topn_pipeline = None
from_interactions_df = None
sample_users = None
SampleN = None
ItemList = None

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
    return sample_users(dataset, 200, SampleN(5), rng=seed)


def evaluate_run(train_ds, test_split, seed: int, n_recs: int = 10):
    model = ImplicitMFScorer(features=64, epochs=10, regularization=0.1, rng=seed)
    pipe = topn_pipeline(model, n=n_recs)
    pipe.train(train_ds)

    users = [key.user_id for key in test_split.test.keys()]
    recs = batch_recommend(pipe, users, n=n_recs, n_jobs=1)

    rec_map = {key.user_id: items for key, items in recs.items()}

    ndcg_metric = NDCG(k=n_recs)
    recall_metric = Recall(k=n_recs)

    per_user = []
    for key, test_items in test_split.test.items():
        uid = key.user_id
        ranking = rec_map.get(uid)
        if ranking is None or len(ranking) == 0:
            continue

        test_ilist = ItemList(test_items)

        ndcg_res = ndcg_metric.calculate(ranking, test_ilist)
        recall_res = recall_metric.calculate(ranking, test_ilist)

        ndcg = float(ndcg_res.result)
        recall = float(recall_res.result)
        per_user.append((ndcg, recall))

    per_user = np.asarray(per_user, dtype=float)
    ndcg_vals = per_user[:, 0] if len(per_user) else np.array([])
    recall_vals = per_user[:, 1] if len(per_user) else np.array([])

    return {
        'seed': seed,
        'ndcg_mean': float(np.nanmean(ndcg_vals)) if len(ndcg_vals) else np.nan,
        'recall_mean': float(np.nanmean(recall_vals)) if len(recall_vals) else np.nan,
        'n_users_eval': int(len(per_user)),
    }


def summarize_runs(results: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    summary = []
    for metric in ['ndcg_mean', 'recall_mean']:
        vals = df[metric].to_numpy(dtype=float)
        vals = vals[~np.isnan(vals)]
        summary.append(
            {
                'metric': metric,
                'mean': float(np.mean(vals)) if len(vals) else np.nan,
                'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                'min': float(np.min(vals)) if len(vals) else np.nan,
                'max': float(np.max(vals)) if len(vals) else np.nan,
                'range': float(np.max(vals) - np.min(vals)) if len(vals) else np.nan,
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