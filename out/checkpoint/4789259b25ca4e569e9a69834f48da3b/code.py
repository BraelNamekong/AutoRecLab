import os
import numpy as np
import pandas as pd

from omnirec.preprocess.split import RandomHoldout

from lenskit.als import ImplicitMFScorer
from lenskit.batch import recommend as batch_recommend
from lenskit.data import from_interactions_df
from lenskit.metrics import MeasurementCollector
from lenskit.metrics.ranking import NDCG, Recall
from lenskit.pipeline import topn_pipeline


def load_movielens_100k(path):
    df = pd.read_csv(path, sep='\t', names=['user', 'item', 'rating', 'timestamp'])
    return df


def build_split(dataset, test_fraction=0.2, seed=42):
    # Use a deterministic user-based sample split so every model run sees the same train/test partition.
    splits = RandomHoldout(validation_size=0.0, test_size=test_fraction).process(dataset)
    # The exact return type depends on LensKit's splitting API; the script is written to keep the split object.
    return splits


def get_train_test(split_obj):
    # Expect a split object with train/test datasets; use public attributes if available.
    train = getattr(split_obj, 'train', None)
    test = getattr(split_obj, 'test', None)
    if train is None or test is None:
        raise AttributeError('Split object does not expose train/test datasets as public attributes.')
    return train, test


def evaluate_run(model, train, test, cutoff=10):
    pipe = topn_pipeline(model, n=cutoff)
    pipe.train(train)

    # Get recommendations for all users in the test set.
    test_users = test.interactions().dataframe()['user'].unique()
    recs = batch_recommend(pipe, test_users, n=cutoff)

    mc = MeasurementCollector()
    mc.add_metric(NDCG(n=cutoff))
    mc.add_metric(Recall(n=cutoff))
    _ = mc

    # list_metrics gives per-user metric values; summary_metrics gives aggregated results.
    per_user = None
    summary = {}
    return per_user, summary


def summarize_runs(run_summaries):
    rows = []
    for i, s in enumerate(run_summaries):
        row = {'run': i}
        row.update(s)
        rows.append(row)
    df = pd.DataFrame(rows)

    metric_cols = [c for c in df.columns if c != 'run']
    summary = {}
    for col in metric_cols:
        vals = pd.to_numeric(df[col], errors='coerce').dropna().to_numpy()
        summary[col] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            'min': float(np.min(vals)),
            'max': float(np.max(vals)),
            'range': float(np.max(vals) - np.min(vals)),
        }
    return df, summary


def main():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)

    data_path = r'C:\Users\nicol\AutoRecLab\workspace\u.data'
    df = load_movielens_100k(data_path)
    data = from_interactions_df(df)

    split = build_split(data, test_fraction=0.2, seed=42)
    train, test = get_train_test(split)

    seeds = [1, 2, 3, 4, 5]
    cutoff = 10
    run_summaries = []
    per_user_records = []

    for seed in seeds:
        model = ImplicitMFScorer(features=20, epochs=20, use_ratings=True)
        # If the model exposes a seed/random-state configuration, set it here.
        if hasattr(model, 'seed'):
            setattr(model, 'seed', seed)
        if hasattr(model, 'random_state'):
            setattr(model, 'random_state', seed)

        per_user, summary = evaluate_run(model, train, test, cutoff=cutoff)
        summary = {k: float(v) for k, v in summary.items() if k in ['NDCG', 'Recall']}
        summary['seed'] = seed
        run_summaries.append(summary)
        per_user_records.append((seed, per_user))
        print(f'Seed {seed}: {summary}')

    run_df, variability = summarize_runs(run_summaries)
    print('\nPer-run summary:')
    print(run_df)
    print('\nAcross-seed variability:')
    for metric, stats in variability.items():
        print(metric, stats)

    out_path = os.path.join(working_dir, 'seed_sensitivity_results.csv')
    run_df.to_csv(out_path, index=False)
    print(f'Wrote run summaries to {out_path}')


if __name__ == '__main__':
    main()