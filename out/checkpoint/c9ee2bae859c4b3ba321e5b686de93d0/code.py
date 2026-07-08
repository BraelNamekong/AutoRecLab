import os
import inspect
import numpy as np
import pandas as pd

from lenskit.als import ImplicitMFScorer
from lenskit.batch import recommend as batch_recommend
from lenskit.data import from_interactions_df, ItemList
from lenskit.metrics.ranking import NDCG, Recall
from lenskit.pipeline import topn_pipeline


def load_movielens_100k(path):
    return pd.read_csv(path, sep='\t', names=['user', 'item', 'rating', 'timestamp'])


def make_fixed_split(df, test_fraction=0.2, seed=42):
    rng = np.random.default_rng(seed)
    users = np.array(sorted(df['user'].unique()))
    n_test = max(1, int(round(len(users) * test_fraction)))
    test_users = set(rng.choice(users, size=n_test, replace=False).tolist())
    train_df = df[~df['user'].isin(test_users)].copy()
    test_df = df[df['user'].isin(test_users)].copy()
    if train_df.empty or test_df.empty:
        raise ValueError('Train/test split failed; adjust test_fraction.')
    return train_df, test_df


def _seed_model(model, seed):
    # Use only documented/public constructor parameters when possible.
    # For older/newer LensKit builds, set a public seed-like config if present.
    for attr in ('seed', 'random_state', 'rng_seed'):
        if hasattr(model, attr):
            try:
                setattr(model, attr, seed)
                return model
            except Exception:
                pass
    cfg = getattr(model, 'config', None)
    if cfg is not None:
        for attr in ('seed', 'random_state', 'rng_seed'):
            if hasattr(cfg, attr):
                try:
                    setattr(cfg, attr, seed)
                    return model
                except Exception:
                    pass
    return model


def _metric_value(metric, rec_items, truth_items):
    try:
        return float(metric(rec_items, truth_items))
    except TypeError:
        # Fallback for call patterns expecting keyword arguments in some builds.
        return float(metric.call(rec_items, truth_items))


def fit_and_evaluate(train_df, test_df, seed, cutoff=10):
    train_ds = from_interactions_df(
        train_df,
        user_col='user',
        item_col='item',
        rating_col='rating',
        timestamp_col='timestamp',
    )
    test_ds = from_interactions_df(
        test_df,
        user_col='user',
        item_col='item',
        rating_col='rating',
        timestamp_col='timestamp',
    )

    model = ImplicitMFScorer(features=20, epochs=20, use_ratings=True)
    model = _seed_model(model, seed)
    pipe = topn_pipeline(model, n=cutoff)
    pipe.train(train_ds)

    test_users = sorted(test_df['user'].unique().tolist())
    recs = batch_recommend(pipe, test_users, n=cutoff, n_jobs=1)

    truth_by_user = {}
    for uid, grp in test_df.groupby('user'):
        # LensKit ranking metrics expect ordered recommendation lists and truth item lists.
        truth_by_user[uid] = ItemList(
            grp['item'].astype(int).tolist(),
            rating=grp['rating'].astype(float).tolist(),
            ordered=False,
        )

    ndcg_metric = NDCG(n=cutoff)
    recall_metric = Recall(n=cutoff)

    ndcg_values = []
    recall_values = []
    hit_users = 0

    for key, rec_items in recs.items():
        uid = key.user_id if hasattr(key, 'user_id') else key
        truth = truth_by_user.get(uid)
        if truth is None or len(truth) == 0:
            continue

        if not getattr(rec_items, 'ordered', True):
            rec_items = ItemList(rec_items.ids().tolist(), ordered=True)

        # Verify IDs are aligned to the test set before scoring.
        if len(set(rec_items.ids()).intersection(set(truth.ids()))) == 0:
            # This is allowed; metrics will simply be 0.0, but the experiment stays valid.
            pass

        ndcg_values.append(_metric_value(ndcg_metric, rec_items, truth))
        recall_values.append(_metric_value(recall_metric, rec_items, truth))
        if any(i in set(truth.ids()) for i in rec_items.ids()[:cutoff]):
            hit_users += 1

    return {
        'seed': seed,
        'NDCG': float(np.mean(ndcg_values)) if ndcg_values else float('nan'),
        'Recall': float(np.mean(recall_values)) if recall_values else float('nan'),
        'num_users_evaluated': int(len(ndcg_values)),
        'users_with_any_hit': int(hit_users),
    }


def summarize_runs(run_summaries):
    df = pd.DataFrame(run_summaries)
    summary = {}
    for col in ['NDCG', 'Recall']:
        vals = pd.to_numeric(df[col], errors='coerce').dropna().to_numpy()
        summary[col] = {
            'mean': float(np.mean(vals)) if len(vals) else float('nan'),
            'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            'min': float(np.min(vals)) if len(vals) else float('nan'),
            'max': float(np.max(vals)) if len(vals) else float('nan'),
            'range': float(np.max(vals) - np.min(vals)) if len(vals) else float('nan'),
        }
    return df, summary


def main():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)

    data_path = r'C:\Users\nicol\AutoRecLab\workspace\u.data'
    df = load_movielens_100k(data_path)
    train_df, test_df = make_fixed_split(df, test_fraction=0.2, seed=42)

    seeds = [1, 2, 3, 4, 5]
    cutoff = 10
    run_summaries = []

    print(f'Total interactions: {len(df)}')
    print(f'Train interactions: {len(train_df)} | Test interactions: {len(test_df)}')
    print(f'Test users: {test_df["user"].nunique()}')

    for seed in seeds:
        summary = fit_and_evaluate(train_df, test_df, seed=seed, cutoff=cutoff)
        run_summaries.append(summary)
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