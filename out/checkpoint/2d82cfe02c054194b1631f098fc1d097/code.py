import os
import numpy as np
import pandas as pd

from lenskit.als import ImplicitMFScorer
from lenskit.batch import recommend as batch_recommend
from lenskit.data import from_interactions_df
from lenskit.pipeline import topn_pipeline


def load_movielens_100k(path):
    df = pd.read_csv(path, sep='\t', names=['user', 'item', 'rating', 'timestamp'])
    return df


def make_fixed_split(df, test_fraction=0.2, seed=42):
    rng = np.random.default_rng(seed)
    users = df['user'].unique()
    test_users = set(rng.choice(users, size=max(1, int(round(len(users) * test_fraction))), replace=False))
    train_df = df[~df['user'].isin(test_users)].copy()
    test_df = df[df['user'].isin(test_users)].copy()

    if test_df.empty:
        raise ValueError('Test split is empty; adjust test_fraction or check the input data.')
    return train_df, test_df


def _ndcg_at_k(rec_items, relevant, cutoff):
    hits = [1 if item in relevant else 0 for item in rec_items[:cutoff]]
    gains = np.asarray(hits, dtype=float)
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float(np.sum(gains * discounts))
    ideal_len = min(len(relevant), cutoff)
    ideal_gains = np.ones(ideal_len, dtype=float)
    ideal_discounts = 1.0 / np.log2(np.arange(2, ideal_len + 2))
    idcg = float(np.sum(ideal_gains * ideal_discounts)) if ideal_len > 0 else 0.0
    return dcg / idcg if idcg > 0 else 0.0


def _recall_at_k(rec_items, relevant, cutoff):
    hits = sum(1 for item in rec_items[:cutoff] if item in relevant)
    return float(hits / len(relevant)) if relevant else 0.0


def fit_and_recommend(train_df, test_df, seed, cutoff=10):
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
    for attr in ('seed', 'random_state', 'rng_seed'):
        if hasattr(model, attr):
            setattr(model, attr, seed)

    pipe = topn_pipeline(model, n=cutoff)
    pipe.train(train_ds)

    test_users = test_df['user'].unique().tolist()
    recs = batch_recommend(pipe, test_users, n=cutoff)

    ndcg_values = []
    recall_values = []

    train_items_by_user = train_df.groupby('user')['item'].apply(set).to_dict()
    test_items_by_user = test_df.groupby('user')['item'].apply(set).to_dict()

    for user, rec_list in recs.items():
        uid = user.user_id if hasattr(user, 'user_id') else user
        rec_items = list(rec_list.ids()) if hasattr(rec_list, 'ids') else list(rec_list)
        relevant = test_items_by_user.get(uid, set())
        if not relevant:
            continue

        ndcg = _ndcg_at_k(rec_items, relevant, cutoff)
        rec = _recall_at_k(rec_items, relevant, cutoff)
        ndcg_values.append(float(ndcg))
        recall_values.append(float(rec))

    summary = {
        'seed': seed,
        'NDCG': float(np.mean(ndcg_values)) if ndcg_values else float('nan'),
        'Recall': float(np.mean(recall_values)) if recall_values else float('nan'),
        'num_users_evaluated': int(len(ndcg_values)),
    }
    return summary


def summarize_runs(run_summaries):
    df = pd.DataFrame(run_summaries)
    summary = {}
    for col in ['NDCG', 'Recall']:
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
    train_df, test_df = make_fixed_split(df, test_fraction=0.2, seed=42)

    seeds = [1, 2, 3, 4, 5]
    cutoff = 10
    run_summaries = []

    for seed in seeds:
        summary = fit_and_recommend(train_df, test_df, seed=seed, cutoff=cutoff)
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