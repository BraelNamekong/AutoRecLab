import os
import json
import numpy as np
import pandas as pd


def make_working_dir():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)
    return working_dir


def load_movielens_u_data(path):
    df = pd.read_csv(
        path,
        sep='\t',
        header=None,
        names=['user', 'item', 'rating', 'timestamp'],
    )
    return df


def train_test_split_interactions(df, test_frac=0.2, seed=42):
    rng = np.random.default_rng(seed)
    test_mask = np.zeros(len(df), dtype=bool)
    for user, idx in df.groupby('user').indices.items():
        idx = np.asarray(list(idx))
        n_test = max(1, int(round(len(idx) * test_frac)))
        chosen = rng.choice(idx, size=n_test, replace=False)
        test_mask[chosen] = True
    test_df = df.loc[test_mask].copy().reset_index(drop=True)
    train_df = df.loc[~test_mask].copy().reset_index(drop=True)
    return train_df, test_df


def build_dataset(df):
    from lenskit.data import from_interactions_df
    return from_interactions_df(df)


def fit_implicit_mf(train_ds, seed, embedding_size=32, epochs=20, batch_size=8192, learning_rate=0.05):
    from lenskit.als import ImplicitMFScorer
    model = ImplicitMFScorer(
        embedding_size=embedding_size,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        random_state=seed,
    )
    model.train(train_ds)
    return model


def recommend_for_users(model, train_ds, users, n):
    from lenskit.batch import recommend
    recs = recommend(model, users, n=n)
    return recs


def evaluate_run(recs, test_df, cutoff):
    from lenskit.metrics import MeasurementCollector
    from lenskit.metrics.ranking import NDCG, Recall

    mc = MeasurementCollector()
    mc.add_metric(NDCG(n=cutoff))
    mc.add_metric(Recall(n=cutoff))
    test_items = test_df.groupby('user')['item'].apply(list).to_dict()
    test_lists = []
    for user in recs['user'].unique():
        items = test_items.get(user, [])
        test_lists.append((user, items))
    return mc


def main():
    working_dir = make_working_dir()
    data_path = r'C:\Users\nicol\AutoRecLab\workspace\u.data'
    df = load_movielens_u_data(data_path)

    train_df, test_df = train_test_split_interactions(df, test_frac=0.2, seed=42)
    train_ds = build_dataset(train_df)
    test_users = sorted(test_df['user'].unique())

    seeds = [1, 7, 21, 42, 84]
    cutoff = 10
    run_rows = []

    for seed in seeds:
        model = fit_implicit_mf(train_ds, seed=seed)
        recs = recommend_for_users(model, train_ds, test_users, n=cutoff)

        # Per-user evaluation using the recommendation dataframe and held-out items.
        # This keeps the evaluation aligned across seeds while the train/test split stays fixed.
        recs_df = recs.to_df() if hasattr(recs, 'to_df') else pd.DataFrame(recs)
        merged = recs_df.merge(test_df[['user', 'item']], on=['user', 'item'], how='left', indicator=True)
        hit_rate_proxy = (merged['_merge'] == 'both').groupby(merged['user']).mean()

        # Compute ranking metrics by comparing top-N lists to held-out user items.
        # Use a lightweight per-user aggregation for reporting variability across seeds.
        user_groups = recs_df.groupby('user')['item'].apply(list).to_dict()
        ndcg_scores = []
        recall_scores = []
        for user in test_users:
            pred_items = user_groups.get(user, [])[:cutoff]
            true_items = set(test_df.loc[test_df['user'] == user, 'item'].tolist())
            if not true_items:
                continue
            hits = len(set(pred_items) & true_items)
            recall_scores.append(hits / len(true_items))
            # Simple DCG-based NDCG proxy using binary relevance for the held-out items.
            gains = [1.0 if item in true_items else 0.0 for item in pred_items]
            dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
            ideal_len = min(len(true_items), cutoff)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_len))
            ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

        run_rows.append({
            'seed': seed,
            f'ndcg@{cutoff}': float(np.mean(ndcg_scores)) if ndcg_scores else np.nan,
            f'recall@{cutoff}': float(np.mean(recall_scores)) if recall_scores else np.nan,
            'n_eval_users': len(ndcg_scores),
        })
        print(f'Seed {seed}: NDCG@{cutoff}={run_rows[-1][f"ndcg@{cutoff}"]:.6f}, Recall@{cutoff}={run_rows[-1][f"recall@{cutoff}"]:.6f}')

    runs = pd.DataFrame(run_rows)
    summary = runs.drop(columns=['seed', 'n_eval_users']).agg(['mean', 'std', 'min', 'max'])
    summary.loc['range'] = summary.loc['max'] - summary.loc['min']

    print('\nPer-run results:')
    print(runs.to_string(index=False))
    print('\nCross-seed summary:')
    print(summary.to_string())

    runs.to_csv(os.path.join(working_dir, 'seed_variability_runs.csv'), index=False)
    summary.to_csv(os.path.join(working_dir, 'seed_variability_summary.csv'))
    with open(os.path.join(working_dir, 'seed_variability_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary.reset_index().to_dict(orient='records'), f, indent=2)


if __name__ == '__main__':
    main()