import os
working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)

import json
import math
import platform
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

experiment_data = {
    'ml100k_u_data': {
        'metrics': {'train': [], 'val': []},
        'losses': {'train': [], 'val': []},
        'predictions': [],
        'ground_truth': [],
        'seeds': [],
        'timestamps': [],
        'metadata': {}
    }
}

try:
    import lenskit
    from lenskit.algorithms.als import ImplicitMF
    from lenskit.algorithms import Recommender
    from lenskit import batch
except Exception as e:
    raise RuntimeError(f'LensKit import failed: {e}')


def pkg_version(mod):
    return getattr(mod, '__version__', 'unknown')


def bootstrap_ci(x, n_boot=2000, alpha=0.05, seed=12345):
    x = np.asarray(x, dtype=float)
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError('bootstrap_ci received empty or non-finite data')
    rng = np.random.RandomState(seed)
    stats = np.empty(n_boot, dtype=float)
    n = len(x)
    for i in range(n_boot):
        sample = x[rng.randint(0, n, size=n)]
        stats[i] = sample.mean()
    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return lo, hi, stats


def validate_and_load(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required dataset file not found: {path}")
    df = pd.read_csv(path, sep='\t', header=None, engine='python')
    if df.shape[1] != 4:
        raise ValueError(f"Expected 4 tab-separated columns in u.data, found {df.shape[1]}")
    df.columns = ['user', 'item', 'rating', 'timestamp']
    if df.empty:
        raise ValueError('Loaded ratings dataframe is empty')
    if df[['user', 'item', 'rating']].isnull().any().any():
        raise ValueError('Found nulls in required columns user/item/rating')
    df['user'] = df['user'].astype(int)
    df['item'] = df['item'].astype(int)
    df['rating'] = df['rating'].astype(float)
    df['timestamp'] = df['timestamp'].astype(int)
    if not np.isfinite(df['rating']).all():
        raise ValueError('Non-finite ratings detected')
    return df[['user', 'item', 'rating', 'timestamp']].copy()


def fixed_split(df, test_ratio=0.2, seed=2025):
    rng = np.random.RandomState(seed)
    mask = rng.rand(len(df)) < (1.0 - test_ratio)
    train = df.loc[mask, ['user', 'item', 'rating']].copy()
    test = df.loc[~mask, ['user', 'item', 'rating']].copy()
    if train.empty or test.empty:
        raise ValueError('Train or test split is empty')
    train_users = set(train['user'].unique())
    train_items = set(train['item'].unique())
    test = test[test['user'].isin(train_users) & test['item'].isin(train_items)].copy()
    if test.empty:
        raise ValueError('Test set became empty after filtering to seen users/items')
    user_counts = test.groupby('user').size()
    valid_users = user_counts[user_counts > 0].index
    test = test[test['user'].isin(valid_users)].copy()
    if test['user'].nunique() == 0:
        raise ValueError('No evaluable test users remain after filtering')
    return train.reset_index(drop=True), test.reset_index(drop=True)


def evaluate_user(truth_items, pred_items, k=20):
    truth_set = set(truth_items)
    pred = list(pred_items)[:k]
    if len(truth_set) == 0:
        return None, None
    hits = np.array([1.0 if i in truth_set else 0.0 for i in pred], dtype=float)
    denom = np.log2(np.arange(2, len(pred) + 2))
    dcg = float((hits / denom).sum()) if len(pred) else 0.0
    idcg_len = min(len(truth_set), len(pred))
    idcg = float((np.ones(idcg_len) / np.log2(np.arange(2, idcg_len + 2))).sum()) if idcg_len > 0 else 0.0
    ndcg = dcg / idcg if idcg > 0 else 0.0
    recall = float(hits.sum() / len(truth_set))
    return recall, ndcg


path = os.path.join(os.getcwd(), 'u.data')
ratings = validate_and_load(path)
train, test = fixed_split(ratings, test_ratio=0.2, seed=2025)
truth_df = test.groupby('user')['item'].apply(list).reset_index(name='items_truth')
test_users = truth_df['user'].tolist()
if len(test_users) == 0:
    raise ValueError('No test users available for recommendation')

metadata = {
    'dataset_path': path,
    'created_at': datetime.utcnow().isoformat() + 'Z',
    'python_version': platform.python_version(),
    'platform': platform.platform(),
    'numpy_version': pkg_version(np),
    'pandas_version': pkg_version(pd),
    'matplotlib_version': pkg_version(plt.matplotlib),
    'lenskit_version': pkg_version(lenskit),
    'split_seed': 2025,
    'test_ratio': 0.2,
    'model': {
        'name': 'ImplicitMF',
        'features': 20,
        'iterations': 15,
        'weight': 40,
        'method': 'cg'
    },
    'evaluation': {
        'topn_k': 20,
        'candidate_protocol': 'LensKit batch.recommend on fixed evaluable test users with n_jobs=1; recommender excludes training history by standard recommend protocol.',
        'ground_truth': 'All held-out test items per user after filtering to users/items seen in train.',
        'metrics': ['Recall@20', 'NDCG@20'],
        'validation_loss_proxy': '1 - mean(NDCG@20)'
    },
    'split_info': {
        'n_ratings': int(len(ratings)),
        'n_train': int(len(train)),
        'n_test': int(len(test)),
        'n_users_total': int(ratings['user'].nunique()),
        'n_items_total': int(ratings['item'].nunique()),
        'n_train_users': int(train['user'].nunique()),
        'n_test_users': int(test['user'].nunique())
    }
}
experiment_data['ml100k_u_data']['metadata'] = metadata

seeds = [1, 7, 21, 42, 84, 123, 256, 512, 1024, 2024]
results = []
all_recalls = []
all_ndcgs = []

for epoch, seed in enumerate(seeds, start=1):
    algo = Recommender.adapt(ImplicitMF(features=20, iterations=15, weight=40, method='cg', rng_spec=seed))
    algo.fit(train)
    recs = batch.recommend(algo, test_users, 20, n_jobs=1)
    if recs.empty:
        raise RuntimeError(f'No recommendations produced for seed {seed}')
    rec_items = recs.groupby('user')['item'].apply(list).reset_index(name='items_pred')
    eval_df = pd.merge(truth_df, rec_items, on='user', how='inner')
    if eval_df.empty:
        raise RuntimeError(f'Empty evaluation join for seed {seed}')

    recalls, ndcgs = [], []
    for row in eval_df.itertuples(index=False):
        recall, ndcg = evaluate_user(row.items_truth, row.items_pred, k=20)
        if recall is not None:
            recalls.append(recall)
            ndcgs.append(ndcg)
    recalls = np.asarray(recalls, dtype=float)
    ndcgs = np.asarray(ndcgs, dtype=float)
    if recalls.size == 0 or ndcgs.size == 0:
        raise RuntimeError(f'No valid per-user metrics for seed {seed}')
    if not np.isfinite(recalls).all() or not np.isfinite(ndcgs).all():
        raise RuntimeError(f'Non-finite metrics encountered for seed {seed}')

    mean_recall = float(recalls.mean())
    mean_ndcg = float(ndcgs.mean())
    val_loss = float(1.0 - mean_ndcg)
    print(f'Epoch {epoch}: validation_loss = {val_loss:.4f}')
    print(f'Seed {seed}: Recall@20 = {mean_recall:.4f}, NDCG@20 = {mean_ndcg:.4f}')

    timestamp = datetime.utcnow().isoformat() + 'Z'
    results.append({
        'seed': seed,
        'recall@20': mean_recall,
        'ndcg@20': mean_ndcg,
        'validation_loss': val_loss,
        'n_train': int(len(train)),
        'n_test': int(len(test)),
        'n_eval_users': int(len(eval_df))
    })
    all_recalls.append(recalls)
    all_ndcgs.append(ndcgs)
    experiment_data['ml100k_u_data']['metrics']['val'].append({
        'epoch': epoch,
        'seed': seed,
        'recall@20': mean_recall,
        'ndcg@20': mean_ndcg,
        'timestamp': timestamp
    })
    experiment_data['ml100k_u_data']['losses']['val'].append({
        'epoch': epoch,
        'seed': seed,
        'validation_loss': val_loss,
        'timestamp': timestamp
    })
    experiment_data['ml100k_u_data']['predictions'].append(recs[['user', 'item', 'rank']].to_records(index=False))
    experiment_data['ml100k_u_data']['ground_truth'].append(test[['user', 'item']].to_records(index=False))
    experiment_data['ml100k_u_data']['seeds'].append(seed)
    experiment_data['ml100k_u_data']['timestamps'].append(timestamp)

results_df = pd.DataFrame(results)
if results_df.empty:
    raise RuntimeError('No seed results collected')
if results_df[['recall@20', 'ndcg@20', 'validation_loss']].isnull().any().any():
    raise RuntimeError('NaN detected in aggregated results')

recall_vals = results_df['recall@20'].to_numpy(dtype=float)
ndcg_vals = results_df['ndcg@20'].to_numpy(dtype=float)
loss_vals = results_df['validation_loss'].to_numpy(dtype=float)
seed_vals = results_df['seed'].to_numpy(dtype=int)

recall_ci_lo, recall_ci_hi, recall_boot = bootstrap_ci(recall_vals)
ndcg_ci_lo, ndcg_ci_hi, ndcg_boot = bootstrap_ci(ndcg_vals)

summary_rows = []
for name, vals in [('Recall@20', recall_vals), ('NDCG@20', ndcg_vals), ('validation_loss', loss_vals)]:
    mean = float(vals.mean())
    std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    vmin = float(vals.min())
    vmax = float(vals.max())
    cv = float(std / mean) if mean != 0 else float('nan')
    rel_range = float((vmax - vmin) / mean) if mean != 0 else float('nan')
    summary_rows.append({'metric': name, 'mean': mean, 'std': std, 'min': vmin, 'max': vmax, 'cv': cv, 'relative_range': rel_range})
summary_df = pd.DataFrame(summary_rows)

ci_df = pd.DataFrame([
    {'metric': 'Recall@20', 'ci95_low': recall_ci_lo, 'ci95_high': recall_ci_hi},
    {'metric': 'NDCG@20', 'ci95_low': ndcg_ci_lo, 'ci95_high': ndcg_ci_hi}
])

print('\nPer-seed results:')
print(results_df.to_string(index=False))
print('\nSummary across seeds:')
print(summary_df.to_string(index=False))
print('\n95% bootstrap CI for mean metrics across seeds:')
print(ci_df.to_string(index=False))

np.save(os.path.join(working_dir, 'ml100k_seed_results.npy'), results_df.to_records(index=False))
np.save(os.path.join(working_dir, 'ml100k_recall_at20.npy'), recall_vals)
np.save(os.path.join(working_dir, 'ml100k_ndcg_at20.npy'), ndcg_vals)
np.save(os.path.join(working_dir, 'ml100k_validation_loss.npy'), loss_vals)
np.save(os.path.join(working_dir, 'ml100k_recall_bootstrap_means.npy'), recall_boot)
np.save(os.path.join(working_dir, 'ml100k_ndcg_bootstrap_means.npy'), ndcg_boot)
np.save(os.path.join(working_dir, 'ml100k_test_truth.npy'), truth_df.to_records(index=False))
np.save(os.path.join(working_dir, 'experiment_data.npy'), experiment_data, allow_pickle=True)
np.savez_compressed(
    os.path.join(working_dir, 'ml100k_seed_experiment_arrays.npz'),
    seeds=seed_vals,
    recall_at20=recall_vals,
    ndcg_at20=ndcg_vals,
    validation_loss=loss_vals
)
with open(os.path.join(working_dir, 'ml100k_metadata.json'), 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)

plt.figure(figsize=(10, 4))
plt.plot(seed_vals, recall_vals, marker='o', label='Recall@20')
plt.plot(seed_vals, ndcg_vals, marker='s', label='NDCG@20')
plt.xlabel('Random seed')
plt.ylabel('Metric value')
plt.title('MovieLens100K ImplicitMF metrics across seeds (fixed split)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(working_dir, 'ml100k_metrics_by_seed.png'), dpi=150)
plt.close()

plt.figure(figsize=(8, 4))
plt.boxplot([recall_vals, ndcg_vals], tick_labels=['Recall@20', 'NDCG@20'])
plt.ylabel('Metric value')
plt.title('Distribution of metrics across seeds')
plt.tight_layout()
plt.savefig(os.path.join(working_dir, 'ml100k_metric_distribution.png'), dpi=150)
plt.close()

plt.figure(figsize=(8, 4))
plt.plot(range(1, len(loss_vals) + 1), loss_vals, marker='d')
plt.xticks(range(1, len(loss_vals) + 1), seed_vals, rotation=45)
plt.xlabel('Seed')
plt.ylabel('Validation loss (1 - NDCG@20)')
plt.title('Validation loss across seeds')
plt.tight_layout()
plt.savefig(os.path.join(working_dir, 'ml100k_validation_loss_by_seed.png'), dpi=150)
plt.close()

ndcg_rel_range = float((ndcg_vals.max() - ndcg_vals.min()) / ndcg_vals.mean())
recall_rel_range = float((recall_vals.max() - recall_vals.min()) / recall_vals.mean())
if ndcg_rel_range >= 0.05 or recall_rel_range >= 0.05:
    conclusion = 'Conclusion: random seed changes produce noticeable metric variation on a fixed split, so reporting seeds is important.'
else:
    conclusion = 'Conclusion: random seed changes produce modest variation on this fixed split, but reporting seeds remains good practice for reproducibility.'
print('\n' + conclusion)
print(f'Relative spread: Recall@20={(recall_vals.max()-recall_vals.min()):.4f} ({recall_rel_range*100:.2f}% of mean), NDCG@20={(ndcg_vals.max()-ndcg_vals.min()):.4f} ({ndcg_rel_range*100:.2f}% of mean)')