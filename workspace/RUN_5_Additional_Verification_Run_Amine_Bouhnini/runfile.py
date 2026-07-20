import os
working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)

import json
import numpy as np
import pandas as pd
from lenskit.algorithms.als import ImplicitMF

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception as e:
    HAVE_MPL = False
    plt = None
    print(f'matplotlib unavailable, skipping plots: {e}')

experiment_data = {
    'ml100k_u_data': {
        'metrics': {'train': [], 'val': []},
        'losses': {'train': [], 'val': []},
        'predictions': [],
        'ground_truth': [],
        'timestamps': [],
        'seeds': [],
        'summary': {}
    }
}

data_path = os.path.join(os.getcwd(), 'u.data')
if not os.path.exists(data_path):
    raise FileNotFoundError(f'Could not find data file: {data_path}')

cols = ['user', 'item', 'rating', 'timestamp']
ratings = pd.read_csv(data_path, sep='\t', names=cols, engine='python')
ratings['user'] = ratings['user'].astype(int)
ratings['item'] = ratings['item'].astype(int)
ratings['rating'] = ratings['rating'].astype(float)
ratings['value'] = 1.0

split_rng = np.random.RandomState(2024)
mask = split_rng.rand(len(ratings)) < 0.8
train = ratings.loc[mask, ['user', 'item', 'value']].copy()
test = ratings.loc[~mask, ['user', 'item', 'value']].copy()

train_users = set(train['user'].unique())
train_items = set(train['item'].unique())
test = test[test['user'].isin(train_users) & test['item'].isin(train_items)].copy()

train_items_by_user = train.groupby('user')['item'].apply(set).to_dict()
test_items_by_user = test.groupby('user')['item'].apply(set).to_dict()
eval_users = sorted(test_items_by_user.keys())
all_items = np.array(sorted(train['item'].unique()), dtype=int)

def dcg_at_k(rels, k):
    rels = np.asarray(rels, dtype=float)[:k]
    if rels.size == 0:
        return 0.0
    denom = np.log2(np.arange(2, rels.size + 2))
    return float((rels / denom).sum())

def ndcg_at_k(rec_items, truth, k=10):
    rels = [1.0 if i in truth else 0.0 for i in rec_items[:k]]
    dcg = dcg_at_k(rels, k)
    ideal = dcg_at_k([1.0] * min(len(truth), k), k)
    return 0.0 if ideal == 0 else dcg / ideal

def recall_at_k(rec_items, truth, k=10):
    return 0.0 if len(truth) == 0 else len(set(rec_items[:k]) & truth) / len(truth)

def score_items(algo, user, items):
    scores = algo.predict_for_user(user, items)
    if isinstance(scores, pd.Series):
        scores = scores.reindex(items).to_numpy(dtype=float)
    else:
        scores = np.asarray(scores, dtype=float)
    return scores

def recommend_topk(algo, user, k=10):
    seen = train_items_by_user.get(user, set())
    cands = np.array([i for i in all_items if i not in seen], dtype=int)
    if cands.size == 0:
        return []
    vals = score_items(algo, user, cands)
    good = np.isfinite(vals)
    cands, vals = cands[good], vals[good]
    if vals.size == 0:
        return []
    order = np.argsort(-vals)[:k]
    return list(cands[order])

seeds = [1, 7, 21, 42, 84, 126, 168, 210, 252, 294]
recalls, ndcgs, val_losses = [], [], []

for run_idx, seed in enumerate(seeds, start=1):
    algo = ImplicitMF(features=20, iterations=15, reg=0.1, weight=40, method='cg', rng_spec=seed)
    algo.fit(train[['user', 'item', 'value']])

    user_recalls, user_ndcgs = [], []
    all_scores, all_labels = [], []
    sample_preds, sample_truth = [], []

    for user in eval_users:
        truth = test_items_by_user[user]
        recs = recommend_topk(algo, user, k=10)
        user_recalls.append(recall_at_k(recs, truth, 10))
        user_ndcgs.append(ndcg_at_k(recs, truth, 10))
        sample_preds.append(recs)
        sample_truth.append(sorted(truth))

        seen = train_items_by_user.get(user, set())
        cands = [i for i in all_items if i not in seen]
        pos = [i for i in cands if i in truth]
        neg = [i for i in cands if i not in truth]
        if not pos:
            continue
        rng = np.random.RandomState(seed + int(user))
        neg_sample = rng.choice(neg, size=min(len(neg), max(1, len(pos) * 5)), replace=False) if neg else np.array([], dtype=int)
        eval_items = np.array(pos + list(neg_sample), dtype=int)
        labels = np.array([1] * len(pos) + [0] * len(neg_sample), dtype=float)
        scores = score_items(algo, user, eval_items)
        good = np.isfinite(scores)
        scores, labels = scores[good], labels[good]
        if scores.size == 0:
            continue
        probs = 1.0 / (1.0 + np.exp(-scores))
        probs = np.clip(probs, 1e-8, 1 - 1e-8)
        all_scores.extend(probs.tolist())
        all_labels.extend(labels.tolist())

    mean_recall = float(np.mean(user_recalls)) if user_recalls else 0.0
    mean_ndcg = float(np.mean(user_ndcgs)) if user_ndcgs else 0.0
    if all_scores:
        scores_arr = np.clip(np.array(all_scores, dtype=float), 1e-8, 1 - 1e-8)
        labels_arr = np.array(all_labels, dtype=float)
        val_loss = float(np.mean(-(labels_arr * np.log(scores_arr) + (1 - labels_arr) * np.log(1 - scores_arr))))
    else:
        val_loss = float('nan')

    recalls.append(mean_recall)
    ndcgs.append(mean_ndcg)
    val_losses.append(val_loss)

    experiment_data['ml100k_u_data']['metrics']['val'].append({'seed': seed, 'recall@10': mean_recall, 'ndcg@10': mean_ndcg})
    experiment_data['ml100k_u_data']['losses']['val'].append({'seed': seed, 'validation_loss': val_loss})
    experiment_data['ml100k_u_data']['predictions'].append(sample_preds[:20])
    experiment_data['ml100k_u_data']['ground_truth'].append(sample_truth[:20])
    experiment_data['ml100k_u_data']['timestamps'].append(run_idx)
    experiment_data['ml100k_u_data']['seeds'].append(seed)

    print(f'Epoch {run_idx}: validation_loss = {val_loss:.4f}')
    print(f'Run seed={seed}: Recall@10 = {mean_recall:.4f}, NDCG@10 = {mean_ndcg:.4f}')

recalls = np.array(recalls, dtype=float)
ndcgs = np.array(ndcgs, dtype=float)
val_losses = np.array(val_losses, dtype=float)

summary = {
    'recall@10_mean': float(np.mean(recalls)),
    'recall@10_std': float(np.std(recalls)),
    'recall@10_min': float(np.min(recalls)),
    'recall@10_max': float(np.max(recalls)),
    'ndcg@10_mean': float(np.mean(ndcgs)),
    'ndcg@10_std': float(np.std(ndcgs)),
    'ndcg@10_min': float(np.min(ndcgs)),
    'ndcg@10_max': float(np.max(ndcgs)),
    'validation_loss_mean': float(np.nanmean(val_losses)),
    'validation_loss_std': float(np.nanstd(val_losses)),
    'recall@10_cv_pct': float(100 * np.std(recalls) / np.mean(recalls)) if np.mean(recalls) != 0 else 0.0,
    'ndcg@10_cv_pct': float(100 * np.std(ndcgs) / np.mean(ndcgs)) if np.mean(ndcgs) != 0 else 0.0
}
experiment_data['ml100k_u_data']['summary'] = summary

print('\nSeed sensitivity summary:')
for k, v in summary.items():
    print(f'{k}: {v:.6f}')

importance_msg = (
    'Reporting random seeds appears important.'
    if (summary['recall@10_cv_pct'] > 2.0 or summary['ndcg@10_cv_pct'] > 2.0)
    else 'Seed sensitivity appears modest for this setup, but reporting seeds is still good practice.'
)
print(f'Conclusion: {importance_msg}')

np.save(os.path.join(working_dir, 'ml100k_recalls.npy'), recalls)
np.save(os.path.join(working_dir, 'ml100k_ndcgs.npy'), ndcgs)
np.save(os.path.join(working_dir, 'ml100k_val_losses.npy'), val_losses)
np.save(os.path.join(working_dir, 'ml100k_seeds.npy'), np.array(seeds, dtype=int))
np.save(os.path.join(working_dir, 'experiment_data.npy'), experiment_data)
with open(os.path.join(working_dir, 'ml100k_seed_summary.json'), 'w') as f:
    json.dump({'summary': summary, 'conclusion': importance_msg}, f, indent=2)

if HAVE_MPL:
    x = np.arange(len(seeds))
    plt.figure(figsize=(10, 4))
    plt.plot(x, recalls, marker='o', label='Recall@10')
    plt.plot(x, ndcgs, marker='s', label='NDCG@10')
    plt.xticks(x, seeds, rotation=45)
    plt.xlabel('Random seed')
    plt.ylabel('Metric value')
    plt.title('MovieLens100K ImplicitMF metric variation across seeds')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, 'ml100k_seed_metrics.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.boxplot([recalls, ndcgs], labels=['Recall@10', 'NDCG@10'])
    plt.ylabel('Metric value')
    plt.title('Distribution of metrics across random seeds')
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, 'ml100k_metric_boxplot.png'), dpi=150)
    plt.close()
else:
    print('Plots were skipped because matplotlib is not installed.')