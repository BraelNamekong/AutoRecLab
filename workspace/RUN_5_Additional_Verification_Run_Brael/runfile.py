import os
working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from lenskit.algorithms.als import ImplicitMF
from lenskit.algorithms import Recommender
from lenskit import batch
from lenskit.topn import RecListAnalysis, ndcg, recall

CONFIG = {
    'DATA_PATH': os.path.join(os.getcwd(), 'u.data'),
    'RELEVANCE_THRESHOLD': 4,
    'TEST_FRACTION': 0.2,
    'SPLIT_SEED': 2024,
    'K': 10,
    'SEEDS': [1, 7, 13, 21, 42, 87, 123, 256, 512, 1024],
    'IMF_FEATURES': 30,
    'IMF_ITERATIONS': 20,
    'IMF_REG': 0.01,
    'IMF_WEIGHT': 40,
}

experiment_data = {
    'movielens100k': {
        'metrics': {'train': [], 'val': []},
        'losses': {'train': [], 'val': []},
        'predictions': [],
        'ground_truth': [],
        'timestamps': [],
        'seeds': [],
        'config': CONFIG.copy(),
        'split_info': {}
    }
}

cols = ['user', 'item', 'rating', 'timestamp']
ratings = pd.read_csv(CONFIG['DATA_PATH'], sep='\t', names=cols, engine='python')
implicit = ratings.loc[ratings['rating'] >= CONFIG['RELEVANCE_THRESHOLD'], ['user', 'item']].drop_duplicates().copy()
implicit['rating'] = 1.0
implicit = implicit.sort_values(['user', 'item']).reset_index(drop=True)

rng = np.random.RandomState(CONFIG['SPLIT_SEED'])
train_parts, test_parts = [], []
for user, grp in implicit.groupby('user', sort=True):
    idx = np.arange(len(grp))
    if len(idx) < 2:
        train_parts.append(grp)
        continue
    n_test = max(1, int(np.ceil(CONFIG['TEST_FRACTION'] * len(idx))))
    perm = rng.permutation(idx)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    if len(train_idx) == 0:
        train_idx = perm[:-1]
        test_idx = perm[-1:]
    train_parts.append(grp.iloc[train_idx])
    test_parts.append(grp.iloc[test_idx])

train_df = pd.concat(train_parts, ignore_index=True)
test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=implicit.columns)

initial_test_users = set(test_df['user'].unique())
train_users = set(train_df['user'].unique())
train_items = set(train_df['item'].unique())
test_df = test_df[test_df['user'].isin(train_users) & test_df['item'].isin(train_items)].copy()
truth_pairs = test_df[['user', 'item']].copy()
truth_sizes = truth_pairs.groupby('user').size()
valid_users = np.array(sorted(truth_sizes[truth_sizes > 0].index.values))
excluded_users = sorted(initial_test_users - set(valid_users))
truth_pairs = truth_pairs[truth_pairs['user'].isin(valid_users)].copy()

experiment_data['movielens100k']['split_info'] = {
    'n_total_interactions': int(len(implicit)),
    'n_train_interactions': int(len(train_df)),
    'n_test_interactions_before_filter': int(len(pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=implicit.columns))),
    'n_test_interactions_after_filter': int(len(truth_pairs)),
    'n_eval_users': int(len(valid_users)),
    'n_excluded_test_users_after_filter': int(len(excluded_users)),
    'excluded_users_sample': excluded_users[:20],
}

print(f"Evaluation uses K={CONFIG['K']} with relevance threshold rating>={CONFIG['RELEVANCE_THRESHOLD']}")
print(f"Fixed split seed={CONFIG['SPLIT_SEED']}; varying only model seeds={CONFIG['SEEDS']}")
print(f"Excluded {len(excluded_users)} test users with empty/invalid ground truth after filtering to train-known items.")

all_metrics = []
all_preds_compact = []

for run_idx, seed in enumerate(CONFIG['SEEDS'], start=1):
    algo = Recommender.adapt(ImplicitMF(
        features=CONFIG['IMF_FEATURES'],
        iterations=CONFIG['IMF_ITERATIONS'],
        reg=CONFIG['IMF_REG'],
        weight=CONFIG['IMF_WEIGHT'],
        rng_spec=seed,
    ))
    algo.fit(train_df)

    recs = batch.recommend(algo, valid_users, CONFIG['K'], n_jobs=1)
    recs = recs[recs['user'].isin(valid_users)].copy()

    rla = RecListAnalysis()
    rla.add_metric(ndcg)
    rla.add_metric(recall)
    results = rla.compute(recs, truth_pairs)

    mean_ndcg = float(results['ndcg'].mean())
    mean_recall = float(results['recall'].mean())
    val_loss = np.nan
    print(f'Epoch {run_idx}: validation_loss = {val_loss:.4f}')
    print(f"Run {run_idx} seed={seed}: NDCG@{CONFIG['K']} = {mean_ndcg:.4f}, Recall@{CONFIG['K']} = {mean_recall:.4f}")

    run_metrics = {
        'run': run_idx,
        'seed': seed,
        f'ndcg@{CONFIG["K"]}': mean_ndcg,
        f'recall@{CONFIG["K"]}': mean_recall,
        'n_eval_users': int(len(valid_users))
    }
    all_metrics.append(run_metrics)
    experiment_data['movielens100k']['metrics']['val'].append(run_metrics)
    experiment_data['movielens100k']['losses']['val'].append({'run': run_idx, 'seed': seed, 'validation_loss': val_loss})
    experiment_data['movielens100k']['predictions'].append(recs[['user', 'item', 'rank', 'score']].to_dict('records'))
    experiment_data['movielens100k']['ground_truth'].append(truth_pairs.to_dict('records'))
    experiment_data['movielens100k']['timestamps'].append(pd.Timestamp.now().isoformat())
    experiment_data['movielens100k']['seeds'].append(seed)
    all_preds_compact.append(recs[['user', 'item', 'rank', 'score']].to_numpy(dtype=object))

metrics_df = pd.DataFrame(all_metrics)
ndcg_col = f'ndcg@{CONFIG["K"]}'
recall_col = f'recall@{CONFIG["K"]}'
summary = {
    'n_runs': int(len(metrics_df)),
    'n_users_eval': int(len(valid_users)),
    'n_train_interactions': int(len(train_df)),
    'n_test_interactions': int(len(truth_pairs)),
    'k': int(CONFIG['K']),
    'ndcg_mean': float(metrics_df[ndcg_col].mean()),
    'ndcg_std': float(metrics_df[ndcg_col].std(ddof=1)),
    'ndcg_min': float(metrics_df[ndcg_col].min()),
    'ndcg_max': float(metrics_df[ndcg_col].max()),
    'ndcg_range': float(metrics_df[ndcg_col].max() - metrics_df[ndcg_col].min()),
    'ndcg_cv': float(metrics_df[ndcg_col].std(ddof=1) / metrics_df[ndcg_col].mean()) if metrics_df[ndcg_col].mean() != 0 else np.nan,
    'recall_mean': float(metrics_df[recall_col].mean()),
    'recall_std': float(metrics_df[recall_col].std(ddof=1)),
    'recall_min': float(metrics_df[recall_col].min()),
    'recall_max': float(metrics_df[recall_col].max()),
    'recall_range': float(metrics_df[recall_col].max() - metrics_df[recall_col].min()),
    'recall_cv': float(metrics_df[recall_col].std(ddof=1) / metrics_df[recall_col].mean()) if metrics_df[recall_col].mean() != 0 else np.nan,
}

print('\nSeed sensitivity summary:')
for k, v in summary.items():
    print(f'{k}: {v:.6f}' if isinstance(v, float) else f'{k}: {v}')

print('\nPer-seed results:')
print(metrics_df.to_string(index=False))

if summary['ndcg_cv'] > 0.01 or summary['recall_cv'] > 0.01 or summary['ndcg_range'] > 0.01 or summary['recall_range'] > 0.01:
    interpretation = 'Seed effects are noticeable enough that reporting random seeds or multi-seed averages is important.'
else:
    interpretation = 'Seed effects are small in this setup, but reporting seeds remains good practice for reproducibility.'
print('\nInterpretation:')
print(interpretation)

np.save(os.path.join(working_dir, 'movielens100k_seed_ndcg.npy'), metrics_df[ndcg_col].to_numpy())
np.save(os.path.join(working_dir, 'movielens100k_seed_recall.npy'), metrics_df[recall_col].to_numpy())
np.save(os.path.join(working_dir, 'movielens100k_seed_values.npy'), metrics_df['seed'].to_numpy())
np.save(os.path.join(working_dir, 'movielens100k_seed_runs.npy'), metrics_df['run'].to_numpy())
np.save(os.path.join(working_dir, 'movielens100k_truth_pairs.npy'), truth_pairs.to_numpy(dtype=object))
np.save(os.path.join(working_dir, 'movielens100k_eval_users.npy'), valid_users)
np.save(os.path.join(working_dir, 'movielens100k_predictions_by_run.npy'), np.array(all_preds_compact, dtype=object), allow_pickle=True)
np.save(os.path.join(working_dir, 'movielens100k_train_df.npy'), train_df.to_numpy(dtype=object))
np.save(os.path.join(working_dir, 'movielens100k_test_df.npy'), truth_pairs.to_numpy(dtype=object))
np.savez_compressed(
    os.path.join(working_dir, 'movielens100k_summary_arrays.npz'),
    ndcg=metrics_df[ndcg_col].to_numpy(),
    recall=metrics_df[recall_col].to_numpy(),
    seeds=metrics_df['seed'].to_numpy(),
    runs=metrics_df['run'].to_numpy(),
)
np.save(os.path.join(working_dir, 'experiment_data.npy'), experiment_data, allow_pickle=True)
metrics_df.to_csv(os.path.join(working_dir, 'movielens100k_seed_metrics.csv'), index=False)
pd.DataFrame([summary]).to_csv(os.path.join(working_dir, 'movielens100k_seed_summary.csv'), index=False)
with open(os.path.join(working_dir, 'movielens100k_seed_interpretation.txt'), 'w', encoding='utf-8') as f:
    f.write(interpretation + '\n')

try:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    plt.plot(metrics_df['seed'], metrics_df[ndcg_col], marker='o', label=ndcg_col.upper())
    plt.plot(metrics_df['seed'], metrics_df[recall_col], marker='s', label=recall_col.upper())
    plt.xlabel('Random seed')
    plt.ylabel('Metric value')
    plt.title('MovieLens100K: metric variation across random seeds')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, 'movielens100k_seed_metric_lines.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.boxplot([metrics_df[ndcg_col], metrics_df[recall_col]], labels=[ndcg_col.upper(), recall_col.upper()])
    plt.ylabel('Metric value')
    plt.title('MovieLens100K: distribution of metrics across seeds')
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, 'movielens100k_seed_metric_boxplot.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.scatter(metrics_df[ndcg_col], metrics_df[recall_col])
    for _, row in metrics_df.iterrows():
        plt.annotate(str(int(row['seed'])), (row[ndcg_col], row[recall_col]), fontsize=8)
    plt.xlabel(ndcg_col.upper())
    plt.ylabel(recall_col.upper())
    plt.title('MovieLens100K: seed-level metric scatter')
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, 'movielens100k_seed_metric_scatter.png'), dpi=150)
    plt.close()
except ImportError:
    pass