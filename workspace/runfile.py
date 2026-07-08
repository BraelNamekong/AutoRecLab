import os
import numpy as np
import pandas as pd
import json
import math
import logging
import time

# Working directory per guidelines
working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

DATA_PATH = os.path.join(os.getcwd(), 'u.data')  # MovieLens 100K

# --------- Data loading and 0-based mapping ---------

def load_and_map_movielens_100k(path):
    df = pd.read_csv(path, sep='\t', header=None, names=['user','item','rating','timestamp'], engine='python')
    df['user'] = df['user'].astype(int)
    df['item'] = df['item'].astype(int)
    df['rating'] = df['rating'].astype(float)
    df['timestamp'] = df['timestamp'].astype(int)
    # Build 0-based mappings for users/items
    unique_users = sorted(df['user'].unique())
    unique_items = sorted(df['item'].unique())
    user_map = {u: idx for idx, u in enumerate(unique_users)}
    item_map = {it: idx for idx, it in enumerate(unique_items)}
    df['user0'] = df['user'].map(user_map)
    df['item0'] = df['item'].map(item_map)
    n_users = len(unique_users)
    n_items = len(unique_items)
    logging.info(f"Loaded {len(df)} interactions; mapped to {n_users} users and {n_items} items with 0-based indices")
    return df, user_map, item_map, n_users, n_items

# --------- Train/test split per seed (per-user) ---------

def train_test_split_by_user_map(df, seed, test_frac=0.2):
    rng = np.random.default_rng(seed)
    train_rows = []
    test_rows = []
    for u0, g in df.groupby('user0'):
        idxs = g.index.values
        m = len(idxs)
        if m == 0:
            continue
        k = max(1, int(math.ceil(m * test_frac)))
        test_idx = rng.choice(idxs, size=k, replace=False)
        train_mask = ~df.index.isin(test_idx)
        train_rows.append(df.loc[train_mask & (df['user0'] == u0)])
        test_rows.append(df.loc[df.index.isin(test_idx)])
    train_df = pd.concat(train_rows) if train_rows else df.iloc[0:0]
    test_df = pd.concat(test_rows) if test_rows else df.iloc[0:0]
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

# --------- Implicit feedback preparation ---------

def to_implicit(train_df):
    # Convert to implicit interaction DataFrame: user0, item0, rating=1.0
    imp = train_df[['user0','item0']].copy()
    imp = imp.rename(columns={'user0':'user','item0':'item'})
    imp['rating'] = 1.0
    return imp

# --------- Evaluation helpers ---------

def ndcg_at_k(rank_pos, k):
    if rank_pos < 1 or rank_pos > k:
        return 0.0
    return 1.0 / math.log2(rank_pos + 1)


def evaluate_seed(model, train_items, test_df, K=10):
    ndcg_sum = 0.0
    recall_sum = 0.0
    count = 0
    items_list = sorted(list(train_items))
    for _, row in test_df.iterrows():
        u = int(row['user0'])
        true_i = int(row['item0'])
        if u not in getattr(model, 'user_map', {}) and getattr(model, 'user_map', None) is not None:
            continue
        scores = []
        for it in items_list:
            try:
                s = model.score(u, it)
            except Exception:
                s = 0.0
            scores.append((it, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        topk = [it for it, _ in scores[:K]]
        if true_i in topk:
            rank_pos = topk.index(true_i) + 1
            ndcg_sum += ndcg_at_k(rank_pos, K)
            recall_sum += 1.0
        count += 1 if count is not None else 0
    if count == 0:
        return 0.0, 0.0
    ndcg = ndcg_sum / count
    recall = recall_sum / count
    return ndcg, recall

# --------- Fallback simple MF (SGD) just in case LensKit not available ---------


def train_sgd_mf(train_df, latent_dim=20, lr=0.05, reg=0.01, n_epochs=10, seed=0):
    users = sorted(train_df['user0'].unique())
    items = sorted(train_df['item0'].unique())
    user_to_idx = {u:i for i,u in enumerate(users)}
    item_to_idx = {it:i for i,it in enumerate(items)}
    n_users = len(users); n_items = len(items)
    rng = np.random.default_rng(seed)
    U = 0.01 * rng.standard_normal((n_users, latent_dim))
    V = 0.01 * rng.standard_normal((n_items, latent_dim))
    interactions = [(user_to_idx[r['user0']], item_to_idx[r['item0']]) for _, r in train_df.iterrows()]
    train_losses = []
    for epoch in range(n_epochs):
        rng.shuffle(interactions)
        loss = 0.0
        for (u_idx, i_idx) in interactions:
            u = U[u_idx]; v = V[i_idx]
            pred = u.dot(v)
            err = 1.0 - pred
            U[u_idx] += lr * (err * v - reg * u)
            V[i_idx] += lr * (err * (U[u_idx]) - reg * v)
            loss += err*err
        train_losses.append((epoch, np.sqrt(loss/len(interactions))))
    class SimpleModel:
        pass
    m = SimpleModel()
    m.U = U; m.V = V; m.user_map = {u:i for i,u in enumerate(users)}; m.item_map = {i: it for it,i in enumerate(items)}
    def score(u, it):
        if (u in m.user_map) and (it in m.item_map):
            ui = m.user_map[u]
            ii = m.item_map[it]
            return float(m.U[ui] @ m.V[ii])
        return 0.0
    m.score = score
    return m, train_losses

# --------- Main experiment orchestrator ---------


def main():
    df, user_map, item_map, n_users, n_items = load_and_map_movielens_100k(DATA_PATH)
    seeds = [0,1,2,3,4,5]
    K = 10
    latent_dim = 20
    n_epochs = 8
    lr = 0.05
    reg = 0.01

    per_seed_results = []
    all_ndcgs = []
    all_recalls = []
    experiment_data = {
        'movielens_100k': {
            'seed_results': [],
            'per_seed_metrics': []
        }
    }

    for s in seeds:
        train_df, test_df = train_test_split_by_user_map(df, seed=s, test_frac=0.2)
        if train_df.empty or test_df.empty:
            logging.warning(f"Seed {s}: empty train/test split, skipping seed.")
            continue
        train_implicit = to_implicit(train_df)
        model_wrapper = None
        use_lenskit = False
        try:
            from lenskit.algorithms.implicit import ImplicitMF
            use_lenskit = True
            algo = ImplicitMF(latent_dim=latent_dim, learning_rate=lr, reg=reg, iterations=n_epochs, seed=s)
            model = algo.fit(train_implicit)
            class LKModelWrapper:
                def __init__(self, model, user_set, item_set):
                    self.model = model
                    self.user_map = {u: True for u in user_set}
                    self.item_map = {i: True for i in item_set}
                def score(self, u, it):
                    if hasattr(self.model, 'score'):
                        return float(self.model.score(u, it))
                    if hasattr(self.model, 'predict'):
                        return float(self.model.predict(u, it))
                    return 0.0
            # create wrapper with training users/items
            user_set = set(train_df['user0'].unique())
            item_set = set(train_df['item0'].unique())
            model_wrapper = LKModelWrapper(model, user_set, item_set)
        except Exception as e:
            logging.warning(f"Seed {s}: LensKit ImplicitMF not used due to error: {e}")
            model_wrapper = None

        if model_wrapper is None:
            model_fallback, train_losses = train_sgd_mf(train_df, latent_dim=latent_dim, lr=lr, reg=reg, n_epochs=n_epochs, seed=s)
            model_wrapper = model_fallback

        train_items = set(train_df['item0'].unique())
        ndcg, recall = evaluate_seed(model_wrapper, train_items, test_df, K=K)
        per_seed_results.append({'seed': s, 'ndcg': ndcg, 'recall': recall})
        all_ndcgs.append(ndcg)
        all_recalls.append(recall)
        logging.info(f"Seed {s}: NDCG@{K}={ndcg:.4f}, Recall@{K}={recall:.4f}")

        # Validation-like loss on test set
        losses = []
        for _, row in test_df.iterrows():
            u = int(row['user0']); it = int(row['item0'])
            s_pred = model_wrapper.score(u, it) if hasattr(model_wrapper, 'score') else 0.0
            losses.append((1.0 - float(s_pred))**2)
        val_loss = float(np.mean(losses)) if losses else 0.0
        logging.info(f"Seed {s}: validation-like loss = {val_loss:.6f}")

        experiment_data['movielens_100k']['seed_results'].append({'seed': s, 'ndcg': ndcg, 'recall': recall, 'val_loss': val_loss})
        experiment_data['movielens_100k']['per_seed_metrics'] = per_seed_results

    if len(all_ndcgs) > 0:
        mean_ndcg = float(np.mean(all_ndcgs))
        std_ndcg = float(np.std(all_ndcgs, ddof=1)) if len(all_ndcgs) > 1 else 0.0
        ci_ndcg = 1.96 * std_ndcg / math.sqrt(len(all_ndcgs)) if len(all_ndcgs) > 0 else 0.0
        mean_recall = float(np.mean(all_recalls))
        std_recall = float(np.std(all_recalls, ddof=1)) if len(all_recalls) > 1 else 0.0
        ci_recall = 1.96 * std_recall / math.sqrt(len(all_recalls)) if len(all_recalls) > 0 else 0.0
        summary = {
            'mean_ndcg': mean_ndcg, 'std_ndcg': std_ndcg, 'ci_ndcg': ci_ndcg,
            'mean_recall': mean_recall, 'std_recall': std_recall, 'ci_recall': ci_recall,
            'num_seeds': len(all_ndcgs)
        }
        logging.info(f"Aggregate across seeds: {summary}")
        experiment_data['movielens_100k']['aggregate'] = summary
        df_seed = pd.DataFrame(per_seed_results, columns=['seed','ndcg','recall'])
        csv_path = os.path.join(working_dir, 'per_seed_metrics.csv')
        df_seed.to_csv(csv_path, index=False)
        logging.info(f"Per-seed metrics saved to {csv_path}")
        json_path = os.path.join(working_dir, 'aggregate_metrics.json')
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logging.info(f"Aggregate metrics saved to {json_path}")
        np.save(os.path.join(working_dir, 'experiment_data.npy'), experiment_data)
    else:
        logging.warning("No seeds produced results; nothing to aggregate.")

# Execute at import time per guidelines
start_time = time.time()
main()
elapsed = time.time() - start_time
logging.info(f"Experiment finished in {elapsed:.2f}s")
