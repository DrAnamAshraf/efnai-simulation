"""
Privacy accounting and utility simulation for the proposed record-level DP federated protocol.

Part A. Epsilon from the PLD accountant (dp_accounting; Poisson-subsampled Gaussian mechanism) for the
        pilot setting: n_local=50, batch 32 (q = 0.64), 100 rounds x 5 local epochs x ceil(50/32)=2 steps = 1000 steps.
Part B. Utility: FedAvg with DP-SGD logistic regression on the 37 simulated FC features
        (ADHD vs control, 3 sites x 50 training subjects), against non-private baselines.
Part C. rPDP-FL composition table, UNDER THE STATED ASSUMPTION that record i is first admitted with
        probability p_i and then batch-sampled with q; with independent Poisson draws the per-step inclusion
        probability is p_i*q. This is a model of the mechanism, to be re-derived once the exact rPDP-FL
        implementation is fixed.
Not modelled: ASAS (no published effect sizes exist to simulate it), feature standardisation cost,
hyper-parameter tuning cost, the d-ComBat step.
"""
import numpy as np, json, math
import dp_accounting as dpa
from dp_accounting import pld
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from simulate_fc_data import simulate, site_offsets, P
from extract_features import extract_all
D = P['dp']; SEED = P['random_seed']
STEPS = D['rounds'] * D['local_epochs'] * math.ceil(D['n_local'] / D['batch_size'])
Q = D['batch_size'] / D['n_local']

def epsilon(sigma, q=Q, steps=STEPS, delta=D['delta']):
    acc = pld.PLDAccountant()
    acc.compose(dpa.SelfComposedDpEvent(dpa.PoissonSampledDpEvent(q, dpa.GaussianDpEvent(sigma)), steps))
    return float(acc.get_epsilon(delta))

def sigma_for(target, lo=1.0, hi=200.0):
    for _ in range(30):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if epsilon(m) > target else (lo, m)
    return round(m, 1)

def sig(z): return 1 / (1 + np.exp(-z))

def fedavg(Xs, ys, sigma, rng, clip=D['clip_norm'], private=True):
    d = Xs[0].shape[1] + 1; w = np.zeros(d)
    for _ in range(D['rounds']):
        ws = []
        for X, y in zip(Xs, ys):
            wk = w.copy(); n = len(y); Xa = np.hstack([X, np.ones((n, 1))])
            for _e in range(D['local_epochs']):
                for _s in range(math.ceil(n / D['batch_size'])):
                    m = rng.random(n) < Q
                    if not m.any() and private: continue
                    g = (sig(Xa[m] @ wk) - y[m])[:, None] * Xa[m]
                    if private:
                        g = g / np.maximum(1.0, np.linalg.norm(g, axis=1, keepdims=True) / clip)
                        grad = (g.sum(0) + rng.normal(0, sigma * clip, d)) / D['batch_size']
                    else:
                        grad = g.mean(0)
                    wk -= D['learning_rate'] * grad
            ws.append(wk)
        w = np.mean(ws, axis=0)
    return w

def build_pool():
    """Pool of simulated feature vectors from the three pilot-like sites (site indices 0-2)."""
    rng = np.random.RandomState(SEED + 555); n = D['pool_size']
    site = np.tile([0, 1, 2], n // 3); y = rng.randint(0, 2, n)
    off = site_offsets(np.random.RandomState(SEED + 10_000), 8)
    X = np.vstack([extract_all(simulate(SEED + 900 + b, site[b*500:(b+1)*500], y[b*500:(b+1)*500], offsets=off))
                   for b in range(n // 500)])
    return X, y, site

if __name__ == '__main__':
    out = {'steps_per_site': STEPS, 'q': Q, 'delta': D['delta']}
    out['epsilon'] = {str(s): epsilon(s) for s in D['sigmas']}
    out['sigma_for_eps'] = {str(t): sigma_for(t) for t in (1, 5, 10, 25)}
    print('eps', {k: round(v, 2) for k, v in out['epsilon'].items()}); print('sigma_for', out['sigma_for_eps'])

    X, y, site = build_pool()
    mu, sd = X[:2000].mean(0), X[:2000].std(0)   # treated as a public reference cohort (assumption)
    Xr, yr, sr = (X[2000:] - mu) / sd, y[2000:], site[2000:]
    rng = np.random.default_rng(SEED)
    res = {k: [] for k in ['central_LR', 'fedavg_nonprivate'] + [f's{s}' for s in D['sigmas']]}
    for rep in range(D['replicates']):
        tr_idx = []
        for k in range(3):
            for cls in (0, 1):
                tr_idx += list(rng.choice(np.where((sr == k) & (yr == cls))[0], 25, replace=False))
        tr_idx = np.array(tr_idx); rest = np.setdiff1d(np.arange(len(yr)), tr_idx)
        te = rng.choice(rest, D['test_size'], replace=False)
        Xs = [Xr[[i for i in tr_idx if sr[i] == k]] for k in range(3)]; ys = [yr[[i for i in tr_idx if sr[i] == k]] for k in range(3)]
        def ev(w):
            z = np.hstack([Xr[te], np.ones((len(te), 1))]) @ w
            return 100 * float(np.mean((z > 0) == yr[te])), float(roc_auc_score(yr[te], z))
        c = LogisticRegression(C=0.3, max_iter=500).fit(Xr[tr_idx], yr[tr_idx]); zc = c.decision_function(Xr[te])
        res['central_LR'].append((100 * float(np.mean((zc > 0) == yr[te])), float(roc_auc_score(yr[te], zc))))
        res['fedavg_nonprivate'].append(ev(fedavg(Xs, ys, 0, rng, private=False)))
        for s in D['sigmas']:
            res[f's{s}'].append(ev(fedavg(Xs, ys, s, rng)))
    summ = {k: dict(acc_mean=float(np.mean([a for a, _ in v])), acc_sd=float(np.std([a for a, _ in v], ddof=1)),
                    auc_mean=float(np.mean([u for _, u in v])), auc_sd=float(np.std([u for _, u in v], ddof=1)))
            for k, v in res.items()}
    out['utility'] = summ; out['replicates'] = D['replicates']
    for k, v in summ.items(): print(k, round(v['acc_mean'], 1), round(v['auc_mean'], 3))

    out['rpdp_assumed_composition'] = {str(s): {str(p): epsilon(s, q=Q * p) for p in (1.0, 0.5, 0.25, 0.1)} for s in (1.1, 5, 10, 18.1)}
    json.dump(out, open('results/dp_utility.json', 'w'), indent=2)
