"""
Known-answer checks. These are what make the simulation informative as a pipeline test:
  1. Oracle: the best achievable linear AUC/accuracy for the generated population (large sample),
     which cross-validated models should approach but not exceed.
  2. Effect-size sweep: performance should rise monotonically with the injected effect size, and sit
     at chance when the effect is zero.
  3. Label permutation: permuting labels must give chance performance (no leakage in the pipeline).
"""
import numpy as np, json
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from simulate_fc_data import simulate, cohort_labels, site_offsets, P
from extract_features import extract_all
from run_classifiers import make_models, run_cv, metrics
SEED = P['random_seed']; K = P['known_answer_checks']

def oracle():
    site, y = cohort_labels()
    off = site_offsets(np.random.RandomState(SEED + 10_000), 8)   # same site offsets as the main cohort
    Xs, ys = [], []
    for b in range(10):
        Xs.append(extract_all(simulate(777 + b, site, y, offsets=off))); ys.append(y)
    X = np.vstack(Xs); yy = np.concatenate(ys)
    Z = StandardScaler().fit_transform(X)
    lda = LinearDiscriminantAnalysis().fit(Z, yy)
    pr = lda.predict_proba(Z)[:, 1]
    return dict(n=len(yy), auc=float(roc_auc_score(yy, pr)), acc=float(100 * accuracy_score(yy, pr >= 0.5)))

def sweep():
    site, y = cohort_labels(); out = {}
    models = make_models()
    for sc in K['effect_scales']:
        rows = {m: [] for m in ('LR', 'GB')}
        for r in range(K['replicates_per_scale']):
            X = extract_all(simulate(SEED + r, site, y, effect_scale=sc))
            for m in rows:
                mt = metrics(y, run_cv(models[m], X, y)); rows[m].append((mt['acc'], mt['auc']))
        out[str(sc)] = {m: dict(acc_mean=float(np.mean([a for a, _ in v])), acc_sd=float(np.std([a for a, _ in v], ddof=1)),
                                auc_mean=float(np.mean([u for _, u in v])), auc_sd=float(np.std([u for _, u in v], ddof=1)))
                        for m, v in rows.items()}
        print('scale', sc, {m: round(out[str(sc)][m]['auc_mean'], 3) for m in rows})
    return out

def permutation():
    X = np.load('results/features.npy'); y = np.load('results/labels.npy')
    obs = metrics(y, run_cv(make_models()['LR'], X, y))
    rng = np.random.RandomState(SEED); aucs, accs = [], []
    for _ in range(K['label_permutations']):
        yp = rng.permutation(y); mt = metrics(yp, run_cv(make_models()['LR'], X, yp)); aucs.append(mt['auc']); accs.append(mt['acc'])
    return dict(model='LR', observed_auc=obs['auc'], observed_acc=obs['acc'], perm_auc_mean=float(np.mean(aucs)),
                perm_auc_sd=float(np.std(aucs, ddof=1)), perm_auc_p95=float(np.percentile(aucs, 95)),
                perm_acc_mean=float(np.mean(accs)), n_perm=len(aucs),
                p_value=float((1 + np.sum(np.array(aucs) >= obs['auc'])) / (1 + len(aucs))))

if __name__ == '__main__':
    res = {'oracle': oracle()}; print('oracle', res['oracle'])
    res['permutation'] = permutation(); print('perm', res['permutation'])
    res['sweep'] = sweep()
    json.dump(res, open('results/known_answer.json', 'w'), indent=2)
