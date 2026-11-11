"""
Step 4: leave-one-site-out evaluation, all four models. No differential privacy is applied here.
(Nothing is federated or privatised in this script; it only checks cross-site generalisation.)
"""
import numpy as np, json
from sklearn.metrics import accuracy_score, roc_auc_score
from run_classifiers import make_models, fit_predict
P = json.load(open('simulation_parameters.json'))

if __name__ == '__main__':
    X = np.load('results/features.npy'); y = np.load('results/labels.npy'); s = np.load('results/site_ids.npy')
    res = {}
    for name, make in make_models().items():
        rows = []
        for k, info in enumerate(P['sites']):
            te = s == k
            prob = fit_predict(make, X[~te], y[~te], X[te])
            rows.append(dict(site=k + 1, name=info['name'], n=int(te.sum()), n_adhd=int(y[te].sum()),
                             acc=100 * accuracy_score(y[te], prob >= 0.5), auc=roc_auc_score(y[te], prob)))
        a = [r['acc'] for r in rows]; u = [r['auc'] for r in rows]
        res[name] = dict(sites=rows, acc_mean=float(np.mean(a)), acc_sd=float(np.std(a, ddof=1)),
                         auc_mean=float(np.mean(u)), auc_sd=float(np.std(u, ddof=1)))
        print(f"{name}: acc {np.mean(a):.2f} +/- {np.std(a, ddof=1):.2f}  auc {np.mean(u):.4f} +/- {np.std(u, ddof=1):.4f}")
    res['dp_applied'] = False
    json.dump(res, open('results/federated_results.json', 'w'), indent=2)
