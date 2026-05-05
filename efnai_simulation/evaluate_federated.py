"""
EFNAI Simulation — Step 4: Federated Site-Held-Out Evaluation
=============================================================
Performs leave-one-site-out evaluation using the GFC.
No differential privacy is applied — results are upper bounds only.
"""

import numpy as np
import json
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings; warnings.filterwarnings('ignore')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

with open('simulation_parameters.json') as f:
    params = json.load(f)

SITES     = params['sites']
clf_p     = params['classifier']['GFC']

if __name__ == '__main__':
    import os; os.makedirs('results', exist_ok=True)

    X        = np.load('results/features.npy')
    y        = np.load('results/labels.npy')
    site_ids = np.load('results/site_ids.npy')

    print("--- Federated site-held-out evaluation (no DP — upper bound) ---")
    site_results = []
    for site_id, site_info in enumerate(SITES):
        te = site_ids == site_id
        tr = ~te
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])
        gfc = GradientBoostingClassifier(
            n_estimators=clf_p['n_estimators'], max_depth=clf_p['max_depth'],
            learning_rate=clf_p['learning_rate'], subsample=clf_p['subsample'],
            min_samples_leaf=clf_p['min_samples_leaf'], random_state=RANDOM_SEED)
        gfc.fit(Xtr, y[tr])
        pred = gfc.predict(Xte)
        prob = gfc.predict_proba(Xte)[:, 1]
        acc  = round(accuracy_score(y[te], pred) * 100, 2)
        auc  = round(roc_auc_score(y[te], prob), 4)
        print(f"  Site {site_id+1} ({site_info['name']:12s}) "
              f"n={site_info['n']:3d}  acc={acc:.2f}%  auc={auc:.4f}")
        site_results.append({'site': site_id+1, 'name': site_info['name'],
                              'n': site_info['n'], 'acc': acc, 'auc': auc})

    accs = [r['acc'] for r in site_results]
    aucs = [r['auc'] for r in site_results]
    print(f"\n  Mean ± SD:  acc={np.mean(accs):.2f}% ± {np.std(accs):.2f}%  "
          f"auc={np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    fed_summary = {
        'site_results': site_results,
        'mean_acc': round(float(np.mean(accs)), 2),
        'std_acc':  round(float(np.std(accs)),  2),
        'mean_auc': round(float(np.mean(aucs)), 4),
        'std_auc':  round(float(np.std(aucs)),  4),
        'dp_applied': False,
        'note': 'Upper bound only — no DP applied to federated evaluation'
    }
    with open('results/federated_results.json', 'w') as f:
        json.dump(fed_summary, f, indent=2)
    print("Saved: results/federated_results.json")
