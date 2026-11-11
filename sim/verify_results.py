"""
Consistency checks on the saved results (does not retrain anything). Each check states what it tests.
Run after run_all.py. Exit code 1 if any check fails.
"""
import json, sys, numpy as np
cv = json.load(open('results/cv_results.json')); fed = json.load(open('results/federated_results.json'))
ka = json.load(open('results/known_answer.json')); dp = json.load(open('results/dp_utility.json'))
y = np.load('results/labels.npy'); X = np.load('results/features.npy')
from sklearn.metrics import roc_auc_score
checks = []
def chk(label, ok): checks.append(ok); print(('PASS ' if ok else 'FAIL ') + label)
chk('cohort size 973 (491 ADHD / 482 controls)', len(y) == 973 and int(y.sum()) == 491)
chk('no single feature separates classes (max single-feature AUC < 0.80)',
    max(max(roc_auc_score(y, X[:, j]), 1 - roc_auc_score(y, X[:, j])) for j in range(37)) < 0.80)
chk('CV AUC of every model is below the oracle linear AUC + 0.02', all(cv[m]['auc'] < ka['oracle']['auc'] + 0.02 for m in cv))
chk('label-permutation AUC is at chance (mean within 0.5 +/- 0.01)', abs(ka['permutation']['perm_auc_mean'] - 0.5) < 0.01)
chk('observed LR AUC exceeds the 95th percentile of permuted AUCs', ka['permutation']['observed_auc'] > ka['permutation']['perm_auc_p95'])
sw = ka['sweep']; sc = sorted(sw, key=float)
chk('LR AUC rises monotonically with injected effect size', all(sw[a]['LR']['auc_mean'] < sw[b]['LR']['auc_mean'] for a, b in zip(sc, sc[1:])))
chk('zero-effect cohorts sit at chance (|AUC - 0.5| < 0.04)', all(abs(sw['0.0'][m]['auc_mean'] - 0.5) < 0.04 for m in ('LR', 'GB')))
chk('federated (leave-one-site-out) files state that no DP was applied', fed['dp_applied'] is False)
chk('epsilon at sigma=1.1 is about 265', abs(dp['epsilon']['1.1'] - 265) < 1)
chk('epsilon decreases as sigma increases', all(dp['epsilon'][a] > dp['epsilon'][b] for a, b in zip(['1.1', '2', '3', '5', '7.5', '10'], ['2', '3', '5', '7.5', '10', '18.1'])))
sys.exit(0 if all(checks) else 1)
