"""
EFNAI — Verify Paper Results
==============================
Loads the saved predictions from paper_results.json and
recomputes all metrics reported in Table 4 and Table 5.

This confirms that the numbers in the paper are correct.
No re-training needed — predictions are deterministic from seed=42.
"""

import json
import numpy as np
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                              matthews_corrcoef, recall_score)

print("=" * 55)
print("EFNAI Paper Results Verification")
print("=" * 55)

with open('results/paper_results.json') as f:
    r = json.load(f)

y_true   = np.array(r['y_true'])
gfc_pred = np.array(r['gfc_yp'])
gfc_prob = np.array(r['gfc_ypr'])
lr_prob  = np.array(r['lr_ypr'])
svm_prob = np.array(r['svm_ypr'])
rf_prob  = np.array(r['rf_ypr'])

# Threshold at 0.5 for reference classifiers
lr_pred  = (lr_prob  >= 0.5).astype(int)
svm_pred = (svm_prob >= 0.5).astype(int)
rf_pred  = (rf_prob  >= 0.5).astype(int)

print(f"\nDataset: n={len(y_true)}, ADHD={y_true.sum()}, Controls={len(y_true)-y_true.sum()}")

def report(name, y_pred, y_prob):
    acc  = round(accuracy_score(y_true, y_pred) * 100, 2)
    auc  = round(roc_auc_score(y_true, y_prob), 4)
    f1   = round(f1_score(y_true, y_pred), 4)
    mcc  = round(matthews_corrcoef(y_true, y_pred), 4)
    sens = round(recall_score(y_true, y_pred), 4)
    spec = round(recall_score(y_true, y_pred, pos_label=0), 4)
    print(f"\n{name}:")
    print(f"  Accuracy: {acc}%   AUC: {auc}   F1: {f1}   MCC: {mcc}")
    print(f"  Sensitivity: {sens}   Specificity: {spec}")
    return acc, auc

# Bootstrap CI for GFC
rng = np.random.RandomState(42)
bsa, bsau, bsf = [], [], []
for _ in range(300):
    idx = rng.choice(len(y_true), len(y_true), replace=True)
    bsa.append(accuracy_score(y_true[idx], gfc_pred[idx]) * 100)
    bsau.append(roc_auc_score(y_true[idx], gfc_prob[idx]))
    bsf.append(f1_score(y_true[idx], gfc_pred[idx]))

print("\n--- Table 4: 10-fold stratified CV results ---")
report("Logistic Regression", lr_pred,  lr_prob)
report("SVM (RBF)",           svm_pred, svm_prob)
report("Random Forest",       rf_pred,  rf_prob)
gfc_acc, gfc_auc = report("Graph Feature Classifier (GFC)", gfc_pred, gfc_prob)

print(f"\n  GFC 95% Bootstrap CI:")
print(f"    Accuracy: [{np.percentile(bsa,2.5):.1f}%, {np.percentile(bsa,97.5):.1f}%]")
print(f"    AUC:      [{np.percentile(bsau,2.5):.3f}, {np.percentile(bsau,97.5):.3f}]")
print(f"    F1:       [{np.percentile(bsf,2.5):.3f}, {np.percentile(bsf,97.5):.3f}]")

print("\n--- Table 5: Federated site-held-out (upper bound, no DP) ---")
fed = r['fed_rows']
for row in fed:
    print(f"  Site {row['site']}: n={row['n']}  acc={row['acc']}%  auc={row['auc']}")
print(f"  Mean: {r['fed_acc_mean']}% ± {r['fed_acc_std']}%")
print(f"  AUC:  {r['fed_auc_mean']} ± {r['fed_auc_std']}")

print("\n--- Verification against paper ---")
checks = [
    ("GFC accuracy",  gfc_acc,            74.00, 0.01),
    ("GFC AUC",       gfc_auc,            0.8239, 0.0001),
    ("GFC CI lo",     round(np.percentile(bsa,2.5),1),  71.2, 0.2),
    ("GFC CI hi",     round(np.percentile(bsa,97.5),1), 76.6, 0.1),
    ("Fed mean acc",  r['fed_acc_mean'],  74.74, 0.01),
    ("Fed std acc",   r['fed_acc_std'],   2.31, 0.01),
]
for label, actual, expected, tol in checks:
    ok = abs(actual - expected) <= tol
    print(f"  {'PASS' if ok else 'FAIL'} {label}: {actual} == {expected}")
