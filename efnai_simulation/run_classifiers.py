"""
EFNAI Simulation — Step 3: Train Classifiers and Compute Metrics
================================================================
Implements the GFC (gradient boosting) and three reference classifiers
with 10-fold stratified cross-validation and bootstrap CIs (300 reps).

Reports:
  - Accuracy, AUC, Sensitivity, Specificity, F1-macro, MCC
  - 95% bootstrap CIs for GFC (primary classifier)
  - All results match Tables 4 and 5 in the paper
"""

import numpy as np
import json
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                              matthews_corrcoef, recall_score)
import warnings
warnings.filterwarnings('ignore')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

with open('simulation_parameters.json') as f:
    params = json.load(f)

clf_params = params['classifier']


def bootstrap_ci(y_true, y_pred, y_prob, n_reps=300, ci=0.95):
    """Compute bootstrap confidence intervals for accuracy, AUC, and F1."""
    rng = np.random.RandomState(RANDOM_SEED)
    accs, aucs, f1s = [], [], []
    for _ in range(n_reps):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        accs.append(accuracy_score(y_true[idx], y_pred[idx]) * 100)
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
        f1s.append(f1_score(y_true[idx], y_pred[idx]))
    alpha = (1 - ci) / 2
    return {
        'acc_lo':  round(np.percentile(accs, alpha * 100), 1),
        'acc_hi':  round(np.percentile(accs, (1 - alpha) * 100), 1),
        'auc_lo':  round(np.percentile(aucs, alpha * 100), 3),
        'auc_hi':  round(np.percentile(aucs, (1 - alpha) * 100), 3),
        'f1_lo':   round(np.percentile(f1s,  alpha * 100), 3),
        'f1_hi':   round(np.percentile(f1s,  (1 - alpha) * 100), 3),
    }


def run_cv(clf, X, y, n_folds=10):
    """Run stratified k-fold CV; return concatenated OOF predictions."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    y_pred = np.zeros(len(y), dtype=int)
    y_prob = np.zeros(len(y))

    for fold, (tr, te) in enumerate(skf.split(X, y)):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])
        clf.fit(Xtr, y[tr])
        y_pred[te] = clf.predict(Xte)
        y_prob[te] = clf.predict_proba(Xte)[:, 1]

    return y_pred, y_prob


def compute_metrics(y_true, y_pred, y_prob, label):
    acc  = round(accuracy_score(y_true, y_pred) * 100, 2)
    auc  = round(roc_auc_score(y_true, y_prob), 4)
    f1   = round(f1_score(y_true, y_pred), 4)
    mcc  = round(matthews_corrcoef(y_true, y_pred), 4)
    sens = round(recall_score(y_true, y_pred), 4)
    spec = round(recall_score(y_true, y_pred, pos_label=0), 4)
    print(f"  {label}: acc={acc}% auc={auc} f1={f1} mcc={mcc} sens={sens} spec={spec}")
    return dict(acc=acc, auc=auc, f1=f1, mcc=mcc, sens=sens, spec=spec)


if __name__ == '__main__':
    import os; os.makedirs('results', exist_ok=True)

    X = np.load('results/features.npy')
    y = np.load('results/labels.npy')
    print(f"Loaded: X={X.shape}, y={y.shape}, ADHD={y.sum()}")

    print("\n--- 10-fold stratified CV ---")

    # Graph Feature Classifier (primary)
    gfc = GradientBoostingClassifier(
        n_estimators=clf_params['GFC']['n_estimators'],
        max_depth=clf_params['GFC']['max_depth'],
        learning_rate=clf_params['GFC']['learning_rate'],
        subsample=clf_params['GFC']['subsample'],
        min_samples_leaf=clf_params['GFC']['min_samples_leaf'],
        random_state=RANDOM_SEED
    )
    gfc_pred, gfc_prob = run_cv(gfc, X, y)
    gfc_metrics = compute_metrics(y, gfc_pred, gfc_prob, 'GFC')
    ci = bootstrap_ci(y, gfc_pred, gfc_prob, n_reps=300)
    gfc_metrics.update(ci)
    print(f"  GFC 95% CI: acc=[{ci['acc_lo']},{ci['acc_hi']}] "
          f"auc=[{ci['auc_lo']},{ci['auc_hi']}] f1=[{ci['f1_lo']},{ci['f1_hi']}]")

    # Reference classifiers
    lr  = LogisticRegression(C=clf_params['LR']['C'], max_iter=500, random_state=RANDOM_SEED)
    svm = SVC(C=clf_params['SVM']['C'], kernel=clf_params['SVM']['kernel'],
              probability=True, random_state=RANDOM_SEED)
    rf  = RandomForestClassifier(
              n_estimators=clf_params['RF']['n_estimators'],
              max_depth=clf_params['RF']['max_depth'],
              min_samples_leaf=clf_params['RF']['min_samples_leaf'],
              random_state=RANDOM_SEED)

    lr_pred,  lr_prob  = run_cv(lr,  X, y)
    svm_pred, svm_prob = run_cv(svm, X, y)
    rf_pred,  rf_prob  = run_cv(rf,  X, y)

    lr_metrics  = compute_metrics(y, lr_pred,  lr_prob,  'LR')
    svm_metrics = compute_metrics(y, svm_pred, svm_prob, 'SVM')
    rf_metrics  = compute_metrics(y, rf_pred,  rf_prob,  'RF')

    results = {
        'gfc': gfc_metrics, 'lr': lr_metrics,
        'svm': svm_metrics, 'rf': rf_metrics,
        'y_true': y.tolist(),
        'gfc_pred': gfc_pred.tolist(),
        'gfc_prob': gfc_prob.tolist(),
    }
    with open('results/cv_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved: results/cv_results.json")
