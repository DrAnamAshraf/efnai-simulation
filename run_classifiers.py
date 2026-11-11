"""
Step 3: four classifiers, 10-fold stratified CV, bootstrap CIs for every model.
No model is designated 'primary'; all are reported on equal terms.
Predictions are thresholded at 0.5 on the predicted probability for every model.
"""
import numpy as np, json, warnings
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef, recall_score
warnings.filterwarnings('ignore')
P = json.load(open('simulation_parameters.json')); SEED = P['random_seed']; C = P['classifiers']

def make_models():
    return {
     'LR':  lambda: LogisticRegression(C=C['LR']['C'], max_iter=500, random_state=SEED),
     'SVM': lambda: SVC(C=C['SVM']['C'], kernel=C['SVM']['kernel'], probability=True, random_state=SEED),
     'RF':  lambda: RandomForestClassifier(n_estimators=C['RF']['n_estimators'], max_depth=C['RF']['max_depth'],
                                           min_samples_leaf=C['RF']['min_samples_leaf'], random_state=SEED),
     'GB':  lambda: GradientBoostingClassifier(n_estimators=C['GB']['n_estimators'], max_depth=C['GB']['max_depth'],
                                               learning_rate=C['GB']['learning_rate'], subsample=C['GB']['subsample'],
                                               min_samples_leaf=C['GB']['min_samples_leaf'], random_state=SEED),
    }

def fit_predict(make, Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    m = make().fit(sc.transform(Xtr), ytr)
    return m.predict_proba(sc.transform(Xte))[:, 1]

def run_cv(make, X, y, seed=SEED, folds=10):
    prob = np.zeros(len(y))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        prob[te] = fit_predict(make, X[tr], y[tr], X[te])
    return prob

def metrics(y, prob):
    pred = (prob >= 0.5).astype(int)
    return dict(acc=100 * accuracy_score(y, pred), auc=roc_auc_score(y, prob), f1=f1_score(y, pred),
                mcc=matthews_corrcoef(y, pred), sens=recall_score(y, pred), spec=recall_score(y, pred, pos_label=0))

def bootstrap(y, prob, reps, seed=SEED):
    rng = np.random.RandomState(seed); a, u = [], []
    for _ in range(reps):
        i = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[i])) < 2: continue
        a.append(100 * accuracy_score(y[i], prob[i] >= 0.5)); u.append(roc_auc_score(y[i], prob[i]))
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return dict(acc_ci=q(a), auc_ci=q(u))

if __name__ == '__main__':
    X = np.load('results/features.npy'); y = np.load('results/labels.npy')
    out = {}
    for name, make in make_models().items():
        prob = run_cv(make, X, y)
        r = metrics(y, prob); r.update(bootstrap(y, prob, P['evaluation']['bootstrap_replications']))
        out[name] = r
        print(f"{name}: acc={r['acc']:.2f} {r['acc_ci']}  auc={r['auc']:.4f} {r['auc_ci']}")
    json.dump(out, open('results/cv_results.json', 'w'), indent=2)
