"""
Step 2: extract 37 graph features from each 116x116 FC matrix.
Network ROI sets come from network_definitions.py (AAL-based approximations).
"""
import numpy as np, json
from network_definitions import DMN, FPN, SAL, AMY, VMPFC, CER

NAMES = [
 'DMN-Int-Mean','DMN-Int-Std','DMN-Int-Q25','DMN-Int-Q75','DMN-Degree',
 'DMN-FPN-Mean','DMN-FPN-Std','DMN-FPN-Q25','DMN-FPN-Q75',
 'FPN-Int-Mean','FPN-Int-Std',
 'Amy-vmPFC-Mean','Amy-vmPFC-Std',
 'SAL-Int-Mean','SAL-Int-Std','SAL-Degree','SAL-FC-Variance',
 'Cereb-FPN-Mean','Cereb-FPN-Std',
 'FC-Global-Mean','FC-Global-Std','FC-Q10','FC-Q90','FC-StrongPos','FC-Negative','FC-Variance','FC-Trace-Sq',
] + [f'Eigenvalue-{k}' for k in range(1, 11)]
assert len(NAMES) == 37

DESC = {
 'DMN-Int-Mean': 'Mean of off-diagonal DMN-DMN FC', 'DMN-Int-Std': 'SD of off-diagonal DMN-DMN FC',
 'DMN-Int-Q25': '25th percentile of off-diagonal DMN-DMN FC', 'DMN-Int-Q75': '75th percentile of off-diagonal DMN-DMN FC',
 'DMN-Degree': 'Mean number of DMN partners with |FC|>0.2, per DMN node (diagonal excluded)',
 'DMN-FPN-Mean': 'Mean DMN-FPN cross-network FC', 'DMN-FPN-Std': 'SD of DMN-FPN FC',
 'DMN-FPN-Q25': '25th percentile of DMN-FPN FC', 'DMN-FPN-Q75': '75th percentile of DMN-FPN FC',
 'FPN-Int-Mean': 'Mean off-diagonal FPN-FPN FC', 'FPN-Int-Std': 'SD of off-diagonal FPN-FPN FC',
 'Amy-vmPFC-Mean': 'Mean amygdala to vmPFC-proxy FC', 'Amy-vmPFC-Std': 'SD of amygdala to vmPFC-proxy FC',
 'SAL-Int-Mean': 'Mean off-diagonal SAL-SAL FC', 'SAL-Int-Std': 'SD of off-diagonal SAL-SAL FC',
 'SAL-Degree': 'Mean number of SAL partners with |FC|>0.2, per SAL node (diagonal excluded)',
 'SAL-FC-Variance': 'Variance of off-diagonal SAL-SAL FC (a connectivity-spread measure; NOT fALFF)',
 'Cereb-FPN-Mean': 'Mean cerebellar to FPN FC', 'Cereb-FPN-Std': 'SD of cerebellar to FPN FC',
 'FC-Global-Mean': 'Mean of upper-triangle FC', 'FC-Global-Std': 'SD of upper-triangle FC',
 'FC-Q10': '10th percentile of upper-triangle FC', 'FC-Q90': '90th percentile of upper-triangle FC',
 'FC-StrongPos': 'Proportion of upper-triangle FC > 0.3', 'FC-Negative': 'Proportion of upper-triangle FC < -0.1',
 'FC-Variance': 'Variance of upper-triangle FC', 'FC-Trace-Sq': 'trace(FC @ FC) / 116',
}
for k in range(1, 11):
    DESC[f'Eigenvalue-{k}'] = f'Eigenvalue {k} of FC, ranked by absolute value (descending)'

def _off(fc, idx):
    m = fc[np.ix_(idx, idx)]
    return m[~np.eye(len(idx), dtype=bool)]

def _deg(fc, idx):
    m = np.abs(fc[np.ix_(idx, idx)]); np.fill_diagonal(m, 0)
    return float((m > 0.2).sum()) / len(idx)

def extract_features(fc):
    n = fc.shape[0]; f = []
    o = _off(fc, DMN); f += [o.mean(), o.std(), np.percentile(o, 25), np.percentile(o, 75), _deg(fc, DMN)]
    c = fc[np.ix_(DMN, FPN)].ravel(); f += [c.mean(), c.std(), np.percentile(c, 25), np.percentile(c, 75)]
    o = _off(fc, FPN); f += [o.mean(), o.std()]
    a = fc[np.ix_(AMY, VMPFC)].ravel(); f += [a.mean(), a.std()]
    o = _off(fc, SAL); f += [o.mean(), o.std(), _deg(fc, SAL), o.var()]
    c = fc[np.ix_(CER, FPN)].ravel(); f += [c.mean(), c.std()]
    ut = fc[np.triu_indices(n, 1)]
    f += [ut.mean(), ut.std(), np.percentile(ut, 10), np.percentile(ut, 90), float((ut > 0.3).mean()),
          float((ut < -0.1).mean()), ut.var(), np.trace(fc @ fc) / n]
    ev = np.linalg.eigvalsh(fc.astype(np.float64))
    f += list(ev[np.argsort(-np.abs(ev))][:10])
    return np.array(f, dtype=np.float64)

def extract_all(fcs):
    return np.stack([extract_features(m) for m in fcs])

if __name__ == '__main__':
    json.dump([{'id': f'F{i+1:02d}', 'name': n, 'definition': DESC[n]} for i, n in enumerate(NAMES)],
              open('feature_definitions.json', 'w'), indent=2)
    print('Wrote feature_definitions.json')
