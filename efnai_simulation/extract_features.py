"""
EFNAI Simulation — Step 2: Extract 37 Graph Features
=====================================================
Extracts all 37 features specified in Table 3 of the paper from each
116×116 FC matrix. Features are z-score standardised within each
cross-validation fold (see run_classifiers.py).

Feature groups:
  F01-F05  DMN intra-network statistics (ROIs 67-90)
  F06-F09  DMN-FPN cross-network statistics
  F10-F11  FPN intra-network statistics (ROIs 1-20)
  F12-F13  Amygdala-vmPFC statistics (ROIs 41-42 vs 1-10)
  F14-F17  Salience network statistics (ROIs 21-40)
  F18-F19  Cerebellar-frontal statistics (ROIs 91-116 vs 1-20)
  F20-F27  Global FC statistics (full upper triangle)
  F28-F37  Top-10 eigenvalues of FC matrix (graph spectral features)
"""

import numpy as np

# Network ROI indices (0-indexed)
DMN  = list(range(66, 90))
FPN  = list(range(0,  20))
SAL  = list(range(20, 40))
AMY  = [40, 41]
VMFP = list(range(0,  10))
CER  = list(range(90, 116))

FEATURE_NAMES = [
    # DMN intra-network (F01-F05)
    'DMN-Int-Mean', 'DMN-Int-Std', 'DMN-Int-Q25', 'DMN-Int-Q75', 'DMN-Degree',
    # DMN-FPN cross-network (F06-F09)
    'DMN-FPN-Mean', 'DMN-FPN-Std', 'DMN-FPN-Q25', 'DMN-FPN-Q75',
    # FPN intra-network (F10-F11)
    'FPN-Int-Mean', 'FPN-Int-Std',
    # Amygdala-vmPFC (F12-F13)
    'Amy-vmPFC-Mean', 'Amy-vmPFC-Std',
    # Salience network (F14-F17)
    'SAL-Int-Mean', 'SAL-Int-Std', 'SAL-Degree', 'fALFF-SAL-Proxy',
    # Cerebellar-frontal (F18-F19)
    'Cereb-Front-Mean', 'Cereb-Front-Std',
    # Global FC (F20-F27)
    'FC-Global-Mean', 'FC-Global-Std', 'FC-Q10', 'FC-Q90',
    'FC-StrongPos', 'FC-Negative', 'FC-Variance', 'FC-Trace-Norm',
    # Spectral (F28-F37)
    'Eigenvalue-1', 'Eigenvalue-2', 'Eigenvalue-3', 'Eigenvalue-4',
    'Eigenvalue-5', 'Eigenvalue-6', 'Eigenvalue-7', 'Eigenvalue-8',
    'Eigenvalue-9', 'Eigenvalue-10',
]
assert len(FEATURE_NAMES) == 37, f"Expected 37 features, got {len(FEATURE_NAMES)}"


def _submat(fc, rows, cols):
    return fc[np.ix_(rows, cols)]


def _intra_stats(fc, idx):
    """Extract mean, std, Q25, Q75 of off-diagonal elements."""
    m = _submat(fc, idx, idx).copy()
    np.fill_diagonal(m, np.nan)
    flat = m[~np.isnan(m)]
    return (np.mean(flat), np.std(flat),
            np.percentile(flat, 25), np.percentile(flat, 75))


def extract_features(fc: np.ndarray) -> np.ndarray:
    """
    Extract all 37 features from a single 116×116 FC matrix.

    Parameters
    ----------
    fc : np.ndarray, shape (116, 116)

    Returns
    -------
    features : np.ndarray, shape (37,)
    """
    n = fc.shape[0]
    features = []

    # F01-F05: DMN intra-network
    dm, ds, dq25, dq75 = _intra_stats(fc, DMN)
    dmn_deg = float((np.abs(_submat(fc, DMN, DMN)) > 0.2).sum()) / len(DMN)
    features.extend([dm, ds, dq25, dq75, dmn_deg])

    # F06-F09: DMN-FPN cross-network
    cross = _submat(fc, DMN, FPN)
    features.extend([np.mean(cross), np.std(cross),
                     np.percentile(cross, 25), np.percentile(cross, 75)])

    # F10-F11: FPN intra-network
    fm, fs, _, _ = _intra_stats(fc, FPN)
    features.extend([fm, fs])

    # F12-F13: Amygdala-vmPFC
    av = _submat(fc, AMY, VMFP)
    features.extend([np.mean(av), np.std(av)])

    # F14-F17: Salience network
    sm, ss, sq25, sq75 = _intra_stats(fc, SAL)
    sal_deg = float((np.abs(_submat(fc, SAL, SAL)) > 0.2).sum()) / len(SAL)
    sal_proxy = np.nanvar(_submat(fc, SAL, SAL))
    features.extend([sm, ss, sal_deg, sal_proxy])

    # F18-F19: Cerebellar-frontal
    cf = _submat(fc, CER, FPN)
    features.extend([np.mean(cf), np.std(cf)])

    # F20-F27: Global FC (upper triangle)
    ut = fc[np.triu_indices(n, 1)]
    features.extend([
        np.mean(ut), np.std(ut),
        np.percentile(ut, 10), np.percentile(ut, 90),
        float((ut > 0.3).mean()), float((ut < -0.1).mean()),
        np.var(ut), np.trace(fc @ fc) / n,
    ])

    # F28-F37: Top-10 eigenvalues (graph spectral)
    eigs = np.sort(np.abs(np.linalg.eigvalsh(fc)))[::-1][:10]
    features.extend(eigs.tolist())

    return np.array(features, dtype=np.float64)


def extract_all(fc_matrices: np.ndarray) -> np.ndarray:
    """Extract features for all subjects."""
    n = fc_matrices.shape[0]
    X = np.zeros((n, 37), dtype=np.float64)
    for i in range(n):
        X[i] = extract_features(fc_matrices[i])
        if (i + 1) % 100 == 0:
            print(f"  Extracted features: {i+1}/{n}")
    return X


if __name__ == '__main__':
    import os; os.makedirs('results', exist_ok=True)
    fc_matrices = np.load('results/fc_matrices.npy')
    print(f"Extracting features from {fc_matrices.shape[0]} FC matrices...")
    X = extract_all(fc_matrices)
    np.save('results/features.npy', X)
    print(f"Features saved: shape={X.shape}")
    print(f"Feature names: {FEATURE_NAMES}")
