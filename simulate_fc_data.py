"""
Step 1: simulate 116x116 functional-connectivity matrices (version 2, re-implemented).

Each subject's matrix is a group template plus (i) a random effect per network block shared by all
edges of that block, (ii) an ADHD shift of -d * tau on the block random effect, (iii) a site offset,
(iv) a global nuisance shift and (v) symmetric edge noise. Labels enter only through (ii).
The model is deliberately simple. It checks that the pipeline recovers known effects;
it does not model real fMRI.
"""
import numpy as np, json
from network_definitions import BLOCKS

with open('simulation_parameters.json') as f:
    P = json.load(f)
GM = P['generative_model']
N_ROI = 116
D_ADHD = GM['adhd_effect_cohen_d']

def _pair_mask(a, b):
    m = np.zeros((N_ROI, N_ROI), dtype=bool)
    if b is None:
        for i in a:
            for j in a:
                if i < j: m[i, j] = m[j, i] = True
    else:
        for i in a:
            for j in b:
                if i != j: m[i, j] = m[j, i] = True
    return m

MASKS = {k: _pair_mask(*v) for k, v in BLOCKS.items()}

def template():
    b = GM['baseline_fc']
    T = np.full((N_ROI, N_ROI), b['background'])
    for name, key in [('DMN_within', 'within_DMN'), ('FPN_within', 'within_FPN'), ('SAL_within', 'within_SAL'),
                      ('DMN_FPN', 'DMN_FPN'), ('AMY_vmPFC', 'AMY_vmPFC'), ('CER_FPN', 'CER_FPN')]:
        T[MASKS[name]] = b[key]
    return T

def site_offsets(rng, n_sites, site_scale=1.0):
    s = GM['site_offset_sd_in_tau_units']
    tau = GM['between_subject_sd_of_block_mean_tau']
    return {k: np.array([rng.normal(0, site_scale * (s['base'] + s['increment_per_site'] * i) * tau)
                         for i in range(n_sites)]) for k in BLOCKS}

def simulate(seed, site, y, effect_scale=1.0, site_scale=1.0, offsets=None):
    """site, y: integer arrays (n,). Returns FC array (n,116,116) float32."""
    rng = np.random.RandomState(seed)
    n = len(y)
    tau = GM['between_subject_sd_of_block_mean_tau']
    T = template()
    if offsets is None:
        offsets = site_offsets(np.random.RandomState(seed + 10_000), int(site.max()) + 1, site_scale)
    Z = rng.randn(n, N_ROI, N_ROI) * GM['edge_noise_sd']
    fc = T[None] + (Z + Z.transpose(0, 2, 1)) / np.sqrt(2)
    for name, mask in MASKS.items():
        d = D_ADHD[name]
        u = rng.randn(n) * tau - effect_scale * d * tau * y + offsets[name][site]
        fc += u[:, None, None] * mask[None]
    fc += (rng.randn(n) * GM['global_shift_sd'])[:, None, None]
    fc = np.clip(fc, -0.99, 0.99)
    idx = np.arange(N_ROI)
    fc[:, idx, idx] = 1.0
    return fc.astype(np.float32)

def cohort_labels():
    site, y = [], []
    for s, info in enumerate(P['sites']):
        site += [s] * info['n']
        y += [1] * info['n_adhd'] + [0] * (info['n'] - info['n_adhd'])
    return np.array(site), np.array(y)

if __name__ == '__main__':
    from extract_features import extract_all
    np.random.seed(P['random_seed'])
    site, y = cohort_labels()
    fc = simulate(P['random_seed'], site, y)
    X = extract_all(fc)
    np.save('results/features.npy', X); np.save('results/labels.npy', y); np.save('results/site_ids.npy', site)
    print('Saved features', X.shape, 'ADHD', int(y.sum()), 'controls', int((1 - y).sum()))
