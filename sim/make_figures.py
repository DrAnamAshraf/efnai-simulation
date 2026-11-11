"""Figures 1-3 from saved results (Figure 1 is a schematic; Figures 2-3 are drawn from results/*.json)."""
import json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
cv = json.load(open('results/cv_results.json')); ka = json.load(open('results/known_answer.json'))
dp = json.load(open('results/dp_utility.json'))

def fig1():
    fig, ax = plt.subplots(figsize=(10, 12)); ax.set_xlim(0, 100); ax.set_ylim(-6, 120); ax.axis('off')
    def box(x, y, w, h, title, body, c):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.5', fc=c + '22', ec=c, lw=1.6))
        ax.text(x + w / 2, y + h - 3.2, title, ha='center', va='top', fontsize=10.5, weight='bold', color=c)
        ax.text(x + w / 2, y + h - 8.2, body, ha='center', va='top', fontsize=8.3, color='#333', linespacing=1.4)
    def arr(x1, y1, x2, y2): ax.annotate('', (x2, y2), (x1, y1), arrowprops=dict(arrowstyle='-|>', color='#555', lw=1.4))
    ax.text(50, 117, 'Figure 1. Proposed EFNAI pipeline (design only; nothing shown here has been implemented)',
            ha='center', fontsize=11.5, weight='bold')
    B, G, O, Pu, R, Gr = '#2a7ca3', '#1f9d55', '#d9731a', '#7b3fa0', '#c0392b', '#6b7b7b'
    box(3, 98, 32, 14, 'rs-fMRI acquisition', '3T, TR 2 s, 10 min\n(300 volumes)', B)
    box(65, 98, 32, 14, 'EEG acquisition', '19 channels, 10-20 system\n256 Hz, 5 min resting state', G)
    box(3, 76, 32, 16, 'fMRI preprocessing', 'fMRIPrep, MNI152\nFD > 0.2 mm frames censored\n(0.5 mm fallback), AAL-116 FC graph', B)
    box(65, 76, 32, 16, 'EEG preprocessing', 'ICA artefact rejection\nWelch PSD, 5 s windows, 50% overlap', G)
    box(39, 76, 22, 16, 'Harmonisation', 'd-ComBat\n(planned)', O)
    arr(19, 98, 19, 92.8); arr(81, 98, 81, 92.8)
    box(3, 54, 32, 16, 'GAT encoder', '2 layers, 4 heads\n116 node tokens -> 64-d', B)
    box(65, 54, 32, 16, 'BiLSTM-Transformer', 'T tokens x 64-d\nTBR / FAA / beta coherence', G)
    box(39, 54, 22, 16, 'Alignment', 'Tier A: InfoNCE\nTier B: lagged\ncross-correlation', Pu)
    arr(19, 76, 19, 70.8); arr(81, 76, 81, 70.8)
    box(22, 32, 56, 15, 'Token-level cross-modal attention', 'bidirectional, 4 heads; quality-aware gating (FD, EEG impedance)\n-> 128-d joint embedding', Pu)
    arr(50, 54, 50, 47.8)
    box(30, 14, 40, 13, 'Four-class head', 'ADHD | anxiety | comorbid | ASAS (hypothesis)\nclass-weighted cross-entropy', O)
    arr(50, 32, 50, 27.8)
    box(3, -4, 94, 14, 'Federated protocol (proposed)', 'FedAvg with DP-SGD (C = 1.0, delta = 1e-5)\nsigma 5-10 gives epsilon of about 10-25 at pilot settings; epsilon = 5 needs sigma of about 18', Gr)
    arr(50, 14, 50, 10.8)
    fig.savefig('figures/Figure1_architecture.png', dpi=200, bbox_inches='tight', facecolor='white'); plt.close(fig)

def fig2():
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4)); names = ['LR', 'SVM', 'RF', 'GB']; cols = ['#2a7ca3'] * 4
    for ax, key, ci, lab, ora in [(axs[0], 'acc', 'acc_ci', 'Accuracy (%)', ka['oracle']['acc']), (axs[1], 'auc', 'auc_ci', 'AUC', ka['oracle']['auc'])]:
        v = [cv[n][key] for n in names]; lo = [cv[n][key] - cv[n][ci][0] for n in names]; hi = [cv[n][ci][1] - cv[n][key] for n in names]
        ax.bar(names, v, yerr=[lo, hi], color=cols, capsize=4)
        ax.axhline(50 if key == 'acc' else 0.5, color='grey', ls=':', label='chance')
        ax.axhline(ora, color='#d9731a', ls='--', label='oracle linear (large sample)')
        ax.set_ylabel(lab); ax.set_ylim(45 if key == 'acc' else 0.45, 66 if key == 'acc' else 0.70); ax.legend(frameon=False, fontsize=8)
        ax.set_title(('Accuracy' if key == 'acc' else 'AUC') + ', 10-fold CV, n = 973\n(bars: 95% bootstrap CI)', fontsize=10)
    sc = sorted(ka['sweep'], key=float)
    for m, c in [('LR', '#2a7ca3'), ('GB', '#1f9d55')]:
        mu = [ka['sweep'][s][m]['auc_mean'] for s in sc]; sd = [ka['sweep'][s][m]['auc_sd'] for s in sc]
        axs[2].errorbar([float(s) for s in sc], mu, yerr=sd, marker='o', capsize=3, label=m, color=c)
    axs[2].axhline(0.5, color='grey', ls=':'); axs[2].set_xlabel('Multiplier on injected effect sizes (1 = base setting)')
    axs[2].set_ylabel('CV AUC'); axs[2].set_title('Known-answer check: AUC vs injected effect\n(mean +/- SD, 5 simulated cohorts)', fontsize=10)
    axs[2].legend(frameon=False)
    for a, t in zip(axs, 'abc'): a.text(-0.12, 1.05, t, transform=a.transAxes, weight='bold', fontsize=13)
    plt.tight_layout(); fig.savefig('figures/Figure2_technical_verification.png', dpi=200, facecolor='white'); plt.close(fig)

def fig3():
    S = dp['utility']; sg = [str(s) for s in [1.1, 2, 3, 5, 7.5, 10, 18.1, 30, 50, 75.5]]
    eps = [dp['epsilon'][s] for s in sg]; fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, k, sk, lab in [(axs[0], 'acc_mean', 'acc_sd', 'Test accuracy (%)'), (axs[1], 'auc_mean', 'auc_sd', 'Test AUC')]:
        mu = [S['s' + s][k] for s in sg]; sd = [S['s' + s][sk] for s in sg]
        ax.errorbar(eps, mu, yerr=sd, marker='o', capsize=3, color='#c0392b', label='DP-SGD FedAvg (mean +/- SD)')
        ax.axhline(S['fedavg_nonprivate'][k], color='#2a7ca3', ls='--', label='non-private FedAvg')
        ax.axhline(S['central_LR'][k], color='#1f9d55', ls=':', label='centralised logistic regression')
        ax.axhline(50 if k == 'acc_mean' else 0.5, color='grey', ls=':', lw=0.8)
        ax.set_xscale('log'); ax.invert_xaxis(); ax.set_xlabel('epsilon (delta = 1e-5), smaller = stronger privacy'); ax.set_ylabel(lab)
        for e, m, s in zip(eps, mu, sg):
            if s in ('1.1', '5', '10', '18.1', '75.5'): ax.annotate(f'sigma={s}', (e, m), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=7.5)
    axs[0].legend(frameon=False, fontsize=8, loc='lower left')
    fig.suptitle(f'Figure 3. Privacy-utility curve on simulated FC features (3 sites x 50, {dp["replicates"]} replicates)', fontsize=11)
    plt.tight_layout(); fig.savefig('figures/Figure3_privacy_utility.png', dpi=200, facecolor='white'); plt.close(fig)

if __name__ == '__main__':
    fig1(); fig2(); fig3(); print('figures written')
