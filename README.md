# EFNAI technical-verification simulation (version 2)

Code for the technical checks in the revised Methods article "EFNAI: a proposed explainable federated neuroimaging framework, with a pre-specified validation protocol, ..." (Anam Ashraf, British University Vietnam).

**What this is.** A small simulation that checks that the 37-feature extraction and evaluation code recovers known injected effects, plus privacy accounting and a DP-SGD utility simulation for the proposed pilot settings.
**What this is not.** It does not implement the GAT encoder, EEG encoder, alignment, fusion or federated training, and it says nothing about performance on real data.

**This version supersedes the earlier archive.** The earlier simulation was calibrated to land inside a published accuracy range and its saved outputs were inconsistent with the reported tables. Do not mix files from the two versions.

## Run

```
pip install -r requirements.txt
python run_all.py        # about 15-25 minutes on a laptop CPU; seed 42
```
Outputs: `results/*.json`, `figures/Figure1-3`, `feature_definitions.json`. `verify_results.py` runs ten consistency checks on the saved outputs and exits non-zero if any fail.

## Files

| File | Purpose |
|---|---|
| `simulation_parameters.json` | All settings. Values marked "assumption" are illustrative choices, not literature estimates. |
| `network_definitions.py` | AAL-116 ROI sets (approximate network assignments; fix in the preregistration). |
| `simulate_fc_data.py` | Generates the cohort (973 participants, 8 sites). Diagnosis enters only through block effects. |
| `extract_features.py` | The 37 features (`feature_definitions.json`). |
| `run_classifiers.py` | LR, SVM, RF, gradient boosting; 10-fold CV; bootstrap CIs for every model. |
| `evaluate_federated.py` | Leave-one-site-out testing (no DP, no real federation). |
| `known_answer_checks.py` | Oracle linear performance, label permutation, effect-size sweep. |
| `dp_accounting_utility.py` | PLD accounting, DP-SGD FedAvg utility, assumed rPDP-FL composition table. |
| `make_figures.py` | Figures 1-3. |
| `verify_results.py` | Consistency checks on saved results. |

## Known limitations
- Only the DMN effect (d = 0.42, range 0.32-0.48 as quoted in the paper) comes from the literature; other effect sizes, template values, noise and site offsets are the author's assumptions.
- The simulator and the feature extractor share the same network definitions, so the check is partly circular by design.
- The DP utility simulation uses a public reference cohort for standardisation, fixed hyper-parameters, binary ADHD only, and cannot model ASAS.
- The rPDP-FL table assumes independent Poisson admission with probability p; it is not a derivation for the published algorithm.
- Seeds fix the results on one machine; small numerical differences are possible with other library versions.
