"""
EFNAI Internal Consistency Validation — Master Script
======================================================
Author: Anam Ashraf (anamashraf100@yahoo.com)
ORCID:  https://orcid.org/0000-0001-7584-9824
Affiliation: Faculty of School of Computing and Innovative Technology,
             British University Vietnam, Hanoi, Vietnam

Paper: An Explainable Federated Neuroimaging AI Architecture for Automated
       ADHD Detection and Differential Diagnosis from Anxiety Disorders
       in University Students. Royal Society Open Science (submitted).

Usage:
    python run_all.py

Produces:
    results/results.json         — all metrics (matches Tables 4 & 5 of paper)
    results/validation_results.png — Figure 2 of paper

Expected output (fixed seed=42):
    GFC accuracy:  74.00% (95% CI: 71.2–76.6%)
    GFC AUC:       0.8239 (95% CI: 0.797–0.849)
    Fed mean acc:  74.74% ± 2.31%

IMPORTANT: This simulation is circular by design. It confirms only that the
evaluation pipeline is correctly parameterised — it does NOT estimate
performance of the proposed EFNAI architecture on real data.
"""

import os
os.makedirs('results', exist_ok=True)

print("=" * 60)
print("EFNAI Internal Consistency Validation")
print("Author: Anam Ashraf | British University Vietnam")
print("=" * 60)
print()

# Run the complete simulation (ported from validation_experiment.py)
exec(open('simulate_fc_data.py').read())
