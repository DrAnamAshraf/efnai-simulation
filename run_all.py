"""Runs the whole pipeline in order. Fixed seed (42). Takes roughly 15-25 minutes on a laptop CPU."""
import subprocess, sys, os
os.makedirs('results', exist_ok=True); os.makedirs('figures', exist_ok=True)
for step in ['extract_features.py', 'simulate_fc_data.py', 'run_classifiers.py', 'evaluate_federated.py',
             'known_answer_checks.py', 'dp_accounting_utility.py', 'make_figures.py', 'verify_results.py']:
    print(f'\n=== {step} ===', flush=True)
    subprocess.run([sys.executable, step], check=True)
