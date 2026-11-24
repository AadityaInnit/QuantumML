"""Reproduce all experiments, figures and executed notebooks from repository root."""
from pathlib import Path
import subprocess
import sys
root = Path(__file__).resolve().parents[1]
for args in [
    ["scripts/run_benchmark.py"],
    ["scripts/run_benchmark.py", "--dataset", "circles"],
    ["scripts/run_concentration.py"],
    ["scripts/run_noise.py", "--n-train", "40", "--n-test", "20"],
    ["scripts/make_figures.py"],
    ["scripts/run_engineered.py", "--stage", "evaluate", "--overwrite"],
    ["scripts/report_engineered.py"],
    ["scripts/build_notebooks.py", "--execute"],
]:
    subprocess.run([sys.executable, *args], cwd=root, check=True)
