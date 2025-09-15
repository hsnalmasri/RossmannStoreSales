# Rossmann Store Sales

Clean, reproducible structure for the Kaggle Rossmann challenge and general retail forecasting.

## Quick Start
1) Put Kaggle files \	rain.csv\, \	est.csv\, \store.csv\ into **data/raw/**.
2) Create a venv and install requirements:
   \\\pwsh
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   \\\
3) Sanity check:
   \\\pwsh
   python .\src\rossmann\run_check.py
   \\\

## Layout
- \src/rossmann\ → python package with data loading and pipeline code  
- \
otebooks\ → Jupyter work  
- \data/raw\ → original CSVs (not committed)  
- \data/processed\ → small derived artifacts (commit if small)  
- \models\ → serialized models (ignored)

## Notes
- Keep large files out of git. Consider Git LFS or DVC later if needed.
