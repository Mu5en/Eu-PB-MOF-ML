# Eu-PB-MOF-ML

Analysis code accompanying the manuscript

> Machine learning-assisted dual-mode fluorometric/R/B colorimetric sensing of quinolone antibiotics using a bi-ligand Eu-MOF probe.
> Submitted to *Talanta*, 2026.

This repository contains the Python scripts used to produce all machine-learning results reported in the main manuscript and supporting information, including group-aware nested cross-validation, leave-one-concentration-out (LOCO) and drug-conditional LOCO (LOCO-DC), 20-repeat random-label controls, four-family feature ablation, fixed-composition binary-mixture classification and regression, and external-batch validation.

## Repository contents

| File | Purpose |
|---|---|
| `fluorescence_ml_si_tables_optimized.py` | Single-component preprocessing, feature extraction (25 ratios + 16 SG derivatives + 18 peak-shape descriptors + 8 PCA scores), PLS-DA / Random Forest / SVM-RBF classification under group-aware nested CV, LOCO/LOCO-DC, random-label control, feature ablation, external-batch prediction. Reproduces Tables 2, S5–S8, S11. |
| `mixture_antibiotic_ml_table3_rf_v2_fixed.py` | 45-spectrum binary-mixture pipeline: 15-composition classification (leave-one-replicate-out), pair classification (internal ratios only), mole-fraction regression (leave-one-composition-out). Reproduces Table 3 and Table S10. |
| `requirements.txt` | Python dependency pins. |
| `LICENSE` | MIT License. |

## Data

The raw fluorescence spectra and extracted feature matrices used by these scripts are **not stored in this repository**. They are deposited on Zenodo and should be downloaded separately:

> Data S1, Zenodo, DOI: **10.5281/zenodo.XXXXXXX**

After downloading, place the data files in a `./data/` folder next to the scripts (or edit the `DATA_DIR` variable at the top of each script).

## Software environment

- Python 3.10 (3.10 ≤ version < 3.12)
- scikit-learn 1.3.x
- SciPy 1.11.x
- NumPy ≥ 1.24, pandas ≥ 2.0, matplotlib ≥ 3.7, joblib ≥ 1.3

Exact versions tested: Python 3.10.13, scikit-learn 1.3.2, SciPy 1.11.4.

## Quick start

```bash
# 1. Clone this repository
git clone https://github.com/<your_handle>/Eu-PB-MOF-ML.git
cd Eu-PB-MOF-ML

# 2. Create a clean environment (recommended)
python -m venv venv
source venv/bin/activate              # macOS / Linux
# venv\Scripts\activate               # Windows PowerShell

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download data from Zenodo (DOI above) and unzip into ./data/

# 5. Reproduce the single-component ML tables and figures
python fluorescence_ml_si_tables_optimized.py

# 6. Reproduce the mixture tables
python mixture_antibiotic_ml_table3_rf_v2_fixed.py
```

Each script writes its outputs (printed metrics, CSV summary, plots) to the working directory. Random seeds are fixed inside the scripts; running on a different machine with the listed library versions should produce identical numerical results within floating-point tolerance.

## Saved model

The saved final classifier bundle (`final_qn_classifier_bundle.joblib`) used for the external-batch prediction in Table S11 is **not released here**, because the model is being reused in ongoing follow-up work. It is available from the corresponding author upon reasonable request, as stated in Table S15 of the manuscript SI.

## License

The code is released under the MIT License (see `LICENSE`).  
The data on Zenodo is released under CC BY 4.0.

## Citation

If you use this code or data, please cite the manuscript (full citation will be inserted upon acceptance) and the Zenodo data deposit.
