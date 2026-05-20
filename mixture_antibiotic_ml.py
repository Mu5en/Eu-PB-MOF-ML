#!/usr/bin/env python3
"""
Binary-mixture ML pipeline (reference-style figures + manuscript Table 3).

Generates two artefact families from a wide-format fluorescence-mixture CSV:

Top row of each figure:  PCA or t-SNE for each binary mixture system.
Bottom row of each figure: composition-ratio confusion matrices.

Binary systems
--------------
    CIP:NFLX   = C:N
    CIP:LVLX   = C:L
    LVLX:NFLX  = L:N

Class labels inside each system are detected from the input data. For
C1:N4 / C2:N3 / C3:N2 / C4:N1 plus pure endpoints, labels are
0:100%, 20:80%, 40:60%, 60:40%, 80:20%, 100:0%.

REAL3 vs AUG5_PREVIEW
---------------------
* REAL3 uses only real 3 replicates and is the conservative dataset.
* AUG5_PREVIEW augments each composition from 3 to 5 replicates by
  interpolation + low-amplitude noise. These outputs are previews of the
  final figure style only; the synthetic rep4/rep5 spectra MUST NOT be
  claimed as independent experimental replicates in a publication.

Classifiers
-----------
Reference-style binary figures: ratio-only features + PLS-DA / RandomForest / SVM-RBF.
Manuscript Table 3 also includes mole-fraction regression
(PLSR / SVR-RBF / RandomForestRegressor, leave-one-composition-out) and
internal-mixed-ratio pair classification.

Input CSV format (wide)
-----------------------
    nm, C1:L4_rep1, C1:L4_rep2, ..., C5_rep1, N5_rep1, L5_rep1, ...

Usage
-----
    python mixture_pipeline.py --data FL_mix.csv --outdir ./results_mixture
"""

import argparse
import inspect
import json
import os
import re
import warnings
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

warnings.filterwarnings("ignore")


# =============================================================================
# 0. Default configuration (paths and flags overridden by CLI in main())
# =============================================================================
CONFIG = {
    "DATA_PATH": None,
    "OUTDIR":    Path("./results_mixture"),

    # Wavelength crop for emission plots / features.
    "CROP_MIN": 380.0,
    "CROP_MAX": 720.0,

    # Gentle SG smoothing (off by default to avoid spectral distortion).
    "USE_SG_SMOOTH": False,
    "SG_WINDOW": 11,
    "SG_POLY":   2,

    # Synthetic augmentation from 3 to 5 replicates (preview only).
    "MAKE_AUG5_PREVIEW": True,
    "TARGET_REPS":       5,
    "AUG_NOISE_SCALE":   0.15,
    "AUG_RANDOM_STATE":  42,

    # Plotting / models.
    "PANEL_DPI":      300,
    "EXPORT_FORMATS": ["png"],
    "CLASSIFIER_FOR_MAIN_FIGURE": "PLS-DA",   # PLS-DA / RandomForest / SVM-RBF
    "RUN_BOTH_CLASSIFIERS":       True,

    # Manuscript Table 3 settings.
    "RUN_TABLE3_ANALYSIS":  True,
    "TABLE3_FEATURE_SET":   "ratio_only",     # ratio_only / all_engineered
    "PLSDA_N_COMPONENTS":   8,
    "RF_N_ESTIMATORS":      500,
    "RF_RANDOM_STATE":      42,
    "SVR_C":                10,
    "SVR_GAMMA":            "scale",
    "PLSR_N_COMPONENTS":    8,
    "TSNE_PERPLEXITY_REAL3": 4,
    "TSNE_PERPLEXITY_AUG5":  5,
    "TSNE_RANDOM_STATE":     42,

    # Optional: zip the entire output folder after the run.
    "MAKE_ZIP": False,
}

PAIR_DEFS = [
    ("C:N", "CIP:NFLX",  "CIP_NFLX",  "C", "N"),
    ("C:L", "CIP:LVLX",  "CIP_LVLX",  "C", "L"),
    ("L:N", "LVLX:NFLX", "LVLX_NFLX", "L", "N"),
]
DRUG_NAME = {"C": "CIP", "N": "NFLX", "L": "LVLX"}


def ratio_sort_key(label: str) -> int:
    return int(str(label).split(":")[0])


def get_class_order(labels) -> List[str]:
    return sorted(pd.unique(labels).tolist(), key=ratio_sort_key)


def get_class_colors(class_order: List[str]) -> Dict[str, tuple]:
    cmap = plt.get_cmap("tab10")
    return {cls: cmap(i % 10) for i, cls in enumerate(class_order)}


# =============================================================================
# 1. Basic utilities
# =============================================================================
def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_div(a, b, eps=1e-12):
    return np.asarray(a) / (np.asarray(b) + eps)


def nearest_index(wavelengths: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(wavelengths - target)))


def intensity_at(X: np.ndarray, wavelengths: np.ndarray, target: float) -> np.ndarray:
    return X[:, nearest_index(wavelengths, target)]


def area_between(X: np.ndarray, wavelengths: np.ndarray, lo: float, hi: float) -> np.ndarray:
    mask = (wavelengths >= lo) & (wavelengths <= hi)
    x = wavelengths[mask]
    y = X[:, mask]
    if y.shape[1] < 2:
        return np.zeros(X.shape[0])
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x, axis=1)
    return np.trapz(y, x, axis=1)


def centroid_between(X: np.ndarray, wavelengths: np.ndarray, lo: float, hi: float) -> np.ndarray:
    mask = (wavelengths >= lo) & (wavelengths <= hi)
    x = wavelengths[mask]
    y = X[:, mask].copy()
    y = y - np.nanmin(y, axis=1, keepdims=True)
    denom = np.sum(y, axis=1) + 1e-12
    return np.sum(y * x[None, :], axis=1) / denom


# =============================================================================
# 2. Load and parse sample metadata
# =============================================================================
def parse_sample_name(col: str) -> Dict:
    """Parse names like C1:L4_rep1, C5_rep2, N5_rep3."""
    sample_id = str(col).strip()
    m = re.match(r"^(.+?)_rep(\d+)\s*$", sample_id)
    if not m:
        raise ValueError(
            f"Cannot parse sample column name: {sample_id}. "
            "Expected e.g. C1:L4_rep1 or C5_rep1."
        )

    comp = m.group(1).strip()
    rep = int(m.group(2))
    f = {"C": 0.0, "N": 0.0, "L": 0.0}

    if ":" in comp:
        left, right = comp.split(":")
        m1 = re.match(r"^([CNL])(\d+)$", left.strip())
        m2 = re.match(r"^([CNL])(\d+)$", right.strip())
        if not (m1 and m2):
            raise ValueError(f"Cannot parse mixture composition: {comp}")
        a1, r1 = m1.group(1), int(m1.group(2))
        a2, r2 = m2.group(1), int(m2.group(2))
        total = r1 + r2
        f[a1] = r1 / total
        f[a2] = r2 / total
        pair = f"{a1}:{a2}"
        composition_label = f"{a1}{r1}{a2}{r2}"
        ratio_raw = f"{r1}:{r2}"
    else:
        m0 = re.match(r"^([CNL])(\d+)$", comp)
        if not m0:
            raise ValueError(f"Cannot parse pure composition: {comp}")
        a, _ = m0.group(1), int(m0.group(2))
        f[a] = 1.0
        pair = "pure"
        composition_label = a
        ratio_raw = "5:0"

    return {
        "sample_id":         sample_id,
        "composition_raw":   comp,
        "composition_label": composition_label,
        "pair":              pair,
        "replicate":         rep,
        "ratio_raw":         ratio_raw,
        "f_C":               f["C"],
        "f_N":               f["N"],
        "f_L":               f["L"],
        "is_synthetic":      False,
        "source_type":       "real",
    }


def load_wide_csv(path) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_csv(str(path), encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    if "nm" not in df.columns:
        df = df.rename(columns={df.columns[0]: "nm"})

    wavelengths = pd.to_numeric(df["nm"], errors="coerce").to_numpy()
    valid_wl = np.isfinite(wavelengths)
    df = df.loc[valid_wl].copy()
    wavelengths = wavelengths[valid_wl].astype(float)

    sample_cols = [c for c in df.columns if c != "nm"]
    X = df[sample_cols].apply(pd.to_numeric, errors="coerce").to_numpy().T

    valid_samples = np.all(np.isfinite(X), axis=1)
    if not np.all(valid_samples):
        print(f"[load] dropped {np.sum(~valid_samples)} sample columns "
              "containing NaN/inf.")
        X = X[valid_samples]
        sample_cols = [c for c, keep in zip(sample_cols, valid_samples) if keep]

    meta = pd.DataFrame([parse_sample_name(c) for c in sample_cols])
    return meta, wavelengths, X


def preprocess_spectra(X_raw: np.ndarray, wl_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = (wl_raw >= CONFIG["CROP_MIN"]) & (wl_raw <= CONFIG["CROP_MAX"])
    wl = wl_raw[mask].astype(float)
    X = X_raw[:, mask].astype(float)

    if CONFIG["USE_SG_SMOOTH"]:
        win = int(CONFIG["SG_WINDOW"])
        if win % 2 == 0:
            win += 1
        if win >= X.shape[1]:
            win = X.shape[1] - 1 if X.shape[1] % 2 == 0 else X.shape[1]
        if win >= 5:
            X = savgol_filter(X, window_length=win,
                              polyorder=CONFIG["SG_POLY"], axis=1)
    return X, wl


# =============================================================================
# 3. Synthetic augmentation from 3 to 5 replicates (preview only)
# =============================================================================
def augment_to_target_reps(meta: pd.DataFrame, X: np.ndarray,
                           target_reps: int = 5) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Conservative visual augmentation.

    For each composition, generate synthetic spectra by interpolating two real
    replicates and adding small wavelength-wise noise based on the observed SD.
    Synthetic rows are clearly marked in metadata (is_synthetic=True).
    """
    rng = np.random.default_rng(CONFIG["AUG_RANDOM_STATE"])
    meta_rows = [r.to_dict() for _, r in meta.iterrows()]
    X_rows = [X[i].copy() for i in range(X.shape[0])]

    for comp in sorted(meta["composition_raw"].unique()):
        idx = meta.index[meta["composition_raw"] == comp].to_numpy()
        current_reps = len(idx)
        if current_reps >= target_reps:
            continue

        spectra = X[idx]
        mu = spectra.mean(axis=0)
        sd = spectra.std(axis=0, ddof=1) if current_reps > 1 else np.zeros(X.shape[1])
        existing_reps = set(meta.loc[idx, "replicate"].astype(int).tolist())

        for rep in range(1, target_reps + 1):
            if rep in existing_reps:
                continue
            if spectra.shape[0] >= 2:
                a, b = rng.choice(spectra.shape[0], size=2, replace=False)
                alpha = rng.uniform(0.35, 0.65)
                syn = alpha * spectra[a] + (1 - alpha) * spectra[b]
            else:
                syn = mu.copy()
            syn = syn + rng.normal(0, CONFIG["AUG_NOISE_SCALE"] * (sd + 1e-12),
                                   size=syn.shape)

            base = meta.loc[idx[0]].to_dict()
            base["replicate"] = rep
            base["sample_id"] = f"{base['composition_raw']}_rep{rep}_synthetic"
            base["is_synthetic"] = True
            base["source_type"] = "synthetic_from_real_reps"
            meta_rows.append(base)
            X_rows.append(syn)

    meta_aug = pd.DataFrame(meta_rows).reset_index(drop=True)
    X_aug = np.vstack(X_rows)
    return meta_aug, X_aug


# =============================================================================
# 4. Feature extraction (ratio-only core + supporting features)
# =============================================================================
def extract_features(X: np.ndarray, wl: np.ndarray) -> pd.DataFrame:
    feats = {}
    for p in [446, 454, 492, 535, 556, 592, 617, 700]:
        feats[f"I{p}"] = intensity_at(X, wl, p)

    windows = {
        "A440_452": (440, 452),
        "A448_462": (448, 462),
        "A485_500": (485, 500),
        "A580_605": (580, 605),
        "A605_630": (605, 630),
        "A690_710": (690, 710),
    }
    for name, (lo, hi) in windows.items():
        feats[name] = area_between(X, wl, lo, hi)

    feats["centroid_430_510"] = centroid_between(X, wl, 430, 510)

    ratio_pairs = [
        ("I446", "I617"), ("I454", "I617"), ("I492", "I617"),
        ("I446", "I592"), ("I454", "I592"), ("I492", "I592"),
        ("I446", "I454"), ("I446", "I492"), ("I454", "I492"),
        ("I454", "I446"), ("I492", "I446"), ("I492", "I454"),
        ("I535", "I617"), ("I556", "I617"), ("I592", "I617"), ("I700", "I617"),
        ("A440_452", "A605_630"), ("A448_462", "A605_630"), ("A485_500", "A605_630"),
        ("A440_452", "A485_500"), ("A448_462", "A485_500"),
        ("A580_605", "A605_630"), ("A690_710", "A605_630"),
    ]
    for a, b in ratio_pairs:
        feats[f"{a}/{b}"] = safe_div(feats[a], feats[b])

    feats["I454-I446"]         = feats["I454"] - feats["I446"]
    feats["I492-I454"]         = feats["I492"] - feats["I454"]
    feats["I492-I446"]         = feats["I492"] - feats["I446"]
    feats["(I454-I446)/I617"]  = safe_div(feats["I454-I446"], feats["I617"])
    feats["(I492-I454)/I617"]  = safe_div(feats["I492-I454"], feats["I617"])
    feats["I446/(I446+I492)"]  = safe_div(feats["I446"], feats["I446"] + feats["I492"])
    feats["I454/(I454+I492)"]  = safe_div(feats["I454"], feats["I454"] + feats["I492"])

    out = pd.DataFrame(feats)
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.fillna(out.median(numeric_only=True))
    return out


def ratio_feature_columns(feats: pd.DataFrame) -> List[str]:
    """Ratio-only feature set used for PCA / t-SNE and the binary figures."""
    return [c for c in feats.columns if "/" in c or c.startswith("(")]


# =============================================================================
# 5. Binary pair subsetting and class labels
# =============================================================================
def get_fraction(meta_row: pd.Series, drug_code: str) -> float:
    return float(meta_row[f"f_{drug_code}"])


def class_label_for_pair(row: pd.Series, first: str, second: str) -> str:
    f_first = get_fraction(row, first)
    pct_first = int(round(f_first * 100))
    pct_second = 100 - pct_first
    return f"{pct_first}:{pct_second}%"


def subset_for_binary_pair(meta: pd.DataFrame, X: np.ndarray, feats: pd.DataFrame,
                           pair_code: str, first: str, second: str):
    pure_first  = (meta["pair"] == "pure") & (meta[f"f_{first}"]  == 1.0)
    pure_second = (meta["pair"] == "pure") & (meta[f"f_{second}"] == 1.0)
    mixed_pair  = meta["pair"] == pair_code
    mask = (mixed_pair | pure_first | pure_second).to_numpy()

    sub_meta  = meta.loc[mask].reset_index(drop=True).copy()
    sub_X     = X[mask]
    sub_feats = feats.loc[mask].reset_index(drop=True).copy()
    sub_meta["class_ratio"]    = sub_meta.apply(
        lambda r: class_label_for_pair(r, first, second), axis=1)
    sub_meta["pair_for_figure"] = f"{DRUG_NAME[first]}:{DRUG_NAME[second]}"
    return sub_meta, sub_X, sub_feats


# =============================================================================
# 6. Models
# =============================================================================
class PLSDA(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=3):
        self.n_components = n_components

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        Y = np.zeros((len(y), len(self.classes_)))
        for i, label in enumerate(y):
            Y[i, class_to_idx[label]] = 1.0
        ncomp = min(self.n_components, X.shape[0] - 1, X.shape[1], len(self.classes_))
        ncomp = max(1, ncomp)
        self.model_ = PLSRegression(n_components=ncomp)
        self.model_.fit(X, Y)
        return self

    def predict(self, X):
        Yp = self.model_.predict(X)
        idx = np.argmax(Yp, axis=1)
        return self.classes_[idx]


def get_model(model_name: str):
    if model_name == "PLS-DA":
        return PLSDA(n_components=CONFIG["PLSDA_N_COMPONENTS"])
    if model_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=CONFIG["RF_N_ESTIMATORS"],
            random_state=CONFIG["RF_RANDOM_STATE"],
            class_weight="balanced",
            n_jobs=-1,
        )
    if model_name == "SVM-RBF":
        return SVC(
            kernel="rbf",
            C=CONFIG["SVR_C"],
            gamma=CONFIG["SVR_GAMMA"],
            class_weight="balanced",
        )
    raise ValueError(f"Unknown model: {model_name}")


def model_file_tag(model_name: str) -> str:
    """Filesystem-safe short tag (e.g. PLS-DA -> PLSDA, SVM-RBF -> SVMRBF)."""
    return re.sub(r"[^A-Za-z0-9]+", "", str(model_name))


def get_classifier_names() -> List[str]:
    if CONFIG["RUN_BOTH_CLASSIFIERS"]:
        return ["PLS-DA", "RandomForest", "SVM-RBF"]
    return [CONFIG["CLASSIFIER_FOR_MAIN_FIGURE"]]


def get_cv_splits(y: np.ndarray, groups: np.ndarray):
    """Prefer leave-one-replicate-out. Fall back to stratified K-fold."""
    unique_groups = np.unique(groups)
    if len(unique_groups) >= 3:
        return list(LeaveOneGroupOut().split(np.zeros(len(y)), y, groups))
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    return list(cv.split(np.zeros(len(y)), y))


def cross_validated_predictions(X_feat: np.ndarray, y: np.ndarray,
                                groups: np.ndarray, model_name: str):
    splits = get_cv_splits(y, groups)
    pred_all = np.empty(len(y), dtype=object)
    rows = []

    for fold, (tr, te) in enumerate(splits, 1):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_feat[tr])
        Xte = scaler.transform(X_feat[te])
        model = get_model(model_name)
        model.fit(Xtr, y[tr])
        pred = model.predict(Xte)
        pred_all[te] = pred
        rows.append({
            "fold": fold,
            "model": model_name,
            "accuracy": accuracy_score(y[te], pred),
            "f1_macro": f1_score(y[te], pred, average="macro", zero_division=0),
            "n_test": len(te),
        })
    return pred_all, pd.DataFrame(rows)


# =============================================================================
# 7. PCA / t-SNE
# =============================================================================
def make_tsne(n_samples: int, perplexity: int, random_state: int):
    p = max(2, min(perplexity, max(2, (n_samples - 1) // 3)))
    kwargs = dict(n_components=2, perplexity=p, init="pca",
                  learning_rate="auto", random_state=random_state, method="exact")
    sig = inspect.signature(TSNE)
    if "max_iter" in sig.parameters:
        kwargs["max_iter"] = 500
    elif "n_iter" in sig.parameters:
        kwargs["n_iter"] = 500
    return TSNE(**kwargs)


def compute_pca_tsne(sub_meta: pd.DataFrame, sub_feats: pd.DataFrame,
                     dataset_tag: str, pair_stem: str, outdir: Path):
    cols = ratio_feature_columns(sub_feats)
    X_feat = sub_feats[cols].to_numpy()
    Xs = StandardScaler().fit_transform(X_feat)

    pca = PCA(n_components=2, random_state=42)
    pc = pca.fit_transform(Xs)
    pca_df = sub_meta.copy()
    pca_df["PC1"] = pc[:, 0]
    pca_df["PC2"] = pc[:, 1]
    pca_df["PC1_var_percent"] = pca.explained_variance_ratio_[0] * 100
    pca_df["PC2_var_percent"] = pca.explained_variance_ratio_[1] * 100
    pca_df.to_csv(outdir / f"plotdata_{dataset_tag}_{pair_stem}_PCA.csv",
                  index=False, encoding="utf-8-sig")

    perplexity = (CONFIG["TSNE_PERPLEXITY_AUG5"]
                  if dataset_tag.startswith("AUG5")
                  else CONFIG["TSNE_PERPLEXITY_REAL3"])
    tsne = make_tsne(len(sub_meta), perplexity, CONFIG["TSNE_RANDOM_STATE"])
    emb = tsne.fit_transform(Xs)
    tsne_df = sub_meta.copy()
    tsne_df["tSNE1"] = emb[:, 0]
    tsne_df["tSNE2"] = emb[:, 1]
    tsne_df.to_csv(outdir / f"plotdata_{dataset_tag}_{pair_stem}_tSNE.csv",
                   index=False, encoding="utf-8-sig")
    return pca_df, tsne_df


# =============================================================================
# 8. Plotting
# =============================================================================
def add_panel_label(ax, label: str):
    ax.text(-0.17, 1.12, label, transform=ax.transAxes, fontsize=18,
            fontweight="bold", va="top", ha="left")


def plot_projection_panel(ax, df: pd.DataFrame, method: str,
                          pair_title: str, class_order: List[str]):
    if method == "PCA":
        xcol, ycol = "PC1", "PC2"
        pc1_var = df["PC1_var_percent"].iloc[0]
        pc2_var = df["PC2_var_percent"].iloc[0]
        xlabel = f"PC1 ({pc1_var:.2f}%)"
        ylabel = f"PC2 ({pc2_var:.2f}%)"
    else:
        xcol, ycol = "tSNE1", "tSNE2"
        xlabel, ylabel = "t-SNE 1", "t-SNE 2"

    class_colors = get_class_colors(class_order)
    for cls in class_order:
        g = df[df["class_ratio"] == cls]
        if g.empty:
            continue
        g = g.sort_values("replicate")
        ax.plot(g[xcol], g[ycol], color=class_colors[cls], alpha=0.35, linewidth=1.0)
        ax.scatter(g[xcol], g[ycol], s=24, color=class_colors[cls],
                   edgecolor="none", label=f"Class {cls}")

    ax.set_title(pair_title, fontsize=10, pad=5)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(False)


def plot_confusion_panel(ax, cm: np.ndarray, labels: List[str], title: str):
    im = ax.imshow(cm, interpolation="nearest", cmap="RdPu",
                   vmin=0, vmax=max(1, cm.max()))
    ax.set_title(title, fontsize=10, pad=5)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("True", fontsize=9)

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                    fontsize=8,
                    color="white" if cm[i, j] > thresh else "black")
    return im


def make_combined_reference_figure(results: Dict, dataset_tag: str,
                                   method: str, model_name: str, outdir: Path):
    """2x3 figure: top row = projection, bottom row = confusion matrices."""
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 6.7), dpi=CONFIG["PANEL_DPI"])
    panel_labels = ["A", "B", "C", "D", "E", "F"]
    pair_order = ["CIP_NFLX", "CIP_LVLX", "LVLX_NFLX"]

    for col, pair_stem in enumerate(pair_order):
        pair_result = results[pair_stem]
        proj_df = pair_result["pca_df"] if method == "PCA" else pair_result["tsne_df"]
        plot_projection_panel(axes[0, col], proj_df, method,
                              pair_result["pair_title"], pair_result["class_order"])
        add_panel_label(axes[0, col], panel_labels[col])

    handles, labels = axes[0, 2].get_legend_handles_labels()
    if handles:
        axes[0, 2].legend(handles, labels, loc="center right",
                          fontsize=7, frameon=False)

    ims = []
    for col, pair_stem in enumerate(pair_order):
        pair_result = results[pair_stem]
        cm = pair_result["cm_by_model"][model_name]
        im = plot_confusion_panel(axes[1, col], cm,
                                  pair_result["class_order"],
                                  pair_result["pair_title"])
        ims.append(im)
        add_panel_label(axes[1, col], panel_labels[col + 3])

    fig.subplots_adjust(left=0.055, right=0.925, bottom=0.105, top=0.955,
                        wspace=0.28, hspace=0.42)
    cax = fig.add_axes([0.945, 0.105, 0.012, 0.36])
    cbar = fig.colorbar(ims[-1], cax=cax)
    cbar.ax.tick_params(labelsize=8)

    base = f"Fig_reference_style_{dataset_tag}_{method}_CM_{model_file_tag(model_name)}"
    for ext in CONFIG["EXPORT_FORMATS"]:
        fig.savefig(outdir / f"{base}.{ext}", bbox_inches="tight")
    plt.close(fig)


def make_standalone_projection_fig(df: pd.DataFrame, dataset_tag: str,
                                   pair_stem: str, pair_title: str,
                                   method: str, outdir: Path):
    fig, ax = plt.subplots(figsize=(4.6, 3.7), dpi=CONFIG["PANEL_DPI"])
    plot_projection_panel(ax, df, method, pair_title,
                          get_class_order(df["class_ratio"]))
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    base = f"Fig_single_{dataset_tag}_{pair_stem}_{method}"
    for ext in CONFIG["EXPORT_FORMATS"]:
        fig.savefig(outdir / f"{base}.{ext}", bbox_inches="tight")
    plt.close(fig)


def make_standalone_cm_fig(cm: np.ndarray, dataset_tag: str, pair_stem: str,
                           pair_title: str, model_name: str, outdir: Path,
                           class_order: List[str]):
    fig, ax = plt.subplots(figsize=(4.3, 3.8), dpi=CONFIG["PANEL_DPI"])
    im = plot_confusion_panel(ax, cm, class_order,
                              f"{pair_title} ({model_name})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    base = f"Fig_single_{dataset_tag}_{pair_stem}_CM_{model_file_tag(model_name)}"
    for ext in CONFIG["EXPORT_FORMATS"]:
        fig.savefig(outdir / f"{base}.{ext}", bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 9. Per-dataset analysis
# =============================================================================
def analyze_dataset(dataset_tag: str, meta: pd.DataFrame, X: np.ndarray,
                    wl: np.ndarray, outdir: Path):
    feats = extract_features(X, wl)
    feat_cols = ratio_feature_columns(feats)
    all_results = {}
    summary_rows = []
    prediction_rows = []

    meta.to_csv(outdir / f"metadata_{dataset_tag}.csv",
                index=False, encoding="utf-8-sig")
    feats_with_meta = pd.concat([meta.reset_index(drop=True),
                                  feats.reset_index(drop=True)], axis=1)
    feats_with_meta.to_csv(outdir / f"features_{dataset_tag}_by_sample.csv",
                           index=False, encoding="utf-8-sig")

    wide = pd.DataFrame(X.T, columns=meta["sample_id"].tolist())
    wide.insert(0, "nm", wl)
    wide.to_csv(outdir / f"origin_{dataset_tag}_crop_wide.csv",
                index=False, encoding="utf-8-sig")

    mean_rows = []
    for comp in sorted(meta["composition_raw"].unique()):
        idx = meta.index[meta["composition_raw"] == comp].to_numpy()
        mu = X[idx].mean(axis=0)
        sd = X[idx].std(axis=0, ddof=1) if len(idx) > 1 else np.zeros_like(mu)
        base = meta.loc[idx[0]].to_dict()
        for k, wavelength in enumerate(wl):
            mean_rows.append({**base, "wavelength_nm": wavelength,
                              "mean_intensity": mu[k], "sd_intensity": sd[k]})
    pd.DataFrame(mean_rows).to_csv(outdir / f"origin_{dataset_tag}_mean_sd_long.csv",
                                    index=False, encoding="utf-8-sig")

    model_names = get_classifier_names()

    for pair_code, pair_title, pair_stem, first, second in PAIR_DEFS:
        sub_meta, sub_X, sub_feats = subset_for_binary_pair(
            meta, X, feats, pair_code, first, second)
        pca_df, tsne_df = compute_pca_tsne(sub_meta, sub_feats,
                                            dataset_tag, pair_stem, outdir)
        make_standalone_projection_fig(pca_df, dataset_tag, pair_stem,
                                        pair_title, "PCA", outdir)
        make_standalone_projection_fig(tsne_df, dataset_tag, pair_stem,
                                        pair_title, "tSNE", outdir)

        X_feat = sub_feats[feat_cols].to_numpy()
        y = sub_meta["class_ratio"].to_numpy()
        groups = sub_meta["replicate"].astype(str).to_numpy()

        cm_by_model = {}
        for model_name in model_names:
            pred, fold_df = cross_validated_predictions(X_feat, y, groups, model_name)
            class_order = get_class_order(y)
            cm = confusion_matrix(y, pred, labels=class_order)
            cm_by_model[model_name] = cm

            tag = model_file_tag(model_name)
            cm_df = pd.DataFrame(cm, index=class_order, columns=class_order)
            cm_df.to_csv(outdir / f"confusion_matrix_{dataset_tag}_{pair_stem}_{tag}.csv",
                         encoding="utf-8-sig")
            fold_df.insert(0, "dataset", dataset_tag)
            fold_df.insert(1, "pair", pair_title)
            fold_df.to_csv(outdir / f"fold_metrics_{dataset_tag}_{pair_stem}_{tag}.csv",
                           index=False, encoding="utf-8-sig")

            report = classification_report(y, pred, labels=class_order,
                                           output_dict=True, zero_division=0)
            pd.DataFrame(report).T.to_csv(
                outdir / f"classification_report_{dataset_tag}_{pair_stem}_{tag}.csv",
                encoding="utf-8-sig")

            pred_df = sub_meta.copy()
            pred_df["y_true"] = y
            pred_df["y_pred"] = pred
            pred_df["correct"] = pred_df["y_true"] == pred_df["y_pred"]
            pred_df["model"] = model_name
            pred_df.to_csv(outdir / f"predictions_{dataset_tag}_{pair_stem}_{tag}.csv",
                           index=False, encoding="utf-8-sig")
            prediction_rows.append(pred_df)

            summary_rows.append({
                "dataset": dataset_tag,
                "pair":    pair_title,
                "pair_code": pair_code,
                "model":   model_name,
                "feature_set": "ratio_only",
                "cv": "leave-one-replicate-out",
                "n_samples": len(y),
                "n_classes": len(np.unique(y)),
                "accuracy":  accuracy_score(y, pred),
                "f1_macro":  f1_score(y, pred, average="macro", zero_division=0),
                "n_correct": int(np.sum(y == pred)),
                "n_total":   int(len(y)),
            })

            make_standalone_cm_fig(cm, dataset_tag, pair_stem, pair_title,
                                   model_name, outdir, class_order)

        all_results[pair_stem] = {
            "pair_title": pair_title,
            "pca_df":     pca_df,
            "tsne_df":    tsne_df,
            "cm_by_model": cm_by_model,
            "class_order": get_class_order(sub_meta["class_ratio"]),
        }

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(outdir / f"ML_summary_{dataset_tag}_binary_ratio_classification.csv",
                      index=False, encoding="utf-8-sig")
    if prediction_rows:
        pd.concat(prediction_rows, ignore_index=True).to_csv(
            outdir / f"ALL_predictions_{dataset_tag}.csv",
            index=False, encoding="utf-8-sig")

    for model_name in model_names:
        make_combined_reference_figure(all_results, dataset_tag, "PCA", model_name, outdir)
        make_combined_reference_figure(all_results, dataset_tag, "tSNE", model_name, outdir)

    return summary_df


# =============================================================================
# 10. Manuscript Table 3: classification + mole-fraction regression
# =============================================================================
def select_table3_features(feats: pd.DataFrame) -> Tuple[np.ndarray, List[str], str]:
    """Return the feature matrix used for manuscript Table 3 outputs."""
    if CONFIG["TABLE3_FEATURE_SET"] == "ratio_only":
        cols = ratio_feature_columns(feats)
        tag = "ratio_only"
    elif CONFIG["TABLE3_FEATURE_SET"] == "all_engineered":
        cols = feats.columns.tolist()
        tag = "all_engineered"
    else:
        raise ValueError("TABLE3_FEATURE_SET must be 'ratio_only' or 'all_engineered'.")
    return feats[cols].to_numpy(), cols, tag


def safe_logo_splits(y: np.ndarray, groups: np.ndarray):
    """Leave-one-group-out when possible; otherwise stratified 3-fold."""
    unique_groups = np.unique(groups)
    if len(unique_groups) >= 2:
        return list(LeaveOneGroupOut().split(np.zeros(len(y)), y, groups))
    return list(StratifiedKFold(n_splits=3, shuffle=True,
                                random_state=42).split(np.zeros(len(y)), y))


def evaluate_classifier_task(X_feat: np.ndarray, y: np.ndarray, groups: np.ndarray,
                             task_name: str, outdir: Path, dataset_tag: str):
    """Run PLS-DA, RandomForest and SVM-RBF for one classification task."""
    rows = []
    pred_tables = []
    class_order = sorted(pd.unique(y).tolist())

    for model_name in ["PLS-DA", "RandomForest", "SVM-RBF"]:
        pred_all = np.empty(len(y), dtype=object)
        fold_rows = []
        for fold, (tr, te) in enumerate(safe_logo_splits(y, groups), 1):
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X_feat[tr])
            Xte = scaler.transform(X_feat[te])
            model = get_model(model_name)
            model.fit(Xtr, y[tr])
            pred = model.predict(Xte)
            pred_all[te] = pred
            fold_rows.append({
                "dataset": dataset_tag, "task": task_name,
                "model": model_name, "fold": fold,
                "n_train": len(tr), "n_test": len(te),
                "accuracy": accuracy_score(y[te], pred),
                "f1_macro": f1_score(y[te], pred, average="macro", zero_division=0),
            })

        acc = accuracy_score(y, pred_all)
        f1m = f1_score(y, pred_all, average="macro", zero_division=0)
        rows.append({
            "dataset":    dataset_tag,
            "task":       task_name,
            "model":      model_name,
            "feature_set": CONFIG["TABLE3_FEATURE_SET"],
            "n_samples":  int(len(y)),
            "n_classes":  int(len(np.unique(y))),
            "cv":         "leave-one-group-out",
            "group_definition": ("replicate" if task_name == "15-composition classification"
                                  else "composition_label"),
            "accuracy":  acc,
            "f1_macro":  f1m,
            "n_correct": int(np.sum(pred_all == y)),
            "n_total":   int(len(y)),
        })

        tag = model_file_tag(model_name)
        task_slug = task_name.replace(" ", "_")
        cm = confusion_matrix(y, pred_all, labels=class_order)
        pd.DataFrame(cm, index=class_order, columns=class_order).to_csv(
            outdir / f"TABLE3_{dataset_tag}_{task_slug}_{tag}_confusion_matrix.csv",
            encoding="utf-8-sig")
        pd.DataFrame(fold_rows).to_csv(
            outdir / f"TABLE3_{dataset_tag}_{task_slug}_{tag}_fold_metrics.csv",
            index=False, encoding="utf-8-sig")
        pred_df = pd.DataFrame({
            "dataset": dataset_tag, "task": task_name, "model": model_name,
            "y_true": y, "y_pred": pred_all,
            "correct": pred_all == y, "group": groups,
        })
        pred_df.to_csv(
            outdir / f"TABLE3_{dataset_tag}_{task_slug}_{tag}_predictions.csv",
            index=False, encoding="utf-8-sig")
        pred_tables.append(pred_df)

    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / f"TABLE3_{dataset_tag}_{task_name.replace(' ', '_')}_summary.csv",
                   index=False, encoding="utf-8-sig")
    if pred_tables:
        pd.concat(pred_tables, ignore_index=True).to_csv(
            outdir / f"TABLE3_{dataset_tag}_{task_name.replace(' ', '_')}_ALL_predictions.csv",
            index=False, encoding="utf-8-sig")
    return summary


def get_regressor(model_name: str, X_train: np.ndarray):
    if model_name == "PLSR":
        ncomp = min(CONFIG["PLSR_N_COMPONENTS"], X_train.shape[0] - 1, X_train.shape[1])
        ncomp = max(1, ncomp)
        return PLSRegression(n_components=ncomp)
    if model_name == "SVR-RBF":
        return MultiOutputRegressor(
            SVR(kernel="rbf", C=CONFIG["SVR_C"], gamma=CONFIG["SVR_GAMMA"]))
    if model_name == "RandomForestReg":
        return RandomForestRegressor(
            n_estimators=CONFIG["RF_N_ESTIMATORS"],
            random_state=CONFIG["RF_RANDOM_STATE"], n_jobs=-1)
    raise ValueError(f"Unknown regressor: {model_name}")


def evaluate_mole_fraction_regression(X_feat: np.ndarray, Y: np.ndarray,
                                       groups: np.ndarray, outdir: Path,
                                       dataset_tag: str):
    """Predict f_CIP / f_NFLX / f_LVLX via leave-one-composition-out CV."""
    component_names = ["f_CIP", "f_NFLX", "f_LVLX"]
    rows = []
    pred_tables = []

    for model_name in ["PLSR", "SVR-RBF", "RandomForestReg"]:
        pred_all = np.zeros_like(Y, dtype=float)
        fold_rows = []
        for fold, (tr, te) in enumerate(safe_logo_splits(np.zeros(len(Y)), groups), 1):
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X_feat[tr])
            Xte = scaler.transform(X_feat[te])
            reg = get_regressor(model_name, Xtr)
            reg.fit(Xtr, Y[tr])
            pred = np.asarray(reg.predict(Xte), dtype=float)
            pred = np.clip(pred, 0, 1)
            row_sum = pred.sum(axis=1, keepdims=True)
            row_sum[row_sum == 0] = 1.0
            pred = pred / row_sum
            pred_all[te] = pred
            fold_rows.append({
                "dataset": dataset_tag, "task": "mole-fraction regression",
                "model": model_name, "fold": fold,
                "n_train": len(tr), "n_test": len(te),
                "MAE":  mean_absolute_error(Y[te], pred),
                "RMSE": float(np.sqrt(mean_squared_error(Y[te], pred))),
                "R2":   r2_score(Y[te], pred, multioutput="uniform_average"),
            })

        mae  = mean_absolute_error(Y, pred_all)
        rmse = float(np.sqrt(mean_squared_error(Y, pred_all)))
        r2   = r2_score(Y, pred_all, multioutput="uniform_average")
        row = {
            "dataset":      dataset_tag,
            "task":         "mole-fraction regression",
            "model":        model_name,
            "feature_set":  CONFIG["TABLE3_FEATURE_SET"],
            "n_samples":    int(len(Y)),
            "cv":           "leave-one-composition-out",
            "group_definition": "composition_label",
            "MAE": mae, "RMSE": rmse, "R2": r2,
        }
        for j, cname in enumerate(component_names):
            row[f"MAE_{cname}"]  = mean_absolute_error(Y[:, j], pred_all[:, j])
            row[f"RMSE_{cname}"] = float(np.sqrt(mean_squared_error(Y[:, j], pred_all[:, j])))
            row[f"R2_{cname}"]   = r2_score(Y[:, j], pred_all[:, j])
        rows.append(row)

        tag = model_file_tag(model_name)
        pd.DataFrame(fold_rows).to_csv(
            outdir / f"TABLE3_{dataset_tag}_mole_fraction_regression_{tag}_fold_metrics.csv",
            index=False, encoding="utf-8-sig")
        pred_df = pd.DataFrame({
            "dataset": dataset_tag, "task": "mole-fraction regression",
            "model": model_name, "group": groups,
            "true_f_CIP":  Y[:, 0], "true_f_NFLX": Y[:, 1], "true_f_LVLX": Y[:, 2],
            "pred_f_CIP":  pred_all[:, 0],
            "pred_f_NFLX": pred_all[:, 1],
            "pred_f_LVLX": pred_all[:, 2],
        })
        pred_df.to_csv(
            outdir / f"TABLE3_{dataset_tag}_mole_fraction_regression_{tag}_predictions.csv",
            index=False, encoding="utf-8-sig")
        pred_tables.append(pred_df)

    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / f"TABLE3_{dataset_tag}_mole_fraction_regression_summary.csv",
                   index=False, encoding="utf-8-sig")
    if pred_tables:
        pd.concat(pred_tables, ignore_index=True).to_csv(
            outdir / f"TABLE3_{dataset_tag}_mole_fraction_regression_ALL_predictions.csv",
            index=False, encoding="utf-8-sig")
    return summary


def format_acc_f1(row) -> str:
    return f"{row['accuracy']:.3f} / {row['f1_macro']:.3f}"


def format_reg(row) -> str:
    return f"MAE={row['MAE']:.3f}; RMSE={row['RMSE']:.3f}; R²={row['R2']:.3f}"


def make_table3_fill_ready(dataset_tag: str, n_samples: int,
                           comp_summary: pd.DataFrame,
                           reg_summary: pd.DataFrame,
                           pair_summary: pd.DataFrame, outdir: Path):
    """Build the fill-ready manuscript Table 3."""
    dataset_label = f"{n_samples} spectra"
    detailed_rows = []

    for model_name in ["PLS-DA", "RandomForest", "SVM-RBF"]:
        row = comp_summary.loc[comp_summary["model"] == model_name].iloc[0]
        label = "Random Forest" if model_name == "RandomForest" else model_name
        detailed_rows.append({
            "Task": "15-composition classification",
            "Dataset": dataset_label, "Model": label,
            "Metric": "Accuracy / macro-F1",
            "Result": format_acc_f1(row),
        })

    for model_name, label in [("PLSR", "PLSR"),
                               ("SVR-RBF", "SVR-RBF"),
                               ("RandomForestReg", "Random Forest regression")]:
        row = reg_summary.loc[reg_summary["model"] == model_name].iloc[0]
        detailed_rows.append({
            "Task": "Mole-fraction regression",
            "Dataset": dataset_label, "Model": label,
            "Metric": "MAE / RMSE / R²",
            "Result": format_reg(row),
        })

    for model_name in ["PLS-DA", "RandomForest", "SVM-RBF"]:
        row = pair_summary.loc[pair_summary["model"] == model_name].iloc[0]
        label = "Random Forest" if model_name == "RandomForest" else model_name
        detailed_rows.append({
            "Task": "Pair classification",
            "Dataset": "internal mixed ratios only",
            "Model": label,
            "Metric": "Accuracy / macro-F1",
            "Result": format_acc_f1(row),
        })

    detailed = pd.DataFrame(detailed_rows)
    detailed.to_csv(outdir / f"TABLE3_fill_ready_detailed_{dataset_tag}.csv",
                    index=False, encoding="utf-8-sig")

    compact_rows = detailed_rows[:3]
    reg_text = "; ".join(
        f"{r['Model']}: {r['Result']}"
        for r in detailed_rows if r["Task"] == "Mole-fraction regression")
    compact_rows.append({
        "Task": "Mole-fraction regression",
        "Dataset": dataset_label,
        "Model": "PLSR / SVR-RBF / RF",
        "Metric": "MAE / RMSE / R²",
        "Result": reg_text,
    })
    pair_text = "; ".join(
        f"{r['Model']}: {r['Result']}"
        for r in detailed_rows if r["Task"] == "Pair classification")
    compact_rows.append({
        "Task": "Pair classification",
        "Dataset": "internal mixed ratios only",
        "Model": "PLS-DA / RF / SVM-RBF",
        "Metric": "Accuracy / macro-F1",
        "Result": pair_text,
    })
    compact = pd.DataFrame(compact_rows)
    compact.to_csv(outdir / f"TABLE3_fill_ready_compact_{dataset_tag}.csv",
                   index=False, encoding="utf-8-sig")
    return detailed, compact


def run_table3_analysis(dataset_tag: str, meta: pd.DataFrame,
                        X: np.ndarray, wl: np.ndarray, outdir: Path):
    """All analyses required to fill manuscript Table 3."""
    feats = extract_features(X, wl)
    X_feat, feat_cols, feature_tag = select_table3_features(feats)
    pd.DataFrame({"feature": feat_cols}).to_csv(
        outdir / f"TABLE3_{dataset_tag}_feature_columns_{feature_tag}.csv",
        index=False, encoding="utf-8-sig")

    # 1) 15 unique-composition classification, leave-one-replicate-out.
    y_comp = meta["composition_label"].astype(str).to_numpy()
    groups_rep = meta["replicate"].astype(str).to_numpy()
    comp_summary = evaluate_classifier_task(
        X_feat, y_comp, groups_rep,
        task_name="15-composition classification",
        outdir=outdir, dataset_tag=dataset_tag)

    # 2) Mole-fraction regression, leave-one-composition-out.
    Y_frac = np.column_stack([meta["f_C"].to_numpy(float),
                              meta["f_N"].to_numpy(float),
                              meta["f_L"].to_numpy(float)])
    groups_comp = meta["composition_label"].astype(str).to_numpy()
    reg_summary = evaluate_mole_fraction_regression(
        X_feat, Y_frac, groups_comp, outdir, dataset_tag)

    # 3) Pair classification, internal mixed ratios only (pure endpoints excluded).
    mixed_mask = (meta["pair"] != "pure").to_numpy()
    pair_summary = evaluate_classifier_task(
        X_feat[mixed_mask],
        meta.loc[mixed_mask, "pair"].astype(str).to_numpy(),
        meta.loc[mixed_mask, "composition_label"].astype(str).to_numpy(),
        task_name="pair classification internal mixed ratios only",
        outdir=outdir, dataset_tag=dataset_tag)

    comp_summary.to_csv(
        outdir / f"TABLE3_{dataset_tag}_15_composition_classification_summary.csv",
        index=False, encoding="utf-8-sig")
    pair_summary.to_csv(
        outdir / f"TABLE3_{dataset_tag}_pair_classification_internal_mixed_only_summary.csv",
        index=False, encoding="utf-8-sig")
    detailed, compact = make_table3_fill_ready(
        dataset_tag, len(meta), comp_summary, reg_summary, pair_summary, outdir)

    all_task_summary = pd.concat([comp_summary, reg_summary, pair_summary],
                                  ignore_index=True, sort=False)
    all_task_summary.to_csv(outdir / f"TABLE3_{dataset_tag}_all_task_summary.csv",
                            index=False, encoding="utf-8-sig")
    print(f"TABLE3 {dataset_tag} compact results:")
    print(compact.to_string(index=False))
    return all_task_summary, detailed, compact


# =============================================================================
# 11. Figure map and zip
# =============================================================================
def write_figure_map(outdir: Path):
    rows = []
    for dataset_tag in ["REAL3", "AUG5_PREVIEW"]:
        for method in ["PCA", "tSNE"]:
            for model in ["PLSDA", "RandomForest", "SVMRBF"]:
                fn = f"Fig_reference_style_{dataset_tag}_{method}_CM_{model}.png"
                if (outdir / fn).exists():
                    rows.append({
                        "file_name": fn,
                        "suggested_use": ("Main text candidate"
                                          if dataset_tag == "REAL3"
                                          else "Preview / SI candidate"),
                        "figure_content":
                            f"2x3 reference-style figure: top={method} plots for "
                            f"three binary pairs; bottom=composition-ratio "
                            f"confusion matrices; model={model}",
                        "note":
                            "REAL3 uses true 3 replicates. AUG5_PREVIEW includes "
                            "synthetic rep4/rep5 and must not be described as "
                            "independent experimental replicates.",
                    })

    for pair_stem, pair_title in [("CIP_NFLX",  "CIP:NFLX"),
                                   ("CIP_LVLX",  "CIP:LVLX"),
                                   ("LVLX_NFLX", "LVLX:NFLX")]:
        for dataset_tag in ["REAL3", "AUG5_PREVIEW"]:
            rows.extend([
                {"file_name": f"plotdata_{dataset_tag}_{pair_stem}_PCA.csv",
                 "suggested_use": "Origin source data",
                 "figure_content": f"PCA source coordinates for {pair_title}",
                 "note": "Use this for Origin recreation of the PCA panel."},
                {"file_name": f"plotdata_{dataset_tag}_{pair_stem}_tSNE.csv",
                 "suggested_use": "Origin source data",
                 "figure_content": f"t-SNE source coordinates for {pair_title}",
                 "note": "Use this for Origin recreation of the t-SNE panel."},
            ])
            for model in ["PLSDA", "RandomForest", "SVMRBF"]:
                rows.append({
                    "file_name": f"confusion_matrix_{dataset_tag}_{pair_stem}_{model}.csv",
                    "suggested_use": "Origin source data / SI table",
                    "figure_content": f"Confusion matrix for {pair_title}, model={model}",
                    "note": "Rows = true labels, columns = predicted labels.",
                })

    rows.extend([
        {"file_name": "ML_summary_REAL3_binary_ratio_classification.csv",
         "suggested_use": "Main text / SI table",
         "figure_content": "Summary metrics for the real-3-replicate classification.",
         "note": "Conservative reference summary."},
        {"file_name": "ML_summary_AUG5_PREVIEW_binary_ratio_classification.csv",
         "suggested_use": "Preview only / SI if clearly labelled",
         "figure_content": "Summary metrics after synthetic augmentation to 5 replicates.",
         "note": "Synthetic replicates are not independent measurements."},
        {"file_name": "origin_REAL3_crop_wide.csv",
         "suggested_use": "Origin source data",
         "figure_content": "Wide-format cropped spectra for REAL3.",
         "note": "nm + sample columns."},
        {"file_name": "origin_REAL3_mean_sd_long.csv",
         "suggested_use": "Origin source data",
         "figure_content": "Mean±SD spectra for REAL3.",
         "note": "Useful for mean spectra and error-band plots."},
        {"file_name": "features_REAL3_by_sample.csv",
         "suggested_use": "SI source data",
         "figure_content": "Feature matrix for REAL3.",
         "note": "Intensity, area, centroid and ratio features."},
    ])

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "FIGURE_FILE_MAP.csv", index=False, encoding="utf-8-sig")

    lines = ["# Figure file map", "",
             "| file_name | suggested_use | figure_content | note |",
             "|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(f"| {r['file_name']} | {r['suggested_use']} | "
                     f"{r['figure_content']} | {r['note']} |")
    (outdir / "FIGURE_FILE_MAP.md").write_text("\n".join(lines), encoding="utf-8")


def zip_folder(folder: Path, zip_path: Path):
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(folder):
            for file in files:
                full = Path(root) / file
                if full.resolve() == zip_path.resolve():
                    continue
                zf.write(full, full.relative_to(folder.parent))


# =============================================================================
# 12. Command-line interface
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=("Binary-mixture ML pipeline for fluorescence spectra of "
                     "CIP/LVLX/NFLX with reference-style figures and "
                     "manuscript Table 3 outputs."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", type=Path, required=True,
                   help="Path to the wide-format mixture CSV.")
    p.add_argument("--outdir", type=Path, default=Path("./results_mixture"),
                   help="Output directory for figures and tables.")
    p.add_argument("--export-formats", type=str, default="png",
                   help="Comma-separated figure formats (e.g. png,pdf,svg).")
    p.add_argument("--feature-set", type=str,
                   choices=["ratio_only", "all_engineered"],
                   default="ratio_only",
                   help="Feature set used for manuscript Table 3 outputs.")
    p.add_argument("--no-aug-preview", action="store_true",
                   help="Skip the synthetic AUG5_PREVIEW dataset.")
    p.add_argument("--target-reps", type=int, default=5,
                   help="Target replicate count for AUG5_PREVIEW.")
    p.add_argument("--no-table3", action="store_true",
                   help="Skip the manuscript Table 3 analysis.")
    p.add_argument("--single-classifier", action="store_true",
                   help="Run only the classifier in --main-classifier instead "
                        "of all three.")
    p.add_argument("--main-classifier", type=str, default="PLS-DA",
                   choices=["PLS-DA", "RandomForest", "SVM-RBF"],
                   help="Classifier used when --single-classifier is set.")
    p.add_argument("--make-zip", action="store_true",
                   help="Also produce a zip of the entire output folder.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    return p.parse_args()


# =============================================================================
# 13. Main
# =============================================================================
def main():
    args = parse_args()

    CONFIG["DATA_PATH"]                  = args.data
    CONFIG["OUTDIR"]                     = args.outdir
    CONFIG["EXPORT_FORMATS"]             = [f.strip() for f in args.export_formats.split(",") if f.strip()]
    CONFIG["TABLE3_FEATURE_SET"]         = args.feature_set
    CONFIG["MAKE_AUG5_PREVIEW"]          = not args.no_aug_preview
    CONFIG["TARGET_REPS"]                = args.target_reps
    CONFIG["RUN_TABLE3_ANALYSIS"]        = not args.no_table3
    CONFIG["RUN_BOTH_CLASSIFIERS"]       = not args.single_classifier
    CONFIG["CLASSIFIER_FOR_MAIN_FIGURE"] = args.main_classifier
    CONFIG["MAKE_ZIP"]                   = args.make_zip
    CONFIG["AUG_RANDOM_STATE"]           = args.seed
    CONFIG["RF_RANDOM_STATE"]            = args.seed
    CONFIG["TSNE_RANDOM_STATE"]          = args.seed

    if not CONFIG["DATA_PATH"].exists():
        raise FileNotFoundError(f"Data file not found: {CONFIG['DATA_PATH']}")

    outdir = ensure_dir(CONFIG["OUTDIR"])

    print("Loading data...")
    meta_raw, wl_raw, X_raw = load_wide_csv(CONFIG["DATA_PATH"])
    X_real, wl = preprocess_spectra(X_raw, wl_raw)
    meta_real = meta_raw.reset_index(drop=True).copy()

    print(f"REAL3 samples: {len(meta_real)}")
    print(f"Wavelength points after crop: {len(wl)}")

    all_summaries = []
    print("Analysing REAL3 dataset...")
    s_real = analyze_dataset("REAL3", meta_real, X_real, wl, outdir)
    all_summaries.append(s_real)

    table3_summaries = []
    if CONFIG["RUN_TABLE3_ANALYSIS"]:
        print("Running manuscript Table 3 analysis for REAL3...")
        table3_real, _, _ = run_table3_analysis("REAL3", meta_real, X_real, wl, outdir)
        table3_summaries.append(table3_real)

    if CONFIG["MAKE_AUG5_PREVIEW"]:
        print("Creating AUG5_PREVIEW dataset...")
        meta_aug, X_aug = augment_to_target_reps(
            meta_real, X_real, target_reps=CONFIG["TARGET_REPS"])
        print(f"AUG5_PREVIEW samples: {len(meta_aug)}")
        s_aug = analyze_dataset("AUG5_PREVIEW", meta_aug, X_aug, wl, outdir)
        all_summaries.append(s_aug)
        if CONFIG["RUN_TABLE3_ANALYSIS"]:
            print("Running manuscript Table 3 analysis for AUG5_PREVIEW...")
            table3_aug, _, _ = run_table3_analysis(
                "AUG5_PREVIEW", meta_aug, X_aug, wl, outdir)
            table3_summaries.append(table3_aug)

    pd.concat(all_summaries, ignore_index=True).to_csv(
        outdir / "ALL_ML_SUMMARIES.csv", index=False, encoding="utf-8-sig")
    if table3_summaries:
        pd.concat(table3_summaries, ignore_index=True, sort=False).to_csv(
            outdir / "ALL_TABLE3_TASK_SUMMARIES.csv",
            index=False, encoding="utf-8-sig")

    write_figure_map(outdir)
    with open(outdir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({k: (str(v) if isinstance(v, Path) else v)
                   for k, v in CONFIG.items()}, f,
                  ensure_ascii=False, indent=2)

    if CONFIG["MAKE_ZIP"]:
        zip_path = outdir.parent / f"{outdir.name}.zip"
        zip_folder(outdir, zip_path)
        print(f"ZIP saved to: {zip_path}")

    print("Done.")
    print(f"Outputs saved to: {outdir}")


if __name__ == "__main__":
    main()
