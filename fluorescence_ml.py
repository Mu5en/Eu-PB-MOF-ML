#!/usr/bin/env python3
"""
Quinolone antibiotic fluorescence-spectrum classification pipeline.

End-to-end machine-learning analysis of replicate fluorescence spectra of three
quinolone antibiotics (LVLX, NFLX, CIP) measured with a bi-ligand Eu-MOF probe:

  * Replicate-aware preprocessing (AsLS baseline + Savitzky-Golay smoothing).
  * 67-feature sensor-array description: intensity ratios, Savitzky-Golay
    derivatives, peak-shape descriptors and full-spectrum PCA scores.
  * Publication-quality PCA (2D/3D), t-SNE and circular HCA figures.
  * Group-aware nested cross-validation with PLS-DA, Random Forest, SVM-RBF.
  * Leave-one-concentration-out (LOCO) and drug-conditional LOCO (LOCO-DC).
  * Optional random-label permutation test and feature-family ablation study.
  * VIP + permutation feature importance with human-readable descriptions.

Input CSV format
----------------
Columns: antibiotic, concentration, replicate, <wavelength_1>, <wavelength_2>, ...
    antibiotic     : one of {CIP, LVLX, NFLX}
    concentration  : numeric (µM); 0 is allowed but excluded from classification
    replicate      : integer replicate index per (antibiotic, concentration)

Usage
-----
    python pipeline.py --data FL_Data.csv --outdir ./results
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.patches import Ellipse
from scipy import sparse
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.signal import savgol_filter
from scipy.sparse.linalg import spsolve
from scipy.stats import chi2

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, adjusted_rand_score, auc, confusion_matrix, f1_score,
    normalized_mutual_info_score, precision_score, recall_score, roc_curve,
)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC

try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SGKF = True
except ImportError:
    from sklearn.model_selection import GroupKFold
    HAS_SGKF = False

warnings.filterwarnings("ignore")


# ============================================================================
# 1. Configuration (populated from command-line arguments in main())
# ============================================================================
CONFIG: Dict = {
    "DATA_PATH": None,
    "OUTDIR":    Path("./results"),

    "EXPECTED_CLASSES":          ["CIP", "LVLX", "NFLX"],
    "EXPECTED_N_CONCENTRATIONS": 10,
    "EXPECTED_N_REPLICATES":     5,

    "CROP_LO": 380, "CROP_HI": 720,
    "ALS_LAM": 1e6, "ALS_P": 0.001, "ALS_NITER": 10,
    "SG_WINDOW": 21, "SG_POLY": 2,
    "DERIV_WINDOW": 21, "DERIV_POLY": 3,

    "N_PCA_FEATURES": 8,
    "N_OUTER": 5, "N_INNER": 3,
    "N_JOBS": 1,
    "RANDOM_STATE": 42,

    "SAVE_FIGURES": True,
    "RUN_LOCO":     True,
    "RUN_LOCO_DC":  True,

    # Random-label permutation test (group-level shuffle of class labels).
    "RUN_RANDOM_LABEL_TEST":  False,
    "RANDOM_LABEL_N_REPEATS": 20,
    "RANDOM_LABEL_MODELS":    ["PLS-DA", "RandomForest", "SVM-RBF"],

    # Ablation study comparing different feature families.
    "RUN_ABLATION": False,
    "ABLATION_FEATURE_SETS": [
        "ratio", "derivative", "peakshape", "pca",
        "ratio+derivative", "ratio+peakshape", "ratio+pca",
        "no_pca", "all",
    ],
    "ABLATION_MODELS": ["PLS-DA", "RandomForest", "SVM-RBF"],
}

DRUG_PEAKS  = {"NFLX": 446, "LVLX": 454, "CIP": 492}
PROBE_PEAKS = {"p535": 535, "p556": 556, "p592": 592, "p617": 617, "p700": 700}
CLASS_ORDER = CONFIG["EXPECTED_CLASSES"]
EPS = 1e-9


# ============================================================================
# 2. Feature naming conventions and auto-description
# ============================================================================
"""
Feature naming conventions for the fluorescence sensor-array data set
=====================================================================

Drug emission peaks
    NFLX -> 446 nm   LVLX -> 454 nm   CIP -> 492 nm

Probe (Eu-PB-MOF) emission peaks
    535 / 556 / 592 / 700 nm   ligand-centred + Eu3+ low-energy transitions
    617 nm                     Eu3+ 5D0->7F2 hypersensitive band (REFERENCE)

Feature families
    I<drug>/I<probe>     intensity ratio: drug peak / probe peak
    I<drug1>/I<drug2>    inter-drug intensity ratio
    I<a>/I<b>            intra-probe intensity ratio (both probe peaks)
    d1@<W>nm  d2@<W>nm   1st / 2nd derivative at wavelength W
    A<lo>_<hi>           integrated area in [lo, hi] nm
    A<a>_<b>/A<c>_<d>    area ratio between two integration windows
                         (peak SHAPE / asymmetry descriptor)
    centroid_<lo>_<hi>   intensity-weighted mean wavelength in [lo, hi]
    I<a>-I<b>            intensity DIFFERENCE between two wavelengths
    PC<n>                full-spectrum principal component
"""

EU_TRANSITIONS = {
    "535": "Eu3+ 5D1->7F1",
    "556": "Eu3+ 5D1->7F2",
    "592": "Eu3+ 5D0->7F1 (magnetic dipole)",
    "617": "Eu3+ 5D0->7F2 (hypersensitive, reference)",
    "700": "Eu3+ 5D0->7F4",
}
DRUG_WL = {str(v): k for k, v in DRUG_PEAKS.items()}  # 446 -> "NFLX", etc.

_EXACT_DESC = {
    "centroid_430_470":
        "Spectral centroid (intensity-weighted mean wavelength) in 430-470 nm; "
        "tracks the drug-induced peak position (446 / 454 / 492 nm).",
    "I454-I446":
        "Intensity difference I(454) - I(446) = LVLX peak - NFLX peak; "
        "key discriminator between LVLX and NFLX (only 8 nm apart).",
    "I492-I454":
        "Intensity difference I(492) - I(454) = CIP peak - LVLX peak.",
    "(I454-I446)/I617":
        "Drug-peak gap (LVLX vs NFLX) normalised by the Eu3+ 617 nm reference.",
    "(I492-I454)/I617":
        "Drug-peak gap (CIP vs LVLX) normalised by the Eu3+ 617 nm reference.",
}


def describe_feature(name: str) -> str:
    """Return a one-line plain-English description of a feature name."""
    if name in _EXACT_DESC:
        return _EXACT_DESC[name]

    m = re.fullmatch(r"PC(\d+)", name)
    if m:
        return (f"Principal component #{m.group(1)} of the standardised full "
                f"spectrum (global variance).")

    m = re.fullmatch(r"d([12])@(\d+)nm", name)
    if m:
        order, wl = m.groups()
        what = "First-order spectral derivative" if order == "1" \
               else "Second-order spectral derivative"
        peak = DRUG_WL.get(wl) or EU_TRANSITIONS.get(wl, "")
        return f"{what} at {wl} nm" + (f" ({peak})" if peak else "") + "."

    m = re.fullmatch(r"A(\d+)_(\d+)/A(\d+)_(\d+)", name)
    if m:
        a, b, c, d = m.groups()
        return (f"Ratio of integrated intensities: I(lambda) dlambda over "
                f"{a}-{b} nm / I(lambda) dlambda over {c}-{d} nm. "
                f"Peak-shape / asymmetry descriptor for the Eu3+ 617 nm region.")

    m = re.fullmatch(r"A(\d+)_(\d+)", name)
    if m:
        a, b = m.groups()
        return f"Integrated area over the {a}-{b} nm window."

    m = re.fullmatch(r"I(NFLX|LVLX|CIP)/I(p?)(\w+)", name)
    if m:
        drug, _, target = m.groups()
        if target in DRUG_PEAKS:
            return (f"Intensity ratio I({DRUG_PEAKS[drug]} nm, {drug}) / "
                    f"I({DRUG_PEAKS[target]} nm, {target}).")
        wl = target.lstrip("p")
        ref = EU_TRANSITIONS.get(wl, f"{wl} nm")
        return (f"Intensity ratio I({DRUG_PEAKS[drug]} nm, {drug}) / "
                f"I({wl} nm, {ref}).")

    m = re.fullmatch(r"I(\d+)/I(\d+)", name)
    if m:
        a, b = m.groups()
        a_desc = EU_TRANSITIONS.get(a, "")
        b_desc = EU_TRANSITIONS.get(b, "")
        return ("Intra-probe intensity ratio: " +
                f"I({a} nm" + (f", {a_desc}" if a_desc else "") + ") / " +
                f"I({b} nm" + (f", {b_desc}" if b_desc else "") +
                "). Sensitive to Eu3+ coordination environment.")

    return "(no description)"


# ============================================================================
# 3. Publication style
# ============================================================================
PALETTE = {
    "LVLX":  "#0072B2",
    "NFLX":  "#D55E00",
    "CIP":   "#009E73",
    "PROBE": "#666666",
}
MARKERS = {"LVLX": "s", "NFLX": "o", "CIP": "^"}

WIDTH_1COL  = 90  / 25.4
WIDTH_15COL = 140 / 25.4
WIDTH_2COL  = 190 / 25.4

ELLIPSE_NSTD_2D = float(np.sqrt(chi2.ppf(0.95, 2)))
ELLIPSE_NSTD_3D = float(np.sqrt(chi2.ppf(0.95, 3)))


def setup_talanta_style() -> None:
    plt.rcParams.update({
        "font.family":      "serif",
        "font.serif":       ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
        "font.size":        8.5,
        "axes.labelsize":   9.5,
        "axes.titlesize":   10,
        "xtick.labelsize":  8,
        "ytick.labelsize":  8,
        "legend.fontsize":  8.5,
        "figure.titlesize": 10.5,
        "mathtext.fontset": "stix",
        "axes.linewidth":     0.8,
        "lines.linewidth":    1.3,
        "patch.linewidth":    0.5,
        "xtick.major.width":  0.8,
        "ytick.major.width":  0.8,
        "xtick.major.size":   3.5,
        "ytick.major.size":   3.5,
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          False,
        "legend.frameon":      False,
        "legend.handlelength": 1.4,
        "legend.handletextpad": 0.4,
        "savefig.dpi":        600,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.05,
    })


def save_publication(fig: plt.Figure, outdir: Path, stem: str,
                     formats: Tuple[str, ...] = ("png", "pdf", "tiff")) -> None:
    if not CONFIG["SAVE_FIGURES"]:
        return
    for fmt in formats:
        path = outdir / f"{stem}.{fmt}"
        try:
            if fmt == "tiff":
                fig.savefig(path, dpi=600, format="tiff",
                            pil_kwargs={"compression": "tiff_lzw"})
            elif fmt == "pdf":
                fig.savefig(path, format="pdf")
            else:
                fig.savefig(path, dpi=300, format=fmt)
        except Exception as exc:
            print(f"  [warn] save {fmt} failed: {exc}")


# ============================================================================
# 4. Confidence ellipse / ellipsoid
# ============================================================================
def add_confidence_ellipse_2d(x, y, ax, color, n_std=ELLIPSE_NSTD_2D,
                              alpha_face=0.13, lw=1.3, ls="--"):
    if x.size < 3:
        return None
    cov = np.cov(x, y)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle  = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))
    width  = 2 * n_std * float(np.sqrt(max(eigvals[0], 0)))
    height = 2 * n_std * float(np.sqrt(max(eigvals[1], 0)))
    center = (float(np.mean(x)), float(np.mean(y)))
    ax.add_patch(Ellipse(center, width, height, angle=angle,
                         facecolor=color, alpha=alpha_face, edgecolor="none"))
    ax.add_patch(Ellipse(center, width, height, angle=angle,
                         facecolor="none", edgecolor=color, linewidth=lw,
                         linestyle=ls, alpha=0.95))


def add_confidence_ellipsoid_3d(points, ax, color, n_std=ELLIPSE_NSTD_3D,
                                alpha_surface=0.08, n_grid=24):
    if points.shape[0] < 4:
        return
    mean = points.mean(axis=0)
    cov = np.cov(points.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 0, None)
    radii = n_std * np.sqrt(eigvals)
    u = np.linspace(0, 2 * np.pi, n_grid)
    v = np.linspace(0, np.pi, n_grid)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    pts = np.stack([xs.ravel() * radii[0], ys.ravel() * radii[1],
                    zs.ravel() * radii[2]], axis=1)
    pts = pts @ eigvecs.T + mean
    XS, YS, ZS = (pts[:, i].reshape(xs.shape) for i in range(3))
    ax.plot_surface(XS, YS, ZS, color=color, alpha=alpha_surface,
                    linewidth=0, antialiased=True, shade=False)
    ax.plot_wireframe(XS, YS, ZS, color=color, linewidth=0.3,
                      alpha=0.55, rstride=4, cstride=4)


# ============================================================================
# 5. Data I/O
# ============================================================================
def load_replicate_data(path: Path):
    df = pd.read_csv(path, sep=None, engine="python")
    n0 = len(df)
    df.columns = [str(c).strip() for c in df.columns]
    for c in ("antibiotic", "concentration", "replicate"):
        if c not in df.columns:
            raise ValueError(f"CSV missing required column: {c}")
    wl_cols = df.columns[3:]
    wavelengths = np.array([float(str(c).strip()) for c in wl_cols], dtype=float)
    df["antibiotic"]    = df["antibiotic"].astype(str).str.strip().str.upper()
    df["concentration"] = pd.to_numeric(df["concentration"], errors="coerce")
    df["replicate"]     = pd.to_numeric(df["replicate"],     errors="coerce")
    df.loc[:, wl_cols]  = df.loc[:, wl_cols].apply(pd.to_numeric, errors="coerce")
    X_tmp = df.loc[:, wl_cols].to_numpy(dtype=float)
    valid = (df["antibiotic"].isin(CLASS_ORDER).to_numpy()
             & df["concentration"].notna().to_numpy()
             & df["replicate"].notna().to_numpy()
             & np.isfinite(X_tmp).all(axis=1))
    df_clean = df.loc[valid].reset_index(drop=True)
    if n0 - len(df_clean):
        print(f"[load] dropped {n0 - len(df_clean)} invalid records")
    X_raw     = df_clean.loc[:, wl_cols].to_numpy(dtype=float)
    y         = df_clean["antibiotic"].astype(str).to_numpy()
    conc      = df_clean["concentration"].astype(float).to_numpy()
    replicate = df_clean["replicate"].astype(int).to_numpy()
    groups    = np.array([f"{a}_{c:.12g}" for a, c in zip(y, conc)])
    return X_raw, y, conc, replicate, wavelengths, groups, df_clean


# ============================================================================
# 6. Preprocessing
# ============================================================================
def asls_baseline(y, lam=1e6, p=0.001, niter=10):
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2)).tocsc()
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def _odd_window(window, n, mn=5):
    window = int(window)
    if window >= n:
        window = n - 1
    if window % 2 == 0:
        window -= 1
    return max(mn, window)


def preprocess_spectra(X_raw, wavelengths):
    mask = (wavelengths >= CONFIG["CROP_LO"]) & (wavelengths <= CONFIG["CROP_HI"])
    Xc, wl = X_raw[:, mask], wavelengths[mask]
    Xb = np.empty_like(Xc)
    for i in range(Xc.shape[0]):
        Xb[i] = Xc[i] - asls_baseline(Xc[i], CONFIG["ALS_LAM"],
                                       CONFIG["ALS_P"], CONFIG["ALS_NITER"])
    Xb = np.clip(Xb, 0, None)
    win = _odd_window(CONFIG["SG_WINDOW"], Xb.shape[1])
    Xs = savgol_filter(Xb, win, CONFIG["SG_POLY"], axis=1)
    return np.clip(Xs, 0, None), wl


# ============================================================================
# 7. Feature engineering
# ============================================================================
def _idx(wl, t): return int(np.argmin(np.abs(wl - t)))
def _div(a, b):  return a / (b + EPS)


def _integ(X, wl, lo, hi):
    m = (wl >= lo) & (wl <= hi)
    if m.sum() < 2:
        raise ValueError(f"Integration window {lo}-{hi} has fewer than 2 points")
    y, x = X[:, m], wl[m]
    return (np.trapezoid if hasattr(np, "trapezoid") else np.trapz)(y, x, axis=1)


def _centroid(X, wl, lo, hi):
    m = (wl >= lo) & (wl <= hi)
    w, Y = wl[m], X[:, m]
    return (Y @ w) / (Y.sum(axis=1) + EPS)


def extract_ratio_features(X, wl):
    di = {k: _idx(wl, v) for k, v in DRUG_PEAKS.items()}
    pi = {k: _idx(wl, v) for k, v in PROBE_PEAKS.items()}
    feats = {}
    for d, idx_d in di.items():
        for p, idx_p in pi.items():
            feats[f"I{d}/I{p}"] = _div(X[:, idx_d], X[:, idx_p])
    drugs = list(di)
    for i in range(len(drugs)):
        for j in range(i + 1, len(drugs)):
            d1, d2 = drugs[i], drugs[j]
            feats[f"I{d1}/I{d2}"] = _div(X[:, di[d1]], X[:, di[d2]])
            feats[f"I{d2}/I{d1}"] = _div(X[:, di[d2]], X[:, di[d1]])
    for p in ("p535", "p556", "p592", "p700"):
        feats[f"I{PROBE_PEAKS[p]}/I617"] = _div(X[:, pi[p]], X[:, pi["p617"]])
    df = pd.DataFrame(feats)
    return df.values, list(df.columns)


def extract_derivative_features(X, wl):
    win = _odd_window(CONFIG["DERIV_WINDOW"], X.shape[1])
    d1 = savgol_filter(X, win, CONFIG["DERIV_POLY"], deriv=1, axis=1)
    d2 = savgol_filter(X, win, CONFIG["DERIV_POLY"], deriv=2, axis=1)
    feats, names = [], []
    for t in (446, 454, 492, 535, 556, 592, 617, 700):
        i = _idx(wl, t)
        feats.append(d1[:, i]); names.append(f"d1@{t}nm")
        feats.append(d2[:, i]); names.append(f"d2@{t}nm")
    return np.array(feats).T, names


def extract_peakshape_features(X, wl):
    regions = {"440_452": (440, 452), "448_462": (448, 462), "485_500": (485, 500),
               "580_605": (580, 605), "605_630": (605, 630), "690_710": (690, 710)}
    A = {k: _integ(X, wl, lo, hi) for k, (lo, hi) in regions.items()}
    feats = {f"A{k}": v for k, v in A.items()}
    for n in ("440_452", "448_462", "485_500", "580_605", "690_710"):
        feats[f"A{n}/A605_630"] = _div(A[n], A["605_630"])
    feats["A440_452/A448_462"] = _div(A["440_452"], A["448_462"])
    feats["A485_500/A448_462"] = _div(A["485_500"], A["448_462"])
    feats["centroid_430_470"]  = _centroid(X, wl, 430, 470)
    i446, i454 = X[:, _idx(wl, 446)], X[:, _idx(wl, 454)]
    i492, i617 = X[:, _idx(wl, 492)], X[:, _idx(wl, 617)]
    feats["I454-I446"]         = i454 - i446
    feats["I492-I454"]         = i492 - i454
    feats["(I454-I446)/I617"]  = _div(i454 - i446, i617)
    feats["(I492-I454)/I617"]  = _div(i492 - i454, i617)
    df = pd.DataFrame(feats)
    return df.values, list(df.columns)


class FullSpectrumPCA:
    def __init__(self, n_components=8):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = None
        self.n_components_ = 0

    def fit(self, X):
        n_comp = max(1, min(self.n_components, X.shape[0] - 1, X.shape[1]))
        self.n_components_ = n_comp
        self.pca = PCA(n_components=n_comp)
        self.pca.fit(self.scaler.fit_transform(X))
        return self

    def transform(self, X):
        return self.pca.transform(self.scaler.transform(X))

    @property
    def names(self):
        return [f"PC{i + 1}" for i in range(self.n_components_)]


_FEATURE_SETS = {
    "ratio":            {"ratio"},
    "derivative":       {"derivative"},
    "peakshape":        {"peakshape"},
    "pca":              {"pca"},
    "ratio+derivative": {"ratio", "derivative"},
    "ratio+peakshape":  {"ratio", "peakshape"},
    "ratio+pca":        {"ratio", "pca"},
    "no_pca":           {"ratio", "derivative", "peakshape"},
    "all":              {"ratio", "derivative", "peakshape", "pca"},
}


def build_features_train_test(X_train, X_test, wl, feature_set="all", n_pca=8):
    parts = _FEATURE_SETS[feature_set]
    blocks_tr, blocks_te, names = [], [], []
    if "ratio" in parts:
        tr, n = extract_ratio_features(X_train, wl)
        te, _ = extract_ratio_features(X_test, wl)
        blocks_tr.append(tr); blocks_te.append(te); names.extend(n)
    if "derivative" in parts:
        tr, n = extract_derivative_features(X_train, wl)
        te, _ = extract_derivative_features(X_test, wl)
        blocks_tr.append(tr); blocks_te.append(te); names.extend(n)
    if "peakshape" in parts:
        tr, n = extract_peakshape_features(X_train, wl)
        te, _ = extract_peakshape_features(X_test, wl)
        blocks_tr.append(tr); blocks_te.append(te); names.extend(n)
    if "pca" in parts:
        fp = FullSpectrumPCA(n_components=n_pca).fit(X_train)
        blocks_tr.append(fp.transform(X_train))
        blocks_te.append(fp.transform(X_test))
        names.extend(fp.names)
    return np.hstack(blocks_tr), np.hstack(blocks_te), names


def build_features_for_viz(X, wl, feature_set="all", n_pca=8):
    F, _, names = build_features_train_test(X, X, wl, feature_set=feature_set, n_pca=n_pca)
    return F, names


# ============================================================================
# 8. Five-replicate QC
# ============================================================================
def replicate_qc_peak_rsd(X_pre, wl, y, conc, groups, outdir: Path):
    idx_617 = _idx(wl, 617)
    rows = []
    for g in np.unique(groups):
        m = groups == g
        ab, c = y[m][0], conc[m][0]
        idx_d = _idx(wl, DRUG_PEAKS[ab])
        I_d, I_617 = X_pre[m, idx_d], X_pre[m, idx_617]
        ratio = _div(I_d, I_617)
        rsd = lambda v: float(np.std(v, ddof=1) / (np.mean(v) + EPS) * 100)
        rows.append({"antibiotic": ab, "concentration": float(c),
                     "n_replicates": int(m.sum()), "drug_peak_nm": DRUG_PEAKS[ab],
                     "I_drug_mean": float(np.mean(I_d)),  "I_drug_rsd_%": rsd(I_d),
                     "I617_mean":   float(np.mean(I_617)),"I617_rsd_%":   rsd(I_617),
                     "ratio_mean":  float(np.mean(ratio)),"ratio_rsd_%":  rsd(ratio)})
    qc = pd.DataFrame(rows).sort_values(["antibiotic", "concentration"])
    qc.to_csv(outdir / "replicate_qc_peak_rsd.csv", index=False)
    summary = qc.groupby("antibiotic")[
        ["I_drug_rsd_%", "I617_rsd_%", "ratio_rsd_%"]].agg(["mean", "max"])
    summary.to_csv(outdir / "replicate_qc_peak_rsd_summary.csv")
    return qc, summary


# ============================================================================
# 9. Models
# ============================================================================
class PLSDA(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=3):
        self.n_components = n_components

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        Y = np.zeros((len(y), len(self.classes_)))
        for i, c in enumerate(self.classes_):
            Y[y == c, i] = 1.0
        n_comp = max(1, min(self.n_components, X.shape[1], X.shape[0] - 1))
        self.pls_ = PLSRegression(n_components=n_comp, scale=False).fit(X, Y)
        self.n_components_ = n_comp
        return self

    def predict(self, X):
        return self.classes_[np.argmax(self.pls_.predict(X), axis=1)]

    def predict_proba(self, X):
        Y = np.clip(self.pls_.predict(X), 0, None)
        s = Y.sum(axis=1, keepdims=True); s[s == 0] = 1
        return Y / s


def calculate_pls_vip(plsda, feat_names) -> pd.Series:
    pls = plsda.pls_; T, W, Q = pls.x_scores_, pls.x_weights_, pls.y_loadings_
    p, h = W.shape
    s = np.diag(T.T @ T @ Q.T @ Q).reshape(h, -1)
    total = max(np.sum(s), EPS)
    vip = np.empty(p)
    for i in range(p):
        w = np.array([(W[i, j] / np.linalg.norm(W[:, j])) ** 2 for j in range(h)])
        vip[i] = np.sqrt(p * np.sum(s.flatten() * w) / total)
    return pd.Series(vip, index=feat_names).sort_values(ascending=False)


def make_models(random_state=42):
    return {
        "PLS-DA": (PLSDA(),
                   {"n_components": [2, 3, 4, 5, 6, 8, 10]}),
        "RandomForest": (RandomForestClassifier(random_state=random_state,
                                                n_jobs=CONFIG["N_JOBS"]),
                         {"n_estimators": [300, 500],
                          "max_depth": [None, 3, 5, 10],
                          "min_samples_split": [2, 4]}),
        "SVM-RBF": (SVC(probability=True, random_state=random_state),
                    {"C": [0.5, 1, 5, 10, 50],
                     "gamma": ["scale", 0.001, 0.01, 0.1, 1]}),
    }


def evaluate_predictions(y_true, y_pred):
    return {
        "accuracy":        accuracy_score(y_true, y_pred),
        "f1_macro":        f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted":     f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_macro":    recall_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
    }


def summarize_cv_results(results):
    rows = []
    for name, r in results.items():
        row = {"Model": name}
        for k in ("accuracy", "f1_macro", "f1_weighted", "recall_macro", "precision_macro"):
            v = np.asarray(r[k], dtype=float)
            row[f"{k}_mean"] = float(v.mean()); row[f"{k}_std"] = float(v.std())
        rows.append(row)
    return pd.DataFrame(rows)


def make_group_cv(y, groups, requested_splits, random_state=42):
    gdf = pd.DataFrame({"y": y, "g": groups}).drop_duplicates("g")
    n_splits = max(2, min(int(requested_splits),
                          int(gdf["y"].value_counts().min()),
                          int(gdf["g"].nunique())))
    if HAS_SGKF:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                    random_state=random_state)
    return GroupKFold(n_splits=n_splits)


# ============================================================================
# 10. Nested group cross-validation
# ============================================================================
def nested_group_cv(X_pre, y, conc, groups, wl, feature_set="all",
                    n_outer=5, n_inner=3, random_state=42,
                    compute_importance: bool = True,
                    model_names: Optional[List[str]] = None):
    outer_cv = make_group_cv(y, groups, n_outer, random_state)
    results = defaultdict(lambda: defaultdict(list))
    importance = defaultdict(list)
    feat_names_last = None

    model_specs = make_models(random_state)
    if model_names is not None:
        model_specs = {k: v for k, v in model_specs.items() if k in set(model_names)}
        if not model_specs:
            raise ValueError("model_names contains no usable model. "
                             "Choose from: PLS-DA, RandomForest, SVM-RBF")

    for fold, (tr, te) in enumerate(outer_cv.split(X_pre, y, groups), 1):
        F_tr, F_te, feat_names_last = build_features_train_test(
            X_pre[tr], X_pre[te], wl, feature_set, CONFIG["N_PCA_FEATURES"])
        sc = StandardScaler()
        F_tr_sc = sc.fit_transform(F_tr); F_te_sc = sc.transform(F_te)
        inner = make_group_cv(y[tr], groups[tr], n_inner, random_state + fold)

        for name, (est, grid) in model_specs.items():
            gs = GridSearchCV(est, grid, cv=inner, scoring="f1_macro",
                              n_jobs=CONFIG["N_JOBS"], refit=True)
            gs.fit(F_tr_sc, y[tr], groups=groups[tr])
            best = gs.best_estimator_
            y_pred = best.predict(F_te_sc)

            proba = best.predict_proba(F_te_sc)
            class_to_col = {c: i for i, c in enumerate(best.classes_)}
            cols = [class_to_col[c] for c in CLASS_ORDER]
            proba_aligned = proba[:, cols]

            for k, v in evaluate_predictions(y[te], y_pred).items():
                results[name][k].append(v)
            results[name]["best_params"].append(gs.best_params_)
            results[name]["all_y_true"].extend(y[te].tolist())
            results[name]["all_y_pred"].extend(y_pred.tolist())
            results[name]["all_y_proba"].extend(proba_aligned.tolist())
            results[name]["test_concentration"].extend(conc[te].tolist())

            if compute_importance:
                try:
                    perm = permutation_importance(
                        best, F_te_sc, y[te],
                        n_repeats=20, random_state=random_state + fold,
                        scoring="f1_macro", n_jobs=CONFIG["N_JOBS"])
                    importance[name].append(perm.importances_mean)
                except Exception:
                    pass
        print(f"[nested CV] fold {fold}/{n_outer} done")

    importance_df = {}
    if feat_names_last:
        for name, arrs in importance.items():
            if arrs:
                arr = np.vstack(arrs)
                importance_df[name] = pd.DataFrame({
                    "feature": feat_names_last,
                    "importance_mean": arr.mean(0),
                    "importance_std":  arr.std(0),
                }).sort_values("importance_mean", ascending=False)
    return dict(results), importance_df, feat_names_last


# ============================================================================
# 11. Classical LOCO (all three drugs lose one concentration simultaneously)
# ============================================================================
def leave_one_concentration_out_cv(X_pre, y, conc, groups, wl,
                                   model_name="PLS-DA", feature_set="all",
                                   random_state=42):
    base_est, grid = make_models(random_state)[model_name]
    rows, yt_all, yp_all = [], [], []
    for i, c_hold in enumerate(np.sort(np.unique(conc)), 1):
        te = np.where(conc == c_hold)[0]; tr = np.where(conc != c_hold)[0]
        if len(np.unique(y[tr])) < len(np.unique(y)):
            continue
        F_tr, F_te, _ = build_features_train_test(
            X_pre[tr], X_pre[te], wl, feature_set, CONFIG["N_PCA_FEATURES"])
        sc = StandardScaler(); F_tr_sc = sc.fit_transform(F_tr); F_te_sc = sc.transform(F_te)
        inner = make_group_cv(y[tr], groups[tr], CONFIG["N_INNER"], random_state + i)
        gs = GridSearchCV(clone(base_est), grid, cv=inner, scoring="f1_macro",
                          n_jobs=CONFIG["N_JOBS"], refit=True)
        gs.fit(F_tr_sc, y[tr], groups=groups[tr])
        y_pred = gs.best_estimator_.predict(F_te_sc)
        row = {"held_out_concentration": float(c_hold),
               "n_test": int(len(te)),
               "test_classes": ",".join(sorted(np.unique(y[te]))),
               "best_params": str(gs.best_params_)}
        row.update(evaluate_predictions(y[te], y_pred))
        rows.append(row)
        yt_all.extend(y[te].tolist()); yp_all.extend(y_pred.tolist())
        print(f"[LOCO] held {c_hold:.2e} done")
    return (pd.DataFrame(rows),
            confusion_matrix(yt_all, yp_all, labels=CLASS_ORDER),
            yt_all, yp_all)


# ============================================================================
# 12. LOCO-DC (drug-conditional leave-one-concentration-out)
# ============================================================================
def leave_one_drug_conc_out_cv(X_pre, y, conc, groups, wl,
                               model_name: str = "PLS-DA",
                               feature_set: str = "all",
                               random_state: int = 42):
    """
    Drug-conditional leave-one-concentration-out cross-validation (LOCO-DC).

    For each (drug, concentration) group:
      - test  = replicates of THAT exact (drug, concentration)
      - train = all OTHER samples, i.e.
                  - same drug at its other concentrations
                  - the two other drugs at all concentrations

    More realistic than classical LOCO: it asks whether the model can
    identify a drug at a previously unseen concentration of THAT drug,
    while the other drugs retain full calibration coverage.
    """
    base_est, grid = make_models(random_state)[model_name]
    rows, y_true_all, y_pred_all = [], [], []
    unique_groups = np.unique(groups)

    for i, g_hold in enumerate(unique_groups, 1):
        te = np.where(groups == g_hold)[0]
        tr = np.where(groups != g_hold)[0]

        if len(np.unique(y[tr])) < len(np.unique(y)):
            print(f"[LOCO-DC] skip {g_hold}: training lacks a class")
            continue

        F_tr, F_te, _ = build_features_train_test(
            X_pre[tr], X_pre[te], wl, feature_set, CONFIG["N_PCA_FEATURES"])
        sc = StandardScaler()
        F_tr_sc = sc.fit_transform(F_tr); F_te_sc = sc.transform(F_te)

        inner = make_group_cv(y[tr], groups[tr], CONFIG["N_INNER"],
                              random_state + i)
        gs = GridSearchCV(clone(base_est), grid, cv=inner, scoring="f1_macro",
                          n_jobs=CONFIG["N_JOBS"], refit=True)
        gs.fit(F_tr_sc, y[tr], groups=groups[tr])
        y_pred = gs.best_estimator_.predict(F_te_sc)

        target_drug = y[te][0]
        target_conc = conc[te][0]
        correct     = int(np.sum(y_pred == target_drug))
        total       = len(y_pred)

        rows.append({
            "target_drug":          target_drug,
            "target_concentration": float(target_conc),
            "n_test":               total,
            "n_correct":            correct,
            "accuracy":             correct / total,
            "predictions":          ",".join(y_pred.tolist()),
            "best_params":          str(gs.best_params_),
        })
        y_true_all.extend(y[te].tolist())
        y_pred_all.extend(y_pred.tolist())
        print(f"[LOCO-DC] {target_drug} @ {target_conc:.2e}: {correct}/{total}")

    summary_df = pd.DataFrame(rows)
    cm = confusion_matrix(y_true_all, y_pred_all, labels=CLASS_ORDER)
    return summary_df, cm, y_true_all, y_pred_all


# ============================================================================
# 13. Publication figures
# ============================================================================
def _scatter_class(ax, X2, y, *, s=70, edge="black", lw=0.6, alpha=0.92, legend=True):
    for lab in CLASS_ORDER:
        m = y == lab
        if m.any():
            ax.scatter(X2[m, 0], X2[m, 1], s=s, marker=MARKERS[lab],
                       color=PALETTE[lab], edgecolor=edge, linewidth=lw,
                       alpha=alpha, label=lab)
    if legend:
        ax.legend(loc="best", handletextpad=0.3, borderpad=0.3)


# ---- Fig 01: PCA 3D ---------------------------------------------------------
def plot_pca_3d(X_pre, y, wl, outdir: Path) -> None:
    F, _ = build_features_for_viz(X_pre, wl, "all", CONFIG["N_PCA_FEATURES"])
    F_sc = StandardScaler().fit_transform(F)
    pca = PCA(n_components=3); F3 = pca.fit_transform(F_sc)

    fig = plt.figure(figsize=(WIDTH_2COL * 0.65, WIDTH_2COL * 0.65))
    ax  = fig.add_subplot(111, projection="3d")
    # Manual axes position keeps PC1/PC2/PC3 labels inside the figure.
    ax.set_position([0.12, 0.10, 0.76, 0.80])

    for lab in CLASS_ORDER:
        m = y == lab
        if m.any():
            add_confidence_ellipsoid_3d(F3[m], ax, color=PALETTE[lab],
                                        alpha_surface=0.08, n_grid=22)

    for lab in CLASS_ORDER:
        m = y == lab
        if m.any():
            ax.scatter(F3[m, 0], F3[m, 1], F3[m, 2],
                       marker=MARKERS[lab], color=PALETTE[lab],
                       edgecolor="black", linewidth=0.4, s=45,
                       label=lab, alpha=0.95, depthshade=False)

    pastel = {"x": "#FFF8E7", "y": "#EAF4FB", "z": "#F4ECF7"}
    ax.xaxis.pane.set_facecolor(pastel["x"]); ax.xaxis.pane.set_edgecolor("0.5")
    ax.yaxis.pane.set_facecolor(pastel["y"]); ax.yaxis.pane.set_edgecolor("0.5")
    ax.zaxis.pane.set_facecolor(pastel["z"]); ax.zaxis.pane.set_edgecolor("0.5")
    ax.grid(True, alpha=0.3, linewidth=0.4)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", labelpad=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", labelpad=12)
    ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)", labelpad=10)
    ax.tick_params(axis="both", which="major", pad=2)
    ax.set_box_aspect((1, 1, 0.85))
    ax.view_init(elev=22, azim=-55)
    ax.legend(loc="upper left", handletextpad=0.3, bbox_to_anchor=(-0.05, 0.98))

    if CONFIG["SAVE_FIGURES"]:
        for fmt in ("png", "pdf", "tiff"):
            path = outdir / f"Fig_01_PCA_3D.{fmt}"
            try:
                if fmt == "tiff":
                    fig.savefig(path, dpi=600, format="tiff",
                                bbox_inches=None, pad_inches=0.4,
                                pil_kwargs={"compression": "tiff_lzw"})
                elif fmt == "pdf":
                    fig.savefig(path, format="pdf",
                                bbox_inches=None, pad_inches=0.4)
                else:
                    fig.savefig(path, dpi=300, format=fmt,
                                bbox_inches=None, pad_inches=0.4)
            except Exception as exc:
                print(f"  [warn] save {fmt} failed: {exc}")
    plt.close(fig)


# ---- Fig 02: PCA 2D ---------------------------------------------------------
def plot_pca_2d(X_pre, y, wl, outdir: Path) -> None:
    F, _ = build_features_for_viz(X_pre, wl, "all", CONFIG["N_PCA_FEATURES"])
    F_sc = StandardScaler().fit_transform(F)
    pca = PCA(n_components=2); F2 = pca.fit_transform(F_sc)

    fig, ax = plt.subplots(figsize=(WIDTH_15COL * 0.7, WIDTH_15COL * 0.6))
    for lab in CLASS_ORDER:
        m = y == lab
        if m.any():
            add_confidence_ellipse_2d(F2[m, 0], F2[m, 1], ax, color=PALETTE[lab])
    _scatter_class(ax, F2, y, s=65)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.axhline(0, color="0.7", lw=0.4, ls=":")
    ax.axvline(0, color="0.7", lw=0.4, ls=":")
    fig.tight_layout()
    save_publication(fig, outdir, "Fig_02_PCA_2D")
    plt.close(fig)


# ---- Fig 03: t-SNE ----------------------------------------------------------
def plot_tsne(X_pre, y, wl, outdir: Path) -> None:
    F, _ = build_features_for_viz(X_pre, wl, "all", CONFIG["N_PCA_FEATURES"])
    F_sc = StandardScaler().fit_transform(F)
    perplex = min(10, max(3, len(y) // 5))
    try:
        tsne = TSNE(n_components=2, perplexity=perplex, learning_rate="auto",
                    max_iter=2000, init="pca", random_state=CONFIG["RANDOM_STATE"])
    except TypeError:
        tsne = TSNE(n_components=2, perplexity=perplex, learning_rate="auto",
                    n_iter=2000, init="pca", random_state=CONFIG["RANDOM_STATE"])
    F_tsne = tsne.fit_transform(F_sc)

    fig, ax = plt.subplots(figsize=(WIDTH_15COL * 0.7, WIDTH_15COL * 0.6))
    for lab in CLASS_ORDER:
        m = y == lab
        if m.any():
            add_confidence_ellipse_2d(F_tsne[m, 0], F_tsne[m, 1], ax, color=PALETTE[lab])
    _scatter_class(ax, F_tsne, y, s=65)
    ax.set_xlabel(f"t-SNE 1 (perplexity = {perplex})")
    ax.set_ylabel("t-SNE 2")
    fig.tight_layout()
    save_publication(fig, outdir, "Fig_03_tSNE")
    plt.close(fig)


# ---- Circular dendrogram (plurality-vote branch colouring) -----------------
def circular_dendrogram(Z: np.ndarray, leaf_labels: List[str],
                        leaf_categories: List[str], category_palette: Dict[str, str],
                        ax, leaf_font_size: float = 4.5,
                        line_width: float = 0.8,
                        mixed_color: str = "#999999",
                        center_text: Optional[str] = None,
                        radial_padding_frac: float = 0.06) -> None:
    """
    Radial (circular) dendrogram with plurality-vote branch colouring.

    Branch colouring: the dominant category among the leaves underneath
    a branch determines that branch's colour; ties fall back to mixed_color.
    """
    n = len(leaf_labels)
    ddata = dendrogram(Z, no_plot=True, get_leaves=True)
    icoord_list = ddata["icoord"]
    dcoord_list = ddata["dcoord"]
    leaves_order = ddata["leaves"]

    max_x = 10.0 * n
    max_dist = max(max(dc) for dc in dcoord_list) if dcoord_list else 1.0

    def x_to_angle(x: float) -> float:
        gap = 0.03
        return (x / max_x) * (2 * np.pi) * (1 - gap) + (np.pi * gap)

    leaf_at_x: Dict[int, int] = {5 + 10 * i: leaves_order[i] for i in range(n)}
    sorted_xs = sorted(leaf_at_x.keys())

    def branch_color(x_left: float, x_right: float) -> str:
        cats = []
        for x in sorted_xs:
            if x_left - 0.5 <= x <= x_right + 0.5:
                cats.append(leaf_categories[leaf_at_x[x]])
        if not cats:
            return mixed_color
        counter = Counter(cats)
        ranked = counter.most_common()
        top_cat, top_count = ranked[0]
        second_count = ranked[1][1] if len(ranked) > 1 else 0
        if top_count == second_count:
            return mixed_color
        return category_palette.get(top_cat, mixed_color)

    for ic, dc in zip(icoord_list, dcoord_list):
        x_left, x_right = ic[0], ic[3]
        d_left, d_top, d_right = dc[0], dc[1], dc[3]
        color = branch_color(x_left, x_right)

        a_l = x_to_angle(x_left); a_r = x_to_angle(x_right)
        r_l = max_dist - d_left
        r_t = max_dist - d_top
        r_r = max_dist - d_right

        ax.plot([a_l, a_l], [r_l, r_t], color=color, lw=line_width,
                solid_capstyle="round")
        ax.plot([a_r, a_r], [r_r, r_t], color=color, lw=line_width,
                solid_capstyle="round")
        if a_r > a_l:
            arc_n = max(2, int(60 * (a_r - a_l) / (2 * np.pi)) + 2)
            arc_a = np.linspace(a_l, a_r, arc_n)
            ax.plot(arc_a, [r_t] * arc_n, color=color, lw=line_width,
                    solid_capstyle="round")

    label_radius = max_dist * (1.0 + radial_padding_frac)
    for i in range(n):
        x_pos = 5 + 10 * i
        ang   = x_to_angle(x_pos)
        orig  = leaves_order[i]
        visual_deg = 90 - np.degrees(ang)
        if -90 <= visual_deg <= 90:
            text_rot, ha = visual_deg, "left"
        else:
            text_rot, ha = visual_deg + 180, "right"
        ax.text(ang, label_radius, leaf_labels[orig],
                rotation=text_rot, rotation_mode="anchor",
                ha=ha, va="center",
                fontsize=leaf_font_size,
                color=category_palette.get(leaf_categories[orig], "0.3"))

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim(0, max_dist * 1.55)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines["polar"].set_visible(False)
    ax.set_facecolor("white")

    if center_text:
        ax.text(0, 0, center_text, ha="center", va="center", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                          edgecolor="0.7", lw=0.5))


def _hca_metrics(Z: np.ndarray, y: np.ndarray, n_clusters: int = 3) -> Dict[str, float]:
    lbl = fcluster(Z, t=n_clusters, criterion="maxclust")
    purity = sum(max(np.sum((y == lab) & (lbl == cl)) for lab in np.unique(y))
                 for cl in np.unique(lbl)) / len(y)
    return {"purity": purity,
            "ARI":    float(adjusted_rand_score(y, lbl)),
            "NMI":    float(normalized_mutual_info_score(y, lbl))}


def _short_leaf_label(ab: str, conc_val: float, rep: int) -> str:
    return f"{ab}{conc_val:.0e}r{rep}".replace("e-0", "e-")


# ---- Fig 04: HCA unsupervised -----------------------------------------------
def plot_hca_unsupervised(X_pre, y, conc, replicate, wl, outdir: Path) -> Dict[str, float]:
    F, _ = build_features_for_viz(X_pre, wl, "all", CONFIG["N_PCA_FEATURES"])
    F_sc = StandardScaler().fit_transform(F)
    Z = linkage(F_sc, method="ward")
    metrics = _hca_metrics(Z, y, n_clusters=3)

    cluster_lbl = fcluster(Z, t=3, criterion="maxclust")
    cluster_palette = {
        1: "#0072B2", 2: "#D55E00", 3: "#009E73",
        4: "#CC79A7", 5: "#56B4E9", 6: "#E69F00",
    }
    leaf_categories = [int(c) for c in cluster_lbl]
    palette = {k: v for k, v in cluster_palette.items() if k in set(leaf_categories)}
    leaf_labels = [_short_leaf_label(y[i], conc[i], replicate[i]) for i in range(len(y))]

    fig = plt.figure(figsize=(WIDTH_2COL * 0.7, WIDTH_2COL * 0.7))
    ax = fig.add_subplot(111, projection="polar")
    circular_dendrogram(Z, leaf_labels, leaf_categories, palette, ax,
                        leaf_font_size=4.5,
                        center_text=(f"purity = {metrics['purity']:.3f}\n"
                                     f"ARI = {metrics['ARI']:.3f}\n"
                                     f"NMI = {metrics['NMI']:.3f}"))
    fig.tight_layout()
    save_publication(fig, outdir, "Fig_04_HCA_unsupervised")
    plt.close(fig)
    return metrics


# ---- Fig 05: HCA supervised -------------------------------------------------
def plot_hca_supervised(X_pre, y, conc, replicate, groups, wl, outdir: Path,
                        n_components: int = 3) -> Dict[str, float]:
    F, _ = build_features_for_viz(X_pre, wl, "all", CONFIG["N_PCA_FEATURES"])
    F_sc = StandardScaler().fit_transform(F)
    plsda = PLSDA(n_components=n_components).fit(F_sc, y)
    LV = plsda.pls_.x_scores_
    Z = linkage(LV, method="ward")
    metrics = _hca_metrics(Z, y, n_clusters=3)

    leaf_categories = list(y)
    leaf_labels = [_short_leaf_label(y[i], conc[i], replicate[i]) for i in range(len(y))]

    fig = plt.figure(figsize=(WIDTH_2COL * 0.7, WIDTH_2COL * 0.7))
    ax = fig.add_subplot(111, projection="polar")
    circular_dendrogram(Z, leaf_labels, leaf_categories, PALETTE, ax,
                        leaf_font_size=4.5,
                        center_text=(f"purity = {metrics['purity']:.3f}\n"
                                     f"ARI = {metrics['ARI']:.3f}\n"
                                     f"NMI = {metrics['NMI']:.3f}"))
    fig.tight_layout()
    save_publication(fig, outdir, "Fig_05_HCA_supervised")
    plt.close(fig)
    return metrics


# ---- Fig 06-08: Confusion matrices ------------------------------------------
def plot_confusion_matrix(results: Dict, model_name: str, outdir: Path,
                          stem: str) -> None:
    r = results[model_name]
    cm = confusion_matrix(r["all_y_true"], r["all_y_pred"], labels=CLASS_ORDER)
    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    fig, ax = plt.subplots(figsize=(WIDTH_1COL * 1.35, WIDTH_1COL * 1.25))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASS_ORDER))); ax.set_yticks(range(len(CLASS_ORDER)))
    ax.set_xticklabels(CLASS_ORDER); ax.set_yticklabels(CLASS_ORDER)
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")

    for i in range(len(CLASS_ORDER)):
        for j in range(len(CLASS_ORDER)):
            txt_color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)",
                    ha="center", va="center", fontsize=9, color=txt_color,
                    fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized count", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_linewidth(0.5)

    fig.tight_layout()
    save_publication(fig, outdir, stem)
    plt.close(fig)


# ---- Fig 09-11: ROC (one-vs-rest) -------------------------------------------
def plot_roc_curve(results: Dict, model_name: str, outdir: Path,
                   stem: str) -> Dict[str, float]:
    r = results[model_name]
    y_true  = np.array(r["all_y_true"])
    y_proba = np.array(r["all_y_proba"])
    y_bin   = label_binarize(y_true, classes=CLASS_ORDER)

    fig, ax = plt.subplots(figsize=(WIDTH_15COL * 0.7, WIDTH_15COL * 0.65))
    aucs: Dict[str, float] = {}
    for i, cls in enumerate(CLASS_ORDER):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr); aucs[cls] = float(roc_auc)
        ax.plot(fpr, tpr, lw=2.0, color=PALETTE[cls],
                label=f"{cls} (AUC = {roc_auc:.3f})")

    macro_auc = float(np.mean(list(aucs.values())))
    ax.plot([0, 1], [0, 1], color="0.55", lw=0.8, ls="--",
            label="Random (AUC = 0.500)")
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.text(0.98, 0.02, f"macro-AUC = {macro_auc:.3f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="0.7", lw=0.5))
    ax.legend(loc="lower right", borderpad=0.4, bbox_to_anchor=(0.99, 0.14))
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25, lw=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    save_publication(fig, outdir, stem)
    plt.close(fig)
    return aucs


# ============================================================================
# 14. CSV export
# ============================================================================
def export_feature_importance_combined(importance_dict: Dict, vip: pd.Series,
                                       outdir: Path) -> pd.DataFrame:
    df = pd.DataFrame({"feature": vip.index, "PLSDA_VIP": vip.values})
    for model_key, col_prefix in [("RandomForest", "RF"), ("SVM-RBF", "SVM")]:
        if model_key in importance_dict:
            imp = importance_dict[model_key].set_index("feature")
            df[f"{col_prefix}_perm_importance_mean"] = (
                df["feature"].map(imp["importance_mean"]).fillna(0.0))
            df[f"{col_prefix}_perm_importance_std"] = (
                df["feature"].map(imp["importance_std"]).fillna(0.0))
        else:
            df[f"{col_prefix}_perm_importance_mean"] = np.nan
            df[f"{col_prefix}_perm_importance_std"]  = np.nan

    df["description"] = df["feature"].apply(describe_feature)

    cols = ["feature", "description"] + [c for c in df.columns
                                          if c not in ("feature", "description")]
    df = df[cols].sort_values("PLSDA_VIP", ascending=False).reset_index(drop=True)
    df.to_csv(outdir / "feature_importance_combined.csv", index=False)
    return df


def export_metrics_summary(cv_summary: pd.DataFrame, results: Dict,
                           all_aucs: Dict, loco_summary: pd.DataFrame,
                           outdir: Path) -> None:
    metrics = ["accuracy", "f1_macro", "recall_macro", "precision_macro"]
    rows = []
    for _, row in cv_summary.iterrows():
        for m in metrics:
            rows.append({"Model": row["Model"], "Metric": m,
                         "Mean": row[f"{m}_mean"], "Std": row[f"{m}_std"]})
    pd.DataFrame(rows).to_csv(outdir / "metrics_summary_long.csv", index=False)
    cv_summary.to_csv(outdir / "metrics_summary_wide.csv", index=False)

    cm_rows = []
    for model_name in cv_summary["Model"]:
        r = results[model_name]
        y_true = np.array(r["all_y_true"]); y_pred = np.array(r["all_y_pred"])
        cm = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
        per_class_precision = precision_score(y_true, y_pred, labels=CLASS_ORDER,
                                              average=None, zero_division=0)
        per_class_recall    = recall_score(y_true, y_pred, labels=CLASS_ORDER,
                                           average=None, zero_division=0)
        per_class_f1        = f1_score(y_true, y_pred, labels=CLASS_ORDER,
                                       average=None, zero_division=0)
        d = {"Model": model_name,
             "Accuracy":        accuracy_score(y_true, y_pred),
             "F1_macro":        f1_score(y_true, y_pred, average="macro",       zero_division=0),
             "F1_weighted":     f1_score(y_true, y_pred, average="weighted",    zero_division=0),
             "Recall_macro":    recall_score(y_true, y_pred, average="macro",   zero_division=0),
             "Precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0)}
        for i, cls in enumerate(CLASS_ORDER):
            d[f"Precision_{cls}"] = per_class_precision[i]
            d[f"Recall_{cls}"]    = per_class_recall[i]
            d[f"F1_{cls}"]        = per_class_f1[i]
        for cls, val in all_aucs.get(model_name, {}).items():
            d[f"AUC_{cls}"] = float(val)
        if model_name in all_aucs:
            d["macro_AUC"] = float(np.mean(list(all_aucs[model_name].values())))
        cm_rows.append(d)
    pd.DataFrame(cm_rows).to_csv(outdir / "cm_metrics_per_model.csv", index=False)

    if len(loco_summary):
        loco_summary.to_csv(outdir / "leave_one_concentration_out_summary.csv",
                            index=False)


# ============================================================================
# 15. Random-label test and feature ablation
# ============================================================================
def permute_labels_by_group(y: np.ndarray, groups: np.ndarray,
                            random_state: int = 42) -> np.ndarray:
    """
    Group-level random-label permutation.

    Labels cannot be shuffled per sample because the n_replicates measurements
    of any (drug, concentration) condition must remain in one group. Instead:
      1. retrieve each group's original class label
      2. permute the class labels among groups
      3. broadcast the permuted label back to all samples of that group
    This preserves the class-count balance and the replicate group structure
    while breaking the (feature, label) association.
    """
    rng = np.random.default_rng(random_state)
    unique_groups = np.array(sorted(np.unique(groups)))
    group_labels = np.array([y[groups == g][0] for g in unique_groups])
    permuted_labels = rng.permutation(group_labels)
    mapping = {g: lab for g, lab in zip(unique_groups, permuted_labels)}
    return np.array([mapping[g] for g in groups])


def plot_random_label_accuracy(random_long: pd.DataFrame, outdir: Path) -> None:
    if random_long.empty or "accuracy_mean" not in random_long.columns:
        return
    models = list(random_long["Model"].drop_duplicates())
    data = [random_long.loc[random_long["Model"] == m, "accuracy_mean"].to_numpy(dtype=float)
            for m in models]

    fig, ax = plt.subplots(figsize=(WIDTH_15COL * 0.75, WIDTH_15COL * 0.55))
    ax.boxplot(data, labels=models, showmeans=True)
    ax.axhline(1 / len(CLASS_ORDER), color="0.4", lw=1.0, ls="--",
               label=f"Chance level = {1/len(CLASS_ORDER):.3f}")
    ax.set_ylabel("Accuracy under random labels")
    ax.set_xlabel("Model")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25, lw=0.4)
    fig.tight_layout()
    save_publication(fig, outdir, "Fig_12_random_label_test")
    plt.close(fig)


def run_random_label_test(X_pre: np.ndarray, y: np.ndarray, conc: np.ndarray,
                          groups: np.ndarray, wl: np.ndarray, outdir: Path,
                          n_repeats: int = 20,
                          model_names: Optional[List[str]] = None,
                          feature_set: str = "all",
                          random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Random-label test.

    If real-label accuracy is high while random-label accuracy is close to
    1/n_classes, the real performance cannot be attributed to data leakage,
    group structure or the pipeline itself.
    """
    long_rows = []

    for rep in range(1, n_repeats + 1):
        rs = random_state + 1000 + rep
        y_perm = permute_labels_by_group(y, groups, random_state=rs)

        for g in np.unique(groups):
            labs = np.unique(y_perm[groups == g])
            if len(labs) != 1:
                raise RuntimeError(f"random-label test broke group {g}: multiple labels.")

        cv_results_perm, _, _ = nested_group_cv(
            X_pre, y_perm, conc, groups, wl,
            feature_set=feature_set,
            n_outer=CONFIG["N_OUTER"],
            n_inner=CONFIG["N_INNER"],
            random_state=rs,
            compute_importance=False,
            model_names=model_names,
        )
        summary = summarize_cv_results(cv_results_perm)
        for _, row in summary.iterrows():
            d = row.to_dict()
            d["repeat"] = rep
            d["feature_set"] = feature_set
            d["chance_level"] = 1 / len(CLASS_ORDER)
            long_rows.append(d)

        acc_str = ", ".join(
            f"{r['Model']}={r['accuracy_mean']:.3f}" for _, r in summary.iterrows()
        )
        print(f"[random-label] repeat {rep}/{n_repeats}: {acc_str}")

    random_long = pd.DataFrame(long_rows)
    random_long.to_csv(outdir / "random_label_test_long.csv", index=False)

    summary_rows = []
    for model, g in random_long.groupby("Model"):
        row = {"Model": model,
               "n_repeats": int(g["repeat"].nunique()),
               "chance_level": 1 / len(CLASS_ORDER)}
        for metric in ("accuracy_mean", "f1_macro_mean",
                       "recall_macro_mean", "precision_macro_mean"):
            if metric in g.columns:
                row[f"{metric}_mean"] = float(g[metric].mean())
                row[f"{metric}_std"]  = float(g[metric].std(ddof=1))
                row[f"{metric}_min"]  = float(g[metric].min())
                row[f"{metric}_max"]  = float(g[metric].max())
        summary_rows.append(row)

    random_summary = pd.DataFrame(summary_rows)
    random_summary.to_csv(outdir / "random_label_test_summary.csv", index=False)
    plot_random_label_accuracy(random_long, outdir)
    return random_long, random_summary


def plot_ablation_accuracy(ablation_df: pd.DataFrame, outdir: Path) -> None:
    if ablation_df.empty:
        return
    pivot = ablation_df.pivot_table(index="feature_set", columns="Model",
                                    values="accuracy_mean", aggfunc="mean")
    order = [fs for fs in CONFIG["ABLATION_FEATURE_SETS"] if fs in pivot.index]
    pivot = pivot.loc[order]

    fig, ax = plt.subplots(figsize=(WIDTH_2COL * 0.85, WIDTH_15COL * 0.55))
    x = np.arange(len(pivot.index))
    models = list(pivot.columns)
    width = 0.8 / max(1, len(models))
    for i, model in enumerate(models):
        ax.bar(x + (i - (len(models) - 1) / 2) * width,
               pivot[model].to_numpy(dtype=float),
               width=width, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=35, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Feature set")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", ncol=min(3, len(models)))
    ax.grid(axis="y", alpha=0.25, lw=0.4)
    fig.tight_layout()
    save_publication(fig, outdir, "Fig_13_ablation_accuracy")
    plt.close(fig)


def run_ablation_study(X_pre: np.ndarray, y: np.ndarray, conc: np.ndarray,
                       groups: np.ndarray, wl: np.ndarray, outdir: Path,
                       feature_sets: Optional[List[str]] = None,
                       model_names: Optional[List[str]] = None,
                       random_state: int = 42) -> pd.DataFrame:
    """
    Feature-family ablation study. Uses the same outer group CV and inner
    group grid-search as the main pipeline. Permutation importance is not
    computed inside ablation to keep runtime tractable.
    """
    feature_sets = feature_sets or list(_FEATURE_SETS.keys())
    rows = []

    for fs in feature_sets:
        if fs not in _FEATURE_SETS:
            raise ValueError(f"Unknown feature_set: {fs}. "
                             f"Choose from: {sorted(_FEATURE_SETS)}")

        _, feat_names = build_features_for_viz(X_pre, wl, fs, CONFIG["N_PCA_FEATURES"])
        n_features = len(feat_names)

        print(f"[ablation] feature_set={fs} | n_features={n_features}")
        cv_results_fs, _, _ = nested_group_cv(
            X_pre, y, conc, groups, wl,
            feature_set=fs,
            n_outer=CONFIG["N_OUTER"],
            n_inner=CONFIG["N_INNER"],
            random_state=random_state + 2000 + len(rows),
            compute_importance=False,
            model_names=model_names,
        )
        summary = summarize_cv_results(cv_results_fs)
        for _, row in summary.iterrows():
            d = row.to_dict()
            d["feature_set"] = fs
            d["n_features"] = n_features
            rows.append(d)

    ablation_df = pd.DataFrame(rows)
    first_cols = ["feature_set", "n_features", "Model"]
    other_cols = [c for c in ablation_df.columns if c not in first_cols]
    ablation_df = ablation_df[first_cols + other_cols]
    ablation_df.to_csv(outdir / "ablation_study_all_models.csv", index=False)

    best = (ablation_df.sort_values(["feature_set", "f1_macro_mean"],
                                    ascending=[True, False])
            .groupby("feature_set", as_index=False)
            .head(1))
    best.to_csv(outdir / "ablation_study_best_by_feature_set.csv", index=False)

    best_model = (ablation_df.sort_values(["Model", "f1_macro_mean"],
                                          ascending=[True, False])
                  .groupby("Model", as_index=False)
                  .head(1))
    best_model.to_csv(outdir / "ablation_study_best_by_model.csv", index=False)

    plot_ablation_accuracy(ablation_df, outdir)
    return ablation_df


# ============================================================================
# 16. Chemical closure report
# ============================================================================
def chemical_closure_report(importance_df, vip: pd.Series, outdir: Path):
    top_perm = " ".join(importance_df.head(30)["feature"].astype(str).tolist()) \
        if importance_df is not None else ""
    top_vip  = " ".join(vip.head(30).index.astype(str).tolist())
    combined = top_perm + " " + top_vip
    checks = {
        "NFLX_446_related":  any(k in combined for k in ("NFLX", "446", "440_452")),
        "LVLX_454_related":  any(k in combined for k in ("LVLX", "454", "448_462")),
        "CIP_492_related":   any(k in combined for k in ("CIP",  "492", "485_500")),
        "Probe_617_related": any(k in combined for k in ("617",  "605_630")),
    }
    df = pd.DataFrame([{"chemical_signal": k, "in_top_features": bool(v)}
                       for k, v in checks.items()])
    df.to_csv(outdir / "chemical_closure_report.csv", index=False)
    return df


# ============================================================================
# 17. Command-line interface
# ============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=("Machine-learning pipeline for replicate fluorescence "
                     "spectra of quinolone antibiotics with an Eu-MOF probe."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", type=Path, required=True,
                   help="Path to the input CSV (FL_Data.csv).")
    p.add_argument("--outdir", type=Path, default=Path("./results"),
                   help="Output directory for figures and CSVs.")
    p.add_argument("--n-outer", type=int, default=5,
                   help="Number of outer cross-validation folds.")
    p.add_argument("--n-inner", type=int, default=3,
                   help="Number of inner cross-validation folds.")
    p.add_argument("--n-jobs", type=int, default=1,
                   help="Number of parallel jobs for sklearn estimators.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed.")
    p.add_argument("--no-loco", action="store_true",
                   help="Skip classical leave-one-concentration-out.")
    p.add_argument("--no-loco-dc", action="store_true",
                   help="Skip drug-conditional leave-one-concentration-out.")
    p.add_argument("--no-figures", action="store_true",
                   help="Do not save figure files (only CSVs).")
    p.add_argument("--run-random-label", action="store_true",
                   help="Run the random-label permutation test.")
    p.add_argument("--random-label-repeats", type=int, default=20,
                   help="Number of random-label permutation repeats.")
    p.add_argument("--run-ablation", action="store_true",
                   help="Run the feature-family ablation study.")
    return p.parse_args()


# ============================================================================
# 18. Main
# ============================================================================
def main() -> None:
    args = parse_args()
    CONFIG["DATA_PATH"]               = args.data
    CONFIG["OUTDIR"]                  = args.outdir
    CONFIG["N_OUTER"]                 = args.n_outer
    CONFIG["N_INNER"]                 = args.n_inner
    CONFIG["N_JOBS"]                  = args.n_jobs
    CONFIG["RANDOM_STATE"]            = args.seed
    CONFIG["SAVE_FIGURES"]            = not args.no_figures
    CONFIG["RUN_LOCO"]                = not args.no_loco
    CONFIG["RUN_LOCO_DC"]             = not args.no_loco_dc
    CONFIG["RUN_RANDOM_LABEL_TEST"]   = args.run_random_label
    CONFIG["RANDOM_LABEL_N_REPEATS"]  = args.random_label_repeats
    CONFIG["RUN_ABLATION"]            = args.run_ablation

    if not CONFIG["DATA_PATH"].exists():
        raise FileNotFoundError(f"Data file not found: {CONFIG['DATA_PATH']}")

    outdir = CONFIG["OUTDIR"]; outdir.mkdir(parents=True, exist_ok=True)
    setup_talanta_style()

    print("=" * 78, "\n1. Load data\n", "=" * 78, sep="")
    X_raw, y, conc, replicate, wavelengths, groups, _ = \
        load_replicate_data(CONFIG["DATA_PATH"])
    print(f"  samples: {X_raw.shape[0]}, wavelengths: {X_raw.shape[1]}")
    print(f"  class distribution: {dict(pd.Series(y).value_counts())}")

    print("\n" + "=" * 78, "\n2. Preprocess\n", "=" * 78, sep="")
    X_pre, wl = preprocess_spectra(X_raw, wavelengths)
    print(f"  after preprocessing: {X_pre.shape}, "
          f"range {wl[0]:.1f}-{wl[-1]:.1f} nm")

    print("\n" + "=" * 78, "\n3. Replicate QC\n", "=" * 78, sep="")
    qc, qc_summary = replicate_qc_peak_rsd(X_pre, wl, y, conc, groups, outdir)
    print(qc_summary)

    print("\n" + "=" * 78, "\n4. Publication figures: PCA / t-SNE / HCA\n",
          "=" * 78, sep="")
    plot_pca_3d(X_pre, y, wl, outdir);   print("  Fig_01_PCA_3D done")
    plot_pca_2d(X_pre, y, wl, outdir);   print("  Fig_02_PCA_2D done")
    plot_tsne(X_pre, y, wl, outdir);     print("  Fig_03_tSNE done")
    hca_u = plot_hca_unsupervised(X_pre, y, conc, replicate, wl, outdir)
    print(f"  Fig_04_HCA_unsupervised done | purity={hca_u['purity']:.3f}")
    hca_s = plot_hca_supervised(X_pre, y, conc, replicate, groups, wl, outdir)
    print(f"  Fig_05_HCA_supervised   done | purity={hca_s['purity']:.3f}")

    print("\n" + "=" * 78, "\n5. Nested group CV\n", "=" * 78, sep="")
    cv_results, importance_dict, _ = nested_group_cv(
        X_pre, y, conc, groups, wl, "all",
        CONFIG["N_OUTER"], CONFIG["N_INNER"], CONFIG["RANDOM_STATE"])
    cv_summary = summarize_cv_results(cv_results)
    cv_summary.to_csv(outdir / "cv_summary_grouped.csv", index=False)
    print(cv_summary.to_string(index=False))

    print("\n" + "=" * 78, "\n6. Confusion matrices and ROC\n", "=" * 78, sep="")
    model_files = [
        ("PLS-DA",       "Fig_06_CM_PLSDA",  "Fig_09_ROC_PLSDA"),
        ("RandomForest", "Fig_07_CM_RF",     "Fig_10_ROC_RF"),
        ("SVM-RBF",      "Fig_08_CM_SVM",    "Fig_11_ROC_SVM"),
    ]
    all_aucs = {}
    for model_name, cm_stem, roc_stem in model_files:
        plot_confusion_matrix(cv_results, model_name, outdir, cm_stem)
        aucs = plot_roc_curve(cv_results, model_name, outdir, roc_stem)
        all_aucs[model_name] = aucs
        print(f"  {model_name}: CM + ROC done | " +
              ", ".join(f"AUC_{k}={v:.3f}" for k, v in aucs.items()))

    print("\n" + "=" * 78, "\n7. Classical LOCO\n", "=" * 78, sep="")
    loco_summary = pd.DataFrame()
    if CONFIG["RUN_LOCO"]:
        loco_summary, loco_cm, _, _ = leave_one_concentration_out_cv(
            X_pre, y, conc, groups, wl, "PLS-DA", "all", CONFIG["RANDOM_STATE"])
        pd.DataFrame(loco_cm, index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(
            outdir / "leave_one_concentration_out_confusion_matrix.csv")
        print(loco_summary[["held_out_concentration", "accuracy", "f1_macro"]]
              .to_string(index=False))
    else:
        print("  skipped.")

    print("\n" + "=" * 78, "\n8. Drug-conditional LOCO-DC\n", "=" * 78, sep="")
    if CONFIG["RUN_LOCO_DC"]:
        locodc_summary, locodc_cm, _, _ = leave_one_drug_conc_out_cv(
            X_pre, y, conc, groups, wl,
            model_name="PLS-DA", feature_set="all",
            random_state=CONFIG["RANDOM_STATE"])

        locodc_summary.to_csv(
            outdir / "leave_one_drug_concentration_out_summary.csv", index=False)
        pd.DataFrame(locodc_cm, index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(
            outdir / "leave_one_drug_concentration_out_confusion_matrix.csv")

        by_drug = (locodc_summary.groupby("target_drug")
                   .agg(n_total=("n_test", "sum"),
                        n_correct=("n_correct", "sum"))
                   .assign(accuracy=lambda d: d["n_correct"] / d["n_total"]))
        by_drug.to_csv(outdir / "leave_one_drug_concentration_out_by_drug.csv")

        overall = (locodc_summary["n_correct"].sum() /
                   locodc_summary["n_test"].sum())
        print(f"  Overall LOCO-DC accuracy: {overall*100:.1f}% "
              f"({locodc_summary['n_correct'].sum()}/"
              f"{locodc_summary['n_test'].sum()})")
        print(by_drug.to_string())
    else:
        print("  skipped.")

    random_label_summary = pd.DataFrame()
    ablation_summary = pd.DataFrame()

    print("\n" + "=" * 78, "\n9. Random-label test\n", "=" * 78, sep="")
    if CONFIG["RUN_RANDOM_LABEL_TEST"]:
        _, random_label_summary = run_random_label_test(
            X_pre, y, conc, groups, wl, outdir,
            n_repeats=int(CONFIG["RANDOM_LABEL_N_REPEATS"]),
            model_names=CONFIG.get("RANDOM_LABEL_MODELS", None),
            feature_set="all",
            random_state=CONFIG["RANDOM_STATE"])
        print(random_label_summary.to_string(index=False))
    else:
        print("  skipped.")

    print("\n" + "=" * 78, "\n10. Feature ablation\n", "=" * 78, sep="")
    if CONFIG["RUN_ABLATION"]:
        ablation_summary = run_ablation_study(
            X_pre, y, conc, groups, wl, outdir,
            feature_sets=CONFIG.get("ABLATION_FEATURE_SETS", None),
            model_names=CONFIG.get("ABLATION_MODELS", None),
            random_state=CONFIG["RANDOM_STATE"])
        print(ablation_summary[["feature_set", "Model", "n_features",
                                "accuracy_mean", "f1_macro_mean"]]
              .to_string(index=False))
    else:
        print("  skipped.")

    print("\n" + "=" * 78, "\n11. Final PLS-DA + VIP\n", "=" * 78, sep="")
    F, _ = build_features_for_viz(X_pre, wl, "all", CONFIG["N_PCA_FEATURES"])
    F_sc = StandardScaler().fit_transform(F)
    inner = make_group_cv(y, groups, CONFIG["N_INNER"], CONFIG["RANDOM_STATE"])
    gs = GridSearchCV(PLSDA(), {"n_components": [2, 3, 4, 5, 6, 8, 10]},
                      cv=inner, scoring="f1_macro",
                      n_jobs=CONFIG["N_JOBS"], refit=True)
    gs.fit(F_sc, y, groups=groups)
    feat_names_full = build_features_for_viz(
        X_pre, wl, "all", CONFIG["N_PCA_FEATURES"])[1]
    vip = calculate_pls_vip(gs.best_estimator_, feat_names_full)
    vip.to_csv(outdir / "plsda_vip_scores.csv", header=["VIP"])
    print(f"  Best n_components: {gs.best_params_['n_components']}")

    print("\n" + "=" * 78, "\n12. Export CSVs\n", "=" * 78, sep="")
    fi_combined = export_feature_importance_combined(importance_dict, vip, outdir)
    print(f"  feature_importance_combined.csv: {len(fi_combined)} features")
    export_metrics_summary(cv_summary, cv_results, all_aucs, loco_summary, outdir)
    print("  metrics_summary_long.csv, metrics_summary_wide.csv, "
          "cm_metrics_per_model.csv done")

    print("\n" + "=" * 78, "\n13. Chemical closure check\n", "=" * 78, sep="")
    best_imp = importance_dict.get(
        cv_summary.sort_values("f1_macro_mean", ascending=False).iloc[0]["Model"])
    closure = chemical_closure_report(best_imp, vip, outdir)
    print(closure.to_string(index=False))

    with open(outdir / "results_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_samples": int(X_raw.shape[0]),
            "n_groups":  int(len(np.unique(groups))),
            "n_concentrations": int(len(np.unique(conc))),
            "class_counts": {str(k): int(v)
                             for k, v in pd.Series(y).value_counts().items()},
            "cv_summary": cv_summary.to_dict("records"),
            "auc_per_model": {m: {c: float(v) for c, v in d.items()}
                              for m, d in all_aucs.items()},
            "hca_unsupervised": {k: float(v) for k, v in hca_u.items()},
            "hca_supervised":   {k: float(v) for k, v in hca_s.items()},
            "random_label_test_summary":
                random_label_summary.to_dict("records")
                if isinstance(random_label_summary, pd.DataFrame) else [],
            "ablation_study":
                ablation_summary.to_dict("records")
                if isinstance(ablation_summary, pd.DataFrame) else [],
        }, f, ensure_ascii=False, indent=2)

    print(f"\nResults written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
