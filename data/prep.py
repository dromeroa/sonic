"""
sonic.data.prep
===============
Jet-level normalisation statistics fitted on QCD training files and
persisted to JSON.

Statistics stored
-----------------
feat_mean / feat_std : (2,) arrays — global mean/std of (Δη, Δφ) over
    all active constituents in the QCD training split.  Used to z-score
    feat channels 0 and 1 in JetDatasetV12.

log_pt_abs_mean / log_pt_abs_std : scalars — global mean/std of
    log(pt_constituent + 1e-6) over all active constituents.  Used to
    z-score feat channel 7 (log_pt_abs_norm) in JetDatasetV12.

Fix applied
-----------
The original code computed log_pt_abs_mean/std inside JetDatasetV12.__init__
on a per-file basis, so signal files were normalised with signal statistics
and QCD files with QCD statistics.  This is a data-leakage bug: the two
populations have different pT distributions, so the normalised channel 7
has incomparable scales between train/val/test splits.

Fix: compute log_pt_abs_mean/std once in PrepV12.fit() on the QCD training
split (same as feat_mean/std), persist to JSON, and apply the fixed values
in JetDatasetV12 via prep.log_pt_abs_mean / prep.log_pt_abs_std.
"""

import json
from typing import List

import numpy as np
import pandas as pd

from sonic.utils.compute import fast_parse, jet_axis


class PrepV12:
    """Fit and apply per-jet Δη/Δφ and log_pt_abs normalisation."""

    def __init__(self, max_p: int = 30):
        self.max_p = max_p
        self.feat_mean: np.ndarray | None = None
        self.feat_std:  np.ndarray | None = None
        # NEW: global log_pt_abs statistics fitted on QCD training data
        self.log_pt_abs_mean: float | None = None
        self.log_pt_abs_std:  float | None = None

    # ------------------------------------------------------------------
    def fit(self, files: List[str], max_fit_rows: int = 60_000):
        print(f"[*] Fitting PrepV12 on {len(files)} file(s) …")
        all_deta, all_dphi, all_logpt = [], [], []

	# [FIX-1] Repartir el presupuesto de filas entre TODOS los archivos
        # (antes: files[:2] hardcoded → scaler sesgado si hay >2 archivos QCD).
        # Leer 3× el cupo para tener margen tras el corte cinemático pT>250/SDMass,
        # luego shuffle para romper cualquier ordenamiento del CSV (por pT, evento,
        # corrida) antes de limitar a rows_per_file.
        rows_per_file = max(1_000, max_fit_rows // max(1, len(files)))
        for f in files:
            df = pd.read_csv(
                f, usecols=["PF_Pt", "PF_Eta", "PF_Phi", "SDMass"],
                nrows=rows_per_file * 3,  # margen para pérdida por corte cinemático
            )
            df = df.sample(frac=1.0, random_state=42).head(rows_per_file).reset_index(drop=True)
	
            pt  = fast_parse(df["PF_Pt"],  self.max_p)
            eta = fast_parse(df["PF_Eta"], self.max_p)
            phi = fast_parse(df["PF_Phi"], self.max_p)

            px, py = pt * np.cos(phi), pt * np.sin(phi)
            jpt = np.sqrt(px.sum(1) ** 2 + py.sum(1) ** 2)
            sdm = df["SDMass"].values.astype(np.float32)
            valid = (jpt > 250.0) & (sdm >= 40.0) & (sdm <= 200.0)
            pt, eta, phi = pt[valid], eta[valid], phi[valid]

            mask = pt > 0
            je, jp = jet_axis(pt, eta, phi)

            # Δη, Δφ statistics (unchanged)
            all_deta.append((eta - je)[mask])
            all_dphi.append(((phi - jp + np.pi) % (2 * np.pi) - np.pi)[mask])

            # log_pt_abs statistics — computed on active constituents only
            log_pt_abs = np.log(pt + 1e-6)          # (N, max_p), zero for padding
            all_logpt.append(log_pt_abs[mask])       # flatten active constituents

        deta  = np.concatenate(all_deta)
        dphi  = np.concatenate(all_dphi)
        logpt = np.concatenate(all_logpt)

        self.feat_mean = np.array([deta.mean(), dphi.mean()], dtype=np.float32)
        self.feat_std  = np.array([deta.std(),  dphi.std()],  dtype=np.float32)

        # FIX: store as Python floats for JSON serialisation
        self.log_pt_abs_mean = float(logpt.mean())
        self.log_pt_abs_std  = float(logpt.std()) + 1e-8

        print(
            f"    deta_mean={self.feat_mean[0]:.5f}  dphi_mean={self.feat_mean[1]:.5f}\n"
            f"    deta_std ={self.feat_std[0]:.5f}   dphi_std ={self.feat_std[1]:.5f}\n"
            f"    log_pt_abs_mean={self.log_pt_abs_mean:.5f}  "
            f"log_pt_abs_std={self.log_pt_abs_std:.5f}"
        )

    # ------------------------------------------------------------------
    def save(self, path: str = "scaler_v12.json"):
        with open(path, "w") as f:
            json.dump(
                {
                    "feat_mean":        self.feat_mean.tolist(),
                    "feat_std":         self.feat_std.tolist(),
                    "log_pt_abs_mean":  self.log_pt_abs_mean,   # NEW
                    "log_pt_abs_std":   self.log_pt_abs_std,    # NEW
                },
                f,
                indent=2,
            )

    def load(self, path: str = "scaler_v12.json"):
        with open(path) as f:
            d = json.load(f)
        self.feat_mean = np.array(d["feat_mean"], dtype=np.float32)
        self.feat_std  = np.array(d["feat_std"],  dtype=np.float32)

        # Backwards-compatible load: old JSON files without log_pt_abs stats
        # will trigger a re-fit warning instead of a silent KeyError
        if "log_pt_abs_mean" in d:
            self.log_pt_abs_mean = float(d["log_pt_abs_mean"])
            self.log_pt_abs_std  = float(d["log_pt_abs_std"])
        else:
            print(
                "[!] PrepV12.load: 'log_pt_abs_mean' not found in scaler JSON. "
                "Re-run PrepV12.fit() and save() to update the scaler file. "
                "Defaulting to None — JetDatasetV12 will raise if feat channel 7 "
                "is requested without valid log_pt_abs statistics."
            )
            self.log_pt_abs_mean = None
            self.log_pt_abs_std  = None

    # ------------------------------------------------------------------
    def check_log_pt_abs(self):
        """Raise a clear error if log_pt_abs stats were not fitted/loaded."""
        if self.log_pt_abs_mean is None or self.log_pt_abs_std is None:
            raise RuntimeError(
                "PrepV12: log_pt_abs_mean/std are None. "
                "Call fit() on QCD training files and save(), "
                "or load() from an updated scaler JSON that contains these keys."
            )
