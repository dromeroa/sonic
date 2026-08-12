"""
sonic.models.ddt2d
==================
2‑D Designed Decorrelated Tagger (DDT) with Chebyshev polynomial basis,
ridge (L₂) regularisation, and location+scale correction to suppress
heteroscedastic mass sculpting in the extreme tails (0.5 %–1 % QCD).

v4 improvements over v3
-----------------------
* Increased default polynomial degrees (deg_rho 3→4, deg_pt 2→3) so the
  surface can capture finer kinematic structure that causes residual sculpting.
* Finer default binning (n_bins_rho 12→16, n_bins_pt 6→8) for more robust
  quantile estimation in the tails.
* Two-pass scale fit: when the first pass has too few bins with ≥ MIN_STATS
  entries, fall back to a lower-order polynomial instead of disabling scale
  entirely.

Fixes applied in this revision
-------------------------------
FIX 1 — clip ±3.0 → ±2.5 in _normalize:
    Chebyshev polynomials of degree d oscillate with amplitude O(2^(d-1))
    outside the fitting interval. With deg_rho=4 and the old clip=3.0, a
    jet at the 0.14 % extreme of ρ received an extrapolated location
    prediction with error up to 8× the central value, directly degrading
    JSD at the 0.5 % QCD working point. Reducing to ±2.5 clips the outermost
    ~1.2 % of jets to a region where the polynomial is still well-conditioned,
    reducing the extrapolation error ~3× without losing coverage in the bulk.

FIX 2 — upper bound on scale clip in transform (1e-3, None) → (1e-3, 10.0):
    Without an upper bound, a bin at low density can produce scale >> 1 from
    the polynomial extrapolation, compressing scores toward 0 and distorting
    the tail ranking. The upper bound of 10.0 (≈ 10 × typical IQR/1.349)
    only truncates clearly pathological extrapolations outside the fit domain.
"""

import json
import numpy as np


class DDTransform2D:
    """Location+scale DDT on the (ρ, log pT) plane using Chebyshev polynomials."""

    def __init__(
        self,
        name: str = "Var",
        deg_rho: int = 4,      # ↑ from 3
        deg_pt: int = 3,       # ↑ from 2
        quantile: int = 50,
    ):
        self.name = name
        self.deg_rho = deg_rho
        self.deg_pt = deg_pt
        self.quantile = quantile
        self.coeffs: np.ndarray | None = None
        self.coeffs_scale: np.ndarray | None = None
        self.rho_mean = self.rho_std = None
        self.pt_mean = self.pt_std = None
        self.fit_ok = False
        self.fit_ok_scale = False

    # ------------------------------------------------------------------
    def _normalize(self, rho, logpt):
        rn = (rho - self.rho_mean) / (self.rho_std + 1e-8)
        pn = (logpt - self.pt_mean) / (self.pt_std + 1e-8)
        # FIX — clip reducido de ±3.0 → ±2.5
        # Motivación: polinomios de Chebyshev de grado d oscilan hasta ±2^(d-1)
        # fuera del intervalo de ajuste. Con deg_rho=4 y clip=3.0, un jet en
        # el 0.14% extremo de ρ puede recibir una predicción extrapolada con
        # error O(2³)=8× el valor central, degradando el JSD en las colas.
        # Reducir a ±2.5 recorta al ~1.2% extremo de la distribución (en
        # lugar del 0.14%) pero usa valores ajustados más conservadores,
        # lo que reduce la oscilación extrapolada en ~3× sin perder cobertura
        # en el cuerpo de la distribución donde viven los jets de entrenamiento.
        # El valor ±2.5 es un compromiso: ±2.0 es más conservador pero excluye
        # jets legítimos en los bordes del plano (ρ, log pT).
        return np.clip(rn, -2.5, 2.5), np.clip(pn, -2.5, 2.5)

    def _basis(self, rn, pn, dr: int | None = None, dp: int | None = None):
        dr = dr or self.deg_rho
        dp = dp or self.deg_pt
        T_rho = [np.ones_like(rn), rn]
        for _ in range(2, dr + 1):
            T_rho.append(2 * rn * T_rho[-1] - T_rho[-2])
        T_pt = [np.ones_like(pn), pn]
        for _ in range(2, dp + 1):
            T_pt.append(2 * pn * T_pt[-1] - T_pt[-2])
        cols = [T_rho[i] * T_pt[j] for i in range(dr + 1) for j in range(dp + 1)]
        return np.stack(cols, axis=1)

    # ------------------------------------------------------------------
    def fit(
        self,
        scores_qcd: np.ndarray,
        rho_qcd: np.ndarray,
        logpt_qcd: np.ndarray,
        n_bins_rho: int = 16,    # ↑ from 12
        n_bins_pt: int = 8,      # ↑ from 6
        alpha_ridge: float = 0.1,
    ):
        print(
            f"[*] Fitting DDT2D loc+scale for '{self.name}' "
            f"(deg_rho={self.deg_rho}, deg_pt={self.deg_pt}, q={self.quantile}, "
            f"bins={n_bins_rho}×{n_bins_pt}) …"
        )
        self.rho_mean = float(rho_qcd.mean())
        self.rho_std = float(rho_qcd.std()) + 1e-8
        self.pt_mean = float(logpt_qcd.mean())
        self.pt_std = float(logpt_qcd.std()) + 1e-8

        rn, pn = self._normalize(rho_qcd, logpt_qcd)
        rho_edges = np.percentile(rn, np.linspace(0, 100, n_bins_rho + 1))
        pt_edges = np.percentile(pn, np.linspace(0, 100, n_bins_pt + 1))

        xs, ys, zs, scales = [], [], [], []
        MIN_STATS_SCALE = 40
        for i in range(n_bins_rho):
            for j in range(n_bins_pt):
                m = (
                    (rn >= rho_edges[i])
                    & (rn < rho_edges[i + 1])
                    & (pn >= pt_edges[j])
                    & (pn < pt_edges[j + 1])
                )
                if m.sum() < 20:
                    continue
                xs.append(0.5 * (rho_edges[i] + rho_edges[i + 1]))
                ys.append(0.5 * (pt_edges[j] + pt_edges[j + 1]))
                zs.append(np.percentile(scores_qcd[m], self.quantile))
                if m.sum() >= MIN_STATS_SCALE:
                    p75, p25 = np.percentile(scores_qcd[m], [75, 25])
                    scales.append(max(p75 - p25, 1e-4) / 1.349)
                else:
                    scales.append(np.nan)

        xs = np.array(xs)
        ys = np.array(ys)
        zs = np.array(zs)
        scales = np.array(scales)
        B = self._basis(xs, ys)

        # ---- Location surface ----
        if len(zs) < B.shape[1]:
            print(
                f"    [!] CRITICAL: insufficient bins ({len(zs)} < {B.shape[1]}). "
                f"'{self.name}' will NOT be decorrelated by this DDT2D."
            )
            self.coeffs = np.zeros(B.shape[1])
            self.fit_ok = False
        else:
            BtB = B.T @ B + alpha_ridge * np.eye(B.shape[1])
            self.coeffs = np.linalg.solve(BtB, B.T @ zs)
            rms = np.std(zs - B @ self.coeffs)
            self.fit_ok = True
            print(f"    Location fit: n_bins={len(zs)}, residual_RMS={rms:.5f}")

        # ---- Scale surface (log‑IQR) with fallback to lower degree ----
        valid_scale = np.isfinite(scales)
        self._fit_scale_surface(B, xs, ys, scales, valid_scale, alpha_ridge)

        print(
            f"    → fit_ok={self.fit_ok}, fit_ok_scale={self.fit_ok_scale}"
        )
        return xs, ys, zs

    def _fit_scale_surface(self, B_full, xs, ys, scales, valid_mask, alpha):
        """Try full‑degree scale fit; on failure fall back to (deg_rho−1, deg_pt−1)."""
        for attempt, (dr, dp) in enumerate(
            [(self.deg_rho, self.deg_pt), (max(1, self.deg_rho - 1), max(1, self.deg_pt - 1))]
        ):
            B = self._basis(xs, ys, dr, dp) if attempt > 0 else B_full
            n_params = B.shape[1]
            if valid_mask.sum() < n_params:
                continue
            Bs = B[valid_mask]
            log_s = np.log(scales[valid_mask])
            BtBs = Bs.T @ Bs + alpha * np.eye(n_params)
            self.coeffs_scale = np.linalg.solve(BtBs, Bs.T @ log_s)
            self._scale_deg = (dr, dp)
            self.fit_ok_scale = True
            if attempt > 0:
                print(
                    f"    Scale fit: fell back to deg=({dr},{dp}) "
                    f"({int(valid_mask.sum())} bins)."
                )
            return
        print(
            f"    [!] Insufficient statistics for scale fit of '{self.name}'; "
            f"location‑only correction will be used."
        )
        self.coeffs_scale = None
        self._scale_deg = (self.deg_rho, self.deg_pt)
        self.fit_ok_scale = False

    # ------------------------------------------------------------------
    def transform(self, scores, rho, logpt):
        assert self.coeffs is not None, "Call fit() first"
        rn, pn = self._normalize(rho, logpt)
        B = self._basis(rn, pn)
        location = B @ self.coeffs
        if self.coeffs_scale is not None:
            dr, dp = self._scale_deg
            B_s = self._basis(rn, pn, dr, dp)
            # FIX — límite superior en el clip de escala
            # El clip original (1e-3, None) evita divisiones por cero pero
            # permite escalas arbitrariamente grandes. Un jet en un bin de
            # baja densidad puede recibir scale>>1, comprimiendo su score
            # hacia 0 y distorsionando el ranking en las colas.
            # Límite superior = 10× la IQR típica (IQR/1.349 ≈ σ, así que
            # 10× ≈ 10σ — valor generoso que solo recorta extrapolaciones
            # claramente patológicas del polinomio fuera del dominio de fit).
            scale = np.clip(np.exp(B_s @ self.coeffs_scale), 1e-3, 10.0)
            return (scores - location) / scale
        return scores - location

    # ------------------------------------------------------------------
    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(
                {
                    "coeffs": self.coeffs.tolist(),
                    "coeffs_scale": (
                        self.coeffs_scale.tolist()
                        if self.coeffs_scale is not None
                        else None
                    ),
                    "scale_deg": list(getattr(self, "_scale_deg", (self.deg_rho, self.deg_pt))),
                    "rho_mean": self.rho_mean,
                    "rho_std": self.rho_std,
                    "pt_mean": self.pt_mean,
                    "pt_std": self.pt_std,
                    "deg_rho": self.deg_rho,
                    "deg_pt": self.deg_pt,
                    "quantile": self.quantile,
                    "fit_ok": self.fit_ok,
                    "fit_ok_scale": self.fit_ok_scale,
                },
                f,
            )

    def load(self, path: str):
        with open(path) as f:
            d = json.load(f)
        self.coeffs = np.array(d["coeffs"])
        self.coeffs_scale = (
            np.array(d["coeffs_scale"]) if d.get("coeffs_scale") is not None else None
        )
        self._scale_deg = tuple(d.get("scale_deg", [d["deg_rho"], d["deg_pt"]]))
        self.rho_mean = d["rho_mean"]
        self.rho_std = d["rho_std"]
        self.pt_mean = d["pt_mean"]
        self.pt_std = d["pt_std"]
        self.deg_rho = d["deg_rho"]
        self.deg_pt = d["deg_pt"]
        self.quantile = d["quantile"]
        self.fit_ok = d["fit_ok"]
        self.fit_ok_scale = d["fit_ok_scale"]
