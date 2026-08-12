"""
sonic.utils.metrics
===================
Statistical metrics and bootstrap uncertainty quantification.
"""

import multiprocessing as mp

import numpy as np
import torch
from sklearn.metrics import roc_curve


# ---------------------------------------------------------------------------
# Jensen–Shannon divergence (histogram‑based)
# ---------------------------------------------------------------------------
def compute_jsd(h_ref: np.ndarray, h_cut: np.ndarray, eps: float = 1e-10) -> float:
    """
    JSD between two *density* histograms (same binning).

    Fix applied — double epsilon bias
    -----------------------------------
    The original code computed:
        kl = sum(p * log(p / (m + eps) + eps))

    This adds eps TWICE: once in the denominator and once inside the log.
    The standard KL form is p * log(p / m). The additive eps inside log
    introduces a systematic positive bias of ~eps/m per bin — O(1e-8) total
    with 40 bins and eps=1e-10. Negligible in practice but mathematically
    incorrect and inconsistent with the scipy reference implementation.

    Fix: compute the ratio p/m, clip it to [1e-300, inf] to avoid log(0)
    on genuinely empty bins (correct: empty bins contribute 0 to KL), then
    take log of the clipped ratio only — no additive eps inside the log.
    """
    p = np.maximum(h_ref, eps)
    p = p / p.sum()
    q = np.maximum(h_cut, eps)
    q = q / q.sum()
    m = 0.5 * (p + q)
    # FIX: single clip on the ratio — no additive eps inside the log
    kl1 = np.sum(p * np.log(np.clip(p / m, 1e-300, None)))
    kl2 = np.sum(q * np.log(np.clip(q / m, 1e-300, None)))
    return float(0.5 * kl1 + 0.5 * kl2)


# ---------------------------------------------------------------------------
# Distance correlation (PyTorch, differentiable)
# ---------------------------------------------------------------------------
def distance_correlation(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Empirical distance correlation between 1-D tensors *x* and *y*.
    Follows Székely et al. (2007), dCor² = dCov²(X,Y) / sqrt(dVar²(X)·dVar²(Y)).

    Numerical fixes applied
    -----------------------
    FIX 1 — dcov2 clamp:
        The doubly-centred matrix product (A*B).sum() can be slightly negative
        due to floating-point cancellation in the successive mean subtractions.
        A negative dcov2 fed into sqrt() produces NaN that propagates silently
        through loss_disco into the model weights.
        Solution: clamp(min=0.0) before the division.

    FIX 2 — denominator clamp:
        dvarx2 * dvary2 can be zero or near-zero when the batch is homogeneous
        (all x or all y identical, common early in training).
        torch.sqrt(0) = 0 is fine, but dvarx2*dvary2 can be slightly negative
        for the same floating-point reason as dcov2.
        Solution: clamp(min=0.0) before the outer sqrt, so we get 0 instead of NaN.

    FIX 3 — final NaN guard:
        As a last resort, torch.nan_to_num converts any residual NaN/Inf to 0.0
        so that a single pathological batch cannot corrupt the training run.

    These fixes are conservative: they do not change the value of dCor when the
    computation is numerically well-conditioned; they only prevent silent NaN
    propagation in edge cases.
    """
    n = x.size(0)
    if n <= 1:
        return torch.tensor(0.0, device=x.device)

    # --- Pairwise L1 distance matrices ---
    a = torch.abs(x.unsqueeze(0) - x.unsqueeze(1))   # (n, n)
    b = torch.abs(y.unsqueeze(0) - y.unsqueeze(1))   # (n, n)

    # --- Double centering (Székely et al. eq. 2.4) ---
    A = a - a.mean(1, keepdim=True) - a.mean(0, keepdim=True) + a.mean()
    B = b - b.mean(1, keepdim=True) - b.mean(0, keepdim=True) + b.mean()

    n2 = float(n * n)

    # FIX 1: clamp dcov2 ≥ 0 before sqrt
    dcov2  = (A * B).sum() / (n2 + eps)
    dcov2  = dcov2.clamp(min=0.0)

    dvarx2 = (A * A).sum() / (n2 + eps)
    dvary2 = (B * B).sum() / (n2 + eps)

    # FIX 2: clamp product of variances ≥ 0 before outer sqrt
    dvar_prod = (dvarx2 * dvary2).clamp(min=0.0)

    dcor = torch.sqrt(dcov2 / (torch.sqrt(dvar_prod) + eps))

    # FIX 3: final NaN/Inf guard
    return torch.nan_to_num(dcor, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# Bootstrap ROC / SIC bands
# ---------------------------------------------------------------------------
def _run_one_boot(args):
    b, idx_sig, idx_qcd, n_sig, n_qcd, scores, labels, fpr_grid, seed = args
    rng = np.random.default_rng(seed + b)
    bs_sig = rng.choice(idx_sig, size=n_sig, replace=True)
    bs_qcd = rng.choice(idx_qcd, size=n_qcd, replace=True)
    idx = np.concatenate([bs_sig, bs_qcd])
    s_b, l_b = scores[idx], labels[idx]

    fpr_b, tpr_b, _ = roc_curve(l_b, s_b)
    tpr_interp = np.interp(fpr_grid, fpr_b, tpr_b)
    with np.errstate(divide="ignore", invalid="ignore"):
        sic_interp = tpr_interp / np.sqrt(fpr_grid)
    return tpr_interp, sic_interp


def bootstrap_roc_sic(scores, labels, n_boot: int = 300, fpr_grid=None, seed: int = 42):
    if fpr_grid is None:
        fpr_grid = np.linspace(0.001, 1.0, 100)

    scores, labels = np.asarray(scores), np.asarray(labels)
    idx_sig = np.where(labels == 1)[0]
    idx_qcd = np.where(labels == 0)[0]

    tasks = [
        (b, idx_sig, idx_qcd, len(idx_sig), len(idx_qcd), scores, labels, fpr_grid, seed)
        for b in range(n_boot)
    ]
    n_cores = min(8, mp.cpu_count())
    with mp.Pool(processes=n_cores) as pool:
        results = pool.map(_run_one_boot, tasks)

    tpr_boot = np.array([r[0] for r in results])
    sic_boot = np.array([r[1] for r in results])

    def _bands(mat):
        return {
            "p16": np.percentile(mat, 16, axis=0),
            "p50": np.percentile(mat, 50, axis=0),
            "p84": np.percentile(mat, 84, axis=0),
        }

    return {"fpr_grid": fpr_grid, "tpr": _bands(tpr_boot), "sic": _bands(sic_boot)}


# ---------------------------------------------------------------------------
# Bootstrap JSD bands (mass‑sculpting uncertainty)
# ---------------------------------------------------------------------------
def _run_one_boot_jsd(args_tuple):
    b, scores_qcd, masses_qcd, fpr_cuts, bins_m, h_ref_fixed, seed = args_tuple
    # FIX: usar h_ref_fixed (histograma de referencia calculado sobre el
    # dataset QCD completo, NO sobre el resample). El error original calculaba
    # h_ref_b sobre masses_qcd[idx], de modo que AMBAS distribuciones variaban
    # en cada bootstrap (la referencia y el corte). Eso infla artificialmente
    # la varianza del CI porque cancela parcialmente el efecto del corte.
    # La definicion correcta de JSD bootstrap es: fijar la referencia (la
    # distribucion de masa QCD sin corte) y solo resamplear el numerador
    # (la distribucion de masa post-corte). Asi el CI captura la incertidumbre
    # estadistica del corte, no la de la referencia.
    rng = np.random.default_rng(seed + 10_000 + b)
    n = len(scores_qcd)
    idx = rng.choice(n, size=n, replace=True)
    s_b, m_b = scores_qcd[idx], masses_qcd[idx]

    out = {}
    for fpr_c in fpr_cuts:
        if fpr_c >= 1.0:
            continue
        thr = np.percentile(s_b, (1.0 - fpr_c) * 100)
        m_pass = m_b[s_b >= thr]
        if len(m_pass) < 5:
            out[fpr_c] = np.nan
            continue
        h_cut_b, _ = np.histogram(m_pass, bins=bins_m, density=True)
        out[fpr_c] = compute_jsd(h_ref_fixed, h_cut_b)
    return out


def bootstrap_jsd(scores_qcd, masses_qcd, fpr_cuts, bins_m,
                  n_boot: int = 300, seed: int = 42):
    scores_qcd = np.asarray(scores_qcd)
    masses_qcd = np.asarray(masses_qcd)
    fpr_cuts_no1 = [f for f in fpr_cuts if f < 1.0]

    # FIX: calcular h_ref UNA SOLA VEZ sobre el dataset completo y pasarlo
    # a cada worker. Antes cada worker recalculaba h_ref sobre su resample,
    # lo que inflaba el CI. Ver comentario en _run_one_boot_jsd.
    h_ref_fixed, _ = np.histogram(masses_qcd, bins=bins_m, density=True)

    tasks = [(b, scores_qcd, masses_qcd, fpr_cuts, bins_m, h_ref_fixed, seed) for b in range(n_boot)]
    n_cores = min(8, mp.cpu_count())
    with mp.Pool(processes=n_cores) as pool:
        results = pool.map(_run_one_boot_jsd, tasks)

    bands = {}
    for fpr_c in fpr_cuts_no1:
        vals = np.array(
            [r[fpr_c] for r in results if not np.isnan(r.get(fpr_c, np.nan))]
        )
        if len(vals) == 0:
            bands[fpr_c] = (float("nan"), float("nan"), float("nan"))
        else:
            bands[fpr_c] = (
                float(np.percentile(vals, 16)),
                float(np.percentile(vals, 50)),
                float(np.percentile(vals, 84)),
            )
    return bands
