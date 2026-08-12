"""
chamfer_ablation.py
===================
Ablation study: three Chamfer Distance variants compared on the same
trained model and the same test split.  Designed for publication in
JHEP / EPJC.

The three variants are evaluated *post hoc* on a frozen model
(no retraining) so that every difference is attributable to the
Chamfer score definition alone and not to confounding training effects.
This is the exact study design reviewers at JHEP/EPJC will expect.

Variants
--------
V0 — Baseline (angular only)
    d_ij = (Δη_ij)² + (Δφ_ij)²
    Replicates losses.py as shipped.

V1 — Angular + energy channels
    d_ij = (Δη_ij)² + (Δφ_ij)²
          + λ_E · [(Δ log_pt_rel_ij)² + (Δ pt_frac_ij)²]

V2 — Angular + radial (delta_R)
    d_ij = (Δη_ij)² + (Δφ_ij)²
          + λ_R · (Δ delta_R_ij)²

For each variant the script:
  1. Recomputes the per-jet Chamfer score from the frozen model output.
  2. Fits a fresh DDT2D (Chebyshev, loc+scale) on the DDT pool.
  3. Runs a fine grid-search for the optimal α·C_DDT − γ·τ₂₁_DDT combo.
  4. Evaluates AUC, SIC_max, Rej@TPR=50%, and JSD at multiple QCD cuts
     on the held-out test set with 300-bootstrap uncertainty bands.
  5. Saves all metrics to a JSON file and produces a publication-quality
     comparison figure (ROC overlay + SIC overlay + JSD vs FPR + mass
     sculpting panel).

Usage
-----
    python chamfer_ablation.py \\
        --model  best_ae_v12_sonic_seed42.pt \\
        --signal sonic/signalM1000_last.csv \\
        --qcd    sonic/qcd_background.csv   \\
        --scaler scaler_v12_sonic.json      \\
        --lambda-energy 0.20                \\
        --lambda-rad    0.35                \\
        --seed   42  --n-boot 300

Output files
------------
    chamfer_ablation_metrics.json   — all numeric results
    chamfer_ablation_figure.pdf     — 4-panel comparison figure
    chamfer_ablation_jsd_table.txt  — LaTeX-ready JSD table
"""

import argparse
import json
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, roc_curve

# ── SONIC imports (project root must be on PYTHONPATH) ──────────────────────
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset

from sonic.utils.config import DEFAULTS, fijar_semilla, configure_system_resources
from sonic.utils.metrics import compute_jsd, bootstrap_roc_sic, bootstrap_jsd
from sonic.data.prep import PrepV12
from sonic.data.datasets import JetDatasetV12
from sonic.models.jet_autoencoder import JetAE_V12
from sonic.models.ddt2d import DDTransform2D
from sonic.inference.eval import fix_dir_on_validation, grid_search_hybrid

warnings.filterwarnings("ignore")


# ============================================================
# 1.  Chamfer variants
# ============================================================

def chamfer_v0(feat, recon, mask):
    """Baseline — angular channels only (Δη, Δφ). Reproduces losses.py."""
    d = torch.cdist(feat[:, :, :2], recon[:, :, :2], p=2) ** 2
    d = d + (1 - mask.unsqueeze(2)) * 1e8 + (1 - mask.unsqueeze(1)) * 1e8
    pj = (d.min(2)[0] * mask).sum(1) + (d.min(1)[0] * mask).sum(1)
    return pj / (mask.sum(1) + 1e-8)


def chamfer_v1(feat, recon, mask, lam_e: float = 0.20):
    """V1 — angular + energy (log_pt_rel [:, 2] and pt_frac [:, 3])."""
    d_geo = torch.cdist(feat[:, :, :2],  recon[:, :, :2],  p=2) ** 2
    d_eng = torch.cdist(feat[:, :, 2:4], recon[:, :, 2:4], p=2) ** 2
    d = d_geo + lam_e * d_eng
    d = d + (1 - mask.unsqueeze(2)) * 1e8 + (1 - mask.unsqueeze(1)) * 1e8
    pj = (d.min(2)[0] * mask).sum(1) + (d.min(1)[0] * mask).sum(1)
    return pj / (mask.sum(1) + 1e-8)


def chamfer_v2(feat, recon, mask, lam_r: float = 0.35):
    """V2 — angular + radial delta_R (feat channel 6)."""
    d_geo = torch.cdist(feat[:, :, :2],  recon[:, :, :2],  p=2) ** 2
    d_rad = torch.cdist(feat[:, :, 6:7], recon[:, :, 6:7], p=2) ** 2
    d = d_geo + lam_r * d_rad
    d = d + (1 - mask.unsqueeze(2)) * 1e8 + (1 - mask.unsqueeze(1)) * 1e8
    pj = (d.min(2)[0] * mask).sum(1) + (d.min(1)[0] * mask).sum(1)
    return pj / (mask.sum(1) + 1e-8)


VARIANT_FNS = {
    "V0_angular":         lambda f, r, m, args: chamfer_v0(f, r, m),
    "V1_angular+energy":  lambda f, r, m, args: chamfer_v1(f, r, m, args.lambda_energy),
    "V2_angular+deltaR":  lambda f, r, m, args: chamfer_v2(f, r, m, args.lambda_rad),
}

# Human-readable labels for plots and tables
VARIANT_LABELS = {
    "V0_angular":        r"V0: $(\Delta\eta, \Delta\varphi)$ only",
    "V1_angular+energy": r"V1: + $\log p_{T}^{\rm rel}$, $p_{T}^{\rm frac}$",
    "V2_angular+deltaR": r"V2: + $\delta R$",
}

VARIANT_COLORS = {
    "V0_angular":        "steelblue",
    "V1_angular+energy": "darkorange",
    "V2_angular+deltaR": "forestgreen",
}

VARIANT_LS = {
    "V0_angular":        "-",
    "V1_angular+energy": "--",
    "V2_angular+deltaR": "-.",
}


# ============================================================
# 2.  Score collection with pluggable Chamfer function
# ============================================================

@torch.no_grad()
def collect_scores(model, loader, device, chamfer_fn, args):
    """
    Runs the frozen model and computes the Chamfer score using
    the supplied variant function.  Returns arrays:
        C, tau21, rho, label, SDMass, log_pT, N2
    """
    model.eval()
    C, T21, R, L, M, LP, N2 = [], [], [], [], [], [], []
    for feat, mask, phys in loader:
        feat, mask = feat.to(device), mask.to(device)
        recon, _ = model(feat)
        c = chamfer_fn(feat, recon, mask, args).cpu().numpy()
        C.extend(c)
        T21.extend(phys[:, 3].numpy())
        R.extend(phys[:, 6].numpy())
        L.extend(phys[:, 5].numpy())
        M.extend(phys[:, 4].numpy())
        LP.extend(phys[:, 7].numpy())
        N2.extend(phys[:, 8].numpy())
    return (np.array(C), np.array(T21), np.array(R),
            np.array(L), np.array(M), np.array(LP), np.array(N2))


# ============================================================
# 3.  Per-variant evaluation pipeline
# ============================================================

def evaluate_variant(
    name, chamfer_fn, model, ddt_loader, gs_loader, te_loader,
    device, args, seed
):
    """
    Full evaluation for one Chamfer variant on the frozen model:
      DDT fit → grid search → test metrics → bootstrap CI.
    Returns a dict with all numeric results.
    """
    D = DEFAULTS
    print(f"\n{'─' * 60}")
    print(f"  Variant: {name}")
    print(f"{'─' * 60}")

    # ── Collect raw scores ───────────────────────────────────────
    C_df, T21_df, R_df, L_df, M_df, LP_df, N2_df = collect_scores(
        model, ddt_loader, device, chamfer_fn, args)
    C_gs, T21_gs, R_gs, L_gs, M_gs, LP_gs, N2_gs = collect_scores(
        model, gs_loader,  device, chamfer_fn, args)
    C_te, T21_te, R_te, L_te, M_te, LP_te, N2_te = collect_scores(
        model, te_loader,  device, chamfer_fn, args)

    qcd_df = L_df == 0
    qcd_gs = L_gs == 0
    qcd_te = L_te == 0

    C_log_df = np.log(C_df + 1e-7)
    C_log_gs = np.log(C_gs + 1e-7)
    C_log_te = np.log(C_te + 1e-7)

    # ── DDT2D fit ────────────────────────────────────────────────
    ddt_C = DDTransform2D(
        name=f"log(C) {name}", deg_rho=4, deg_pt=3, quantile=D["DDT_QUANT"]
    )
    ddt_C.fit(C_log_df[qcd_df], R_df[qcd_df], LP_df[qcd_df])

    ddt_T21 = DDTransform2D(
        name="Tau21", deg_rho=4, deg_pt=3, quantile=D["DDT_QUANT"]
    )
    ddt_T21.fit(T21_df[qcd_df], R_df[qcd_df], LP_df[qcd_df])

    # ── DDT transform ────────────────────────────────────────────
    C_gs_ddt  = ddt_C.transform(C_log_gs,  R_gs, LP_gs)
    T21_gs_ddt = ddt_T21.transform(T21_gs, R_gs, LP_gs)
    C_te_ddt  = ddt_C.transform(C_log_te,  R_te, LP_te)
    T21_te_ddt = ddt_T21.transform(T21_te, R_te, LP_te)

    # ── Grid search (validation set, disjoint from DDT pool) ────
    best_params, _ = grid_search_hybrid(C_gs_ddt, T21_gs_ddt, L_gs)
    alpha, gamma = best_params["alpha"], best_params["gamma"]

    s_gs_combo = alpha * C_gs_ddt - gamma * T21_gs_ddt
    sign, _ = fix_dir_on_validation(s_gs_combo, L_gs)

    # ── Test scores ──────────────────────────────────────────────
    s_te_raw = sign * (alpha * C_log_te - gamma * T21_te)
    s_te_ddt = sign * (alpha * C_te_ddt - gamma * T21_te_ddt)

    # AUC
    auc_combined = roc_auc_score(L_te, s_te_ddt)
    sign_ae, _ = fix_dir_on_validation(C_gs_ddt, L_gs)
    auc_ae = roc_auc_score(L_te, sign_ae * C_te_ddt)

    # Pearson residual correlations
    rho_corr, _ = pearsonr(s_te_ddt[qcd_te], R_te[qcd_te])
    pt_corr,  _ = pearsonr(s_te_ddt[qcd_te], LP_te[qcd_te])

    # ROC / SIC
    fpr, tpr, _ = roc_curve(L_te, s_te_ddt)
    with np.errstate(divide="ignore", invalid="ignore"):
        sic = np.where(fpr > 0, tpr / np.sqrt(fpr), np.nan)

    idx50  = np.argmin(np.abs(tpr - 0.50))
    rej50  = 1.0 / fpr[idx50] if fpr[idx50] > 0 else np.nan
    sic_max = float(np.nanmax(sic))

    # JSD at multiple FPR working points
    bins_m   = np.linspace(40, 200, 41)
    fpr_cuts = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005]
    qcd_scores = s_te_ddt[qcd_te]
    qcd_masses = M_te[qcd_te]
    h_ref, _ = np.histogram(qcd_masses, bins=bins_m, density=True)
    jsd_vals = {}
    for fpr_c in fpr_cuts:
        if fpr_c >= 1.0:
            continue
        thr = np.percentile(qcd_scores, (1.0 - fpr_c) * 100)
        m_pass = qcd_masses[qcd_scores >= thr]
        h_cut, _ = np.histogram(m_pass, bins=bins_m, density=True)
        jsd_vals[fpr_c] = float(compute_jsd(h_ref, h_cut)) if len(m_pass) >= 5 else np.nan

    # Bootstrap
    print(f"  Bootstrap (n={args.n_boot}) …")
    boot_roc = bootstrap_roc_sic(s_te_ddt, L_te, n_boot=args.n_boot, seed=seed)
    jsd_boot = bootstrap_jsd(qcd_scores, qcd_masses, fpr_cuts, bins_m,
                              n_boot=args.n_boot, seed=seed)

    # Console summary
    print(f"  AUC (combined DDT) = {auc_combined:.4f}")
    print(f"  AUC (AE only)      = {auc_ae:.4f}")
    print(f"  Rej@TPR=50%        = {rej50:.1f}")
    print(f"  SIC_max            = {sic_max:.3f}")
    print(f"  Pearson(s, ρ)      = {rho_corr:+.4f}")
    print(f"  Pearson(s, logpT)  = {pt_corr:+.4f}")
    print(f"  {'FPR':>6}  JSD      68% CI")
    for fpr_c in [0.5, 0.1, 0.05, 0.01, 0.005]:
        jsd = jsd_vals.get(fpr_c, np.nan)
        ci  = jsd_boot.get(fpr_c, (np.nan, np.nan, np.nan))
        ok  = "✓" if (np.isfinite(jsd) and jsd < 0.04) else "[!]"
        print(f"  {fpr_c * 100:>5.1f}%  {jsd:.4f}  "
              f"[{ci[0]:.4f}, {ci[2]:.4f}]  {ok}")

    return dict(
        name=name,
        label=VARIANT_LABELS[name],
        # ROC / SIC arrays (for plotting)
        fpr=fpr.tolist(), tpr=tpr.tolist(), sic=sic.tolist(),
        # Bootstrap bands
        boot_fpr_grid=boot_roc["fpr_grid"].tolist(),
        boot_tpr_p16=boot_roc["tpr"]["p16"].tolist(),
        boot_tpr_p50=boot_roc["tpr"]["p50"].tolist(),
        boot_tpr_p84=boot_roc["tpr"]["p84"].tolist(),
        boot_sic_p16=boot_roc["sic"]["p16"].tolist(),
        boot_sic_p50=boot_roc["sic"]["p50"].tolist(),
        boot_sic_p84=boot_roc["sic"]["p84"].tolist(),
        # JSD bootstrap bands
        jsd_boot={str(k): list(v) for k, v in jsd_boot.items()},
        # Scalar metrics
        auc_combined=float(auc_combined),
        auc_ae_only=float(auc_ae),
        rej_at_50tpr=float(rej50),
        sic_max=float(sic_max),
        rho_pearson=float(rho_corr),
        logpt_pearson=float(pt_corr),
        jsd_by_fpr={str(k): float(v) for k, v in jsd_vals.items()},
        alpha=float(alpha),
        gamma=float(gamma),
        # Store arrays needed for mass-sculpting plot
        qcd_scores=qcd_scores.tolist(),
        qcd_masses=qcd_masses.tolist(),
    )


# ============================================================
# 4.  Comparison figure (publication quality)
# ============================================================

def plot_comparison(results: list, save_path: str, n_boot: int):
    """
    4-panel figure:
      [0] ROC overlay with 68% CI bands
      [1] SIC overlay with 68% CI bands
      [2] JSD vs FPR (log-log) with bootstrap error bars
      [3] Mass sculpting at 1% QCD for all variants
    """
    FPR_CUTS_PLOT = [0.5, 0.1, 0.05, 0.01, 0.005]
    bins_m = np.linspace(40, 200, 41)
    bin_cen = 0.5 * (bins_m[:-1] + bins_m[1:])

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(
        "Chamfer Distance ablation study — same frozen model, three score definitions\n"
        f"Bootstrap n={n_boot}, shaded bands = 68% CI",
        fontsize=11, fontweight="bold", y=1.01
    )

    # ── Panel 0: ROC ─────────────────────────────────────────────
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4, label="random")
    for r in results:
        name  = r["name"]
        col   = VARIANT_COLORS[name]
        ls    = VARIANT_LS[name]
        label = f"{r['label']}  AUC={r['auc_combined']:.4f}"
        ax.plot(r["fpr"], r["tpr"], color=col, lw=2, ls=ls, label=label)
        fg = np.array(r["boot_fpr_grid"])
        ax.fill_between(fg,
                        np.array(r["boot_tpr_p16"]),
                        np.array(r["boot_tpr_p84"]),
                        color=col, alpha=0.12)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC curve")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(True, alpha=0.3)

    # ── Panel 1: SIC ─────────────────────────────────────────────
    ax = axes[1]
    for r in results:
        name  = r["name"]
        col   = VARIANT_COLORS[name]
        ls    = VARIANT_LS[name]
        fg    = np.array(r["boot_fpr_grid"])
        label = f"{r['label']}  SIC_max={r['sic_max']:.3f}"
        ax.plot(fg,
                np.array(r["boot_sic_p50"]),
                color=col, lw=2, ls=ls, label=label)
        ax.fill_between(fg,
                        np.array(r["boot_sic_p16"]),
                        np.array(r["boot_sic_p84"]),
                        color=col, alpha=0.12)
    ax.set_xlim(0, 0.4)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("FPR")
    ax.set_ylabel(r"SIC  $=\varepsilon_S / \sqrt{\varepsilon_B}$")
    ax.set_title("Significance improvement (SIC)")
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: JSD vs FPR (log-log) ───────────────────────────
    ax = axes[2]
    ax.axhline(0.04, color="gray", ls=":", lw=1.2, label="JSD target 0.04")
    x_jsd = np.array(FPR_CUTS_PLOT)
    for r in results:
        name = r["name"]
        col  = VARIANT_COLORS[name]
        ls   = VARIANT_LS[name]
        y_mid, y_lo, y_hi = [], [], []
        for fpr_c in FPR_CUTS_PLOT:
            jsd_c = r["jsd_by_fpr"].get(str(fpr_c), np.nan)
            ci    = r["jsd_boot"].get(str(fpr_c), [np.nan, np.nan, np.nan])
            y_mid.append(jsd_c)
            y_lo.append(ci[0] if len(ci) > 0 else np.nan)
            y_hi.append(ci[2] if len(ci) > 2 else np.nan)
        y_mid = np.array(y_mid, dtype=float)
        y_lo  = np.array(y_lo,  dtype=float)
        y_hi  = np.array(y_hi,  dtype=float)
        ax.plot(x_jsd, y_mid, color=col, lw=2, ls=ls,
                marker="o", ms=5, label=r["label"])
        ax.fill_between(x_jsd, y_lo, y_hi, color=col, alpha=0.15)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("QCD pass fraction (FPR)")
    ax.set_ylabel("JSD(SDMass)")
    ax.set_title("Mass sculpting (lower = better)")
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.3, which="both")

    # ── Panel 3: SDMass distribution at 1% QCD ──────────────────
    ax = axes[3]
    fpr_show = 0.01          # 1% QCD working point
    for r in results:
        name     = r["name"]
        col      = VARIANT_COLORS[name]
        ls       = VARIANT_LS[name]
        qs       = np.array(r["qcd_scores"])
        qm       = np.array(r["qcd_masses"])
        h_ref, _ = np.histogram(qm, bins=bins_m, density=True)
        thr      = np.percentile(qs, (1.0 - fpr_show) * 100)
        m_pass   = qm[qs >= thr]
        h_cut, _ = np.histogram(m_pass, bins=bins_m, density=True)
        jsd_c    = r["jsd_by_fpr"].get(str(fpr_show), np.nan)
        label    = f"{r['label']}  JSD={jsd_c:.4f}"
        ax.step(bin_cen, h_cut, color=col, lw=2, ls=ls,
                where="mid", label=label)
    # inclusive QCD reference from first result
    qs_ref   = np.array(results[0]["qcd_scores"])
    qm_ref   = np.array(results[0]["qcd_masses"])
    h_ref, _ = np.histogram(qm_ref, bins=bins_m, density=True)
    ax.step(bin_cen, h_ref, color="black", lw=1.2, ls=":",
            where="mid", label="Inclusive QCD (ref.)", alpha=0.7)
    ax.axvspan(65, 95, color="gray", alpha=0.12, label="W-mass window")
    ax.set_xlabel("SDMass [GeV]")
    ax.set_ylabel("Density (normalised)")
    ax.set_title(f"Mass sculpting at {fpr_show * 100:.0f}% QCD")
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n[*] Figure saved → {save_path}")


# ============================================================
# 5.  LaTeX table for the paper
# ============================================================

def write_latex_table(results: list, path: str):
    """
    Produces a LaTeX longtable with:
      Variant | AUC | Rej@50% | SIC_max | JSD(1%) | JSD(0.5%) | ρ_Pearson
    Ready to paste into the Results section.
    """
    FPR_01  = "0.01"
    FPR_005 = "0.005"
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Ablation study: three Chamfer Distance definitions evaluated "
        r"on the same frozen SONIC AE v12 model. "
        r"All metrics computed on the held-out test set. "
        r"Uncertainties from 300 bootstrap replicas (68\% CI). "
        r"JSD target $< 0.04$ (dashed line in Fig.~\ref{fig:ablation}).}"
    )
    lines.append(r"\label{tab:chamfer_ablation}")
    lines.append(
        r"\begin{tabular}{lcccccc}"
    )
    lines.append(r"\hline\hline")
    lines.append(
        r"Variant & AUC & Rej@TPR\,50\% & SIC$_{\max}$ & "
        r"JSD (1\,\% QCD) & JSD (0.5\,\% QCD) & $\rho$(score,\,$\rho$) \\"
    )
    lines.append(r"\hline")

    for r in results:
        jsd01  = r["jsd_by_fpr"].get(FPR_01,  np.nan)
        jsd005 = r["jsd_by_fpr"].get(FPR_005, np.nan)
        ci01   = r["jsd_boot"].get(FPR_01,  [np.nan, np.nan, np.nan])
        ci005  = r["jsd_boot"].get(FPR_005, [np.nan, np.nan, np.nan])

        def fmt_jsd(val, ci):
            if not np.isfinite(val):
                return r"\textemdash"
            lo = ci[0] if len(ci) > 0 and np.isfinite(ci[0]) else val
            hi = ci[2] if len(ci) > 2 and np.isfinite(ci[2]) else val
            flag = r"$^*$" if val > 0.04 else ""
            return rf"{val:.4f}$^{{+{hi - val:.4f}}}_{{-{val - lo:.4f}}}${flag}"

        rej_str = f"{r['rej_at_50tpr']:.0f}" if np.isfinite(r["rej_at_50tpr"]) else r"\textemdash"
        row = (
            rf"{r['label']} & "
            rf"{r['auc_combined']:.4f} & "
            rf"{rej_str} & "
            rf"{r['sic_max']:.3f} & "
            rf"{fmt_jsd(jsd01, ci01)} & "
            rf"{fmt_jsd(jsd005, ci005)} & "
            rf"{r['rho_pearson']:+.4f} \\"
        )
        lines.append(row)

    lines.append(r"\hline\hline")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\begin{tablenotes}"
        r"\item[$^*$] Exceeds JSD target of 0.04 — sculpting not fully suppressed."
        r"\end{tablenotes}"
    )
    lines.append(r"\end{table}")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[*] LaTeX table saved → {path}")


# ============================================================
# 6.  CLI entry point
# ============================================================

def build_parser():
    D = DEFAULTS
    p = argparse.ArgumentParser(
        description="SONIC Chamfer ablation study (V0 vs V1 vs V2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",          required=True,
                   help="Path to frozen model checkpoint (.pt)")
    p.add_argument("--signal",         required=True,
                   help="Signal CSV file (signalM1000_last.csv)")
    p.add_argument("--qcd",            required=True,
                   help="QCD background CSV file")
    p.add_argument("--scaler",         default=D["SCALER_PATH"],
                   help="PrepV12 scaler JSON")
    p.add_argument("--max-rows",       type=int,  default=30000,
                   help="Max rows per CSV (-1 = all)")
    p.add_argument("--lambda-energy",  type=float, default=0.20,
                   help="λ_E weight for V1 energy channels")
    p.add_argument("--lambda-rad",     type=float, default=0.35,
                   help="λ_R weight for V2 delta_R channel")
    p.add_argument("--seed",           type=int,  default=42)
    p.add_argument("--n-boot",         type=int,  default=300)
    p.add_argument("--num-workers",    type=int,  default=0)
    p.add_argument("--batch-size",     type=int,  default=256,
                   help="Batch size for score collection (CPU-safe)")
    p.add_argument("--out-prefix",     type=str,  default="chamfer_ablation",
                   help="Prefix for all output files")
    return p


def main(argv=None):
    configure_system_resources()
    args = build_parser().parse_args(argv)
    fijar_semilla(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")
    print(f"[*] λ_E={args.lambda_energy}  λ_R={args.lambda_rad}  seed={args.seed}")

    D   = DEFAULTS
    MAX_P    = D["MAX_P"]
    MAX_ROWS = None if args.max_rows == -1 else args.max_rows

    # ── Preprocessor ─────────────────────────────────────────────
    prep = PrepV12(MAX_P)
    prep.load(args.scaler)

    # ── Datasets (same split for all variants) ───────────────────
    qcd_ds = JetDatasetV12(args.qcd,    prep, MAX_P,
                           max_rows=MAX_ROWS * 3 if MAX_ROWS else None)
    sig_ds = JetDatasetV12(args.signal, prep, MAX_P,
                           max_rows=MAX_ROWS)
    all_ds = ConcatDataset([qcd_ds, sig_ds])

    qcd_idx, sig_idx = [], []
    offset = 0
    for ds in all_ds.datasets:
        sig_m = ds.phys[:, 5] == 1.0
        qcd_m = ds.phys[:, 5] == 0.0
        sig_idx.extend((torch.where(sig_m)[0] + offset).tolist())
        qcd_idx.extend((torch.where(qcd_m)[0] + offset).tolist())
        offset += len(ds)
    print(f"[*] {len(qcd_idx)} QCD  /  {len(sig_idx)} signal jets")

    # Deterministic split: 50% train (unused here) | 30% val | 20% test
    g = torch.Generator().manual_seed(args.seed)
    qcd_shuf = torch.tensor(qcd_idx)[torch.randperm(len(qcd_idx), generator=g)].tolist()
    sig_shuf = torch.tensor(sig_idx)[torch.randperm(len(sig_idx), generator=g)].tolist()

    nq = len(qcd_shuf)
    nq_tr = int(0.50 * nq); nq_va = int(0.30 * nq)
    qcd_va  = qcd_shuf[nq_tr: nq_tr + nq_va]
    qcd_te  = qcd_shuf[nq_tr + nq_va:]

    ns = len(sig_shuf)
    ns_tr = int(0.50 * ns); ns_va = int(0.30 * ns)
    sig_va  = sig_shuf[ns_tr: ns_tr + ns_va]
    sig_te  = sig_shuf[ns_tr + ns_va:]

    # Split validation into DDT pool and grid-search pool (no overlap)
    nq_vh   = len(qcd_va) // 2
    qcd_va_ddt = qcd_va[:nq_vh]
    qcd_va_gs  = qcd_va[nq_vh:]

    dl_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                 pin_memory=(device.type == "cuda"))
    ddt_loader = DataLoader(Subset(all_ds, qcd_va_ddt), **dl_kw)
    gs_loader  = DataLoader(Subset(all_ds, qcd_va_gs + sig_va), **dl_kw)
    te_loader  = DataLoader(Subset(all_ds, qcd_te + sig_te), **dl_kw)
    print(f"[*] DDT pool={len(qcd_va_ddt)} | GS pool={len(qcd_va_gs)+len(sig_va)} "
          f"| Test={len(qcd_te)+len(sig_te)}")

    # ── Load frozen model ─────────────────────────────────────────
    model = JetAE_V12(n_part=MAX_P, latent=D["LATENT"], ch=D["CHANNELS"]).to(device)
    state = torch.load(args.model, map_location=device, weights_only=False)
    # Handle torch.compile wrapper
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[*] Model loaded: {n_params:,} parameters (frozen)")

    # ── Evaluate each variant ─────────────────────────────────────
    all_results = []
    for name, fn in VARIANT_FNS.items():
        res = evaluate_variant(
            name, fn, model,
            ddt_loader, gs_loader, te_loader,
            device, args, args.seed,
        )
        all_results.append(res)

    # ── Save metrics ──────────────────────────────────────────────
    metrics_path = f"{args.out_prefix}_metrics.json"
    # JSON-serialisable copy (drop large arrays to keep file small)
    json_results = []
    for r in all_results:
        jr = {k: v for k, v in r.items()
              if k not in ("fpr", "tpr", "sic",
                           "boot_fpr_grid",
                           "boot_tpr_p16", "boot_tpr_p50", "boot_tpr_p84",
                           "boot_sic_p16", "boot_sic_p50", "boot_sic_p84",
                           "qcd_scores", "qcd_masses")}
        json_results.append(jr)
    with open(metrics_path, "w") as f:
        json.dump({
            "lambda_energy": args.lambda_energy,
            "lambda_rad":    args.lambda_rad,
            "seed":          args.seed,
            "n_boot":        args.n_boot,
            "results":       json_results,
        }, f, indent=2)
    print(f"[*] Metrics saved → {metrics_path}")

    # ── Comparison figure ─────────────────────────────────────────
    fig_path = f"{args.out_prefix}_figure.pdf"
    plot_comparison(all_results, fig_path, args.n_boot)

    # ── LaTeX table ───────────────────────────────────────────────
    tex_path = f"{args.out_prefix}_jsd_table.tex"
    write_latex_table(all_results, tex_path)

    # ── Console summary table ─────────────────────────────────────
    print(f"\n{'=' * 75}")
    print(f"  ABLATION SUMMARY — seed={args.seed}  λ_E={args.lambda_energy}  λ_R={args.lambda_rad}")
    print(f"  {'Variant':<28}  {'AUC':>7}  {'Rej@50%':>9}  {'SIC_max':>8}  "
          f"{'JSD(1%)':>8}  {'JSD(0.5%)':>10}  {'ρ_corr':>8}")
    print(f"  {'─' * 73}")
    for r in all_results:
        j01  = r["jsd_by_fpr"].get("0.01",  np.nan)
        j005 = r["jsd_by_fpr"].get("0.005", np.nan)
        ok01  = "✓" if np.isfinite(j01)  and j01  < 0.04 else "[!]"
        ok005 = "✓" if np.isfinite(j005) and j005 < 0.04 else "[!]"
        print(
            f"  {r['label']:<28}  "
            f"{r['auc_combined']:>7.4f}  "
            f"{r['rej_at_50tpr']:>9.1f}  "
            f"{r['sic_max']:>8.3f}  "
            f"{j01:>7.4f}{ok01}  "
            f"{j005:>9.4f}{ok005}  "
            f"{r['rho_pearson']:>+8.4f}"
        )
    print(f"{'=' * 75}")
    print("\n[✓] Ablation complete.")


if __name__ == "__main__":
    main()
