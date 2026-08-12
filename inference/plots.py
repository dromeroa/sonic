"""
sonic.inference.plots
=====================
Visualisation functions for training curves, DDT 2‑D fits, and the full
10‑panel physics‑analysis dashboard.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, roc_curve

from sonic.utils.metrics import compute_jsd


# ---------------------------------------------------------------------------
# Training history
# ---------------------------------------------------------------------------
def plot_training(history: dict, save_path: str = "training_history_v12.png"):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    ep = range(1, len(history["train"]) + 1)

    axes[0].plot(ep, history["train"], label="Train total", color="steelblue", lw=2)
    axes[0].plot(ep, history["val"], label="Val total", color="darkorange", ls="--", lw=2)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Total loss")
    axes[0].set_title("Total loss (Chamfer + Neff + DisCo)")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(ep, history["train_ch"], label="Chamfer", color="steelblue", lw=2)
    axes[1].plot(ep, history["train_neff"], label="Neff", color="purple", lw=2)
    if history.get("train_disco"):
        axes[1].plot(ep, history["train_disco"], label="DisCo", color="forestgreen", lw=2)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Component")
    axes[1].set_title("Loss decomposition"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    if history.get("val_jsd"):
        axes[2].plot(ep, history["val_jsd"], label="JSD(0.5% QCD)", color="crimson", lw=2)
        axes[2].axhline(0.04, color="gray", ls="--", lw=1, label="target 0.04")
        axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("JSD")
        axes[2].set_title("Validation JSD(0.5% QCD) per epoch")
        axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# DDT 2‑D fit diagnostics
# ---------------------------------------------------------------------------
def plot_ddt2d_fit(xs, ys, zs, ddt_model, rho_qcd, logpt_qcd, s_raw_qcd, s_ddt_qcd, save_path):
    if len(xs) == 0:
        print(f"[!] Cannot plot DDT fit for '{ddt_model.name}' (no bins).")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sc = axes[0].scatter(xs, ys, c=zs, cmap="RdYlBu_r", s=60, zorder=5)
    plt.colorbar(sc, ax=axes[0], label="Score quantile")
    axes[0].set_xlabel("ρ normalised"); axes[0].set_ylabel("log pT normalised")
    title_ok = "" if ddt_model.fit_ok else "  [NO FIT]"
    axes[0].set_title(f"DDT2D quantiles — {ddt_model.name}{title_ok}")
    axes[0].grid(True, alpha=0.3)

    r1, _ = pearsonr(s_raw_qcd, rho_qcd)
    axes[1].hexbin(rho_qcd, s_raw_qcd, gridsize=50, cmap="Blues", mincnt=1, bins="log")
    axes[1].set_xlabel("ρ"); axes[1].set_ylabel(f"{ddt_model.name} raw")
    axes[1].set_title(f"Raw vs ρ  (r={r1:+.3f})"); axes[1].grid(True, alpha=0.3)

    r2, _ = pearsonr(s_ddt_qcd, rho_qcd)
    axes[2].hexbin(rho_qcd, s_ddt_qcd, gridsize=50, cmap="Greens", mincnt=1, bins="log")
    axes[2].set_xlabel("ρ"); axes[2].set_ylabel(f"{ddt_model.name} DDT2D")
    axes[2].set_title(f"DDT2D vs ρ  (r={r2:+.3f})"); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# Full 10‑panel dashboard
# ---------------------------------------------------------------------------
def plot_all(
    C_log, fill_var, s_raw, s_ddt, labels, masses, rho, auc, boot_stats,
    auc_ae_only, auc_tau21_only=None, s_tau21_ddt=None,
    auc_n2_only=None, s_n2_ddt=None, jsd_boot_bands=None, save_prefix="v12",
):
    tau21 = fill_var
    sig = labels == 1
    qcd = labels == 0
    s_dir = s_ddt
    s_raw_dir = s_raw

    fpr_ddt, tpr_ddt, _ = roc_curve(labels, s_dir)
    fpr_raw, tpr_raw, _ = roc_curve(labels, s_raw_dir)
    auc_raw = roc_auc_score(labels, s_raw_dir)

    with np.errstate(divide="ignore"):
        rej_ddt = np.where(fpr_ddt > 0, 1.0 / fpr_ddt, np.nan)

    idx50 = np.argmin(np.abs(tpr_ddt - 0.50))
    rej50 = rej_ddt[idx50]

    # WP de máximo SIC sobre la curva ROC del discriminante combinado DDT.
    # TPR/sqrt(FPR) es máximo en el punto de mayor significancia estadística,
    # que es más informativo que TPR=50% para reportar en publicación.
    with np.errstate(divide="ignore", invalid="ignore"):
        sic_roc = np.where(fpr_ddt > 0, tpr_ddt / np.sqrt(fpr_ddt), np.nan)
    idx_sic_max = np.nanargmax(sic_roc)
    tpr_sic_max = tpr_ddt[idx_sic_max]
    fpr_sic_max = fpr_ddt[idx_sic_max]
    rej_sic_max = rej_ddt[idx_sic_max]

    percs = np.linspace(50, 99.5, 200)
    sig_vals, eff_s, eff_b = [], [], []
    N_s, N_b = sig.sum(), qcd.sum()
    # CORRECCIÓN: umbral definido solo sobre QCD.
    # np.percentile sobre la muestra mixta (QCD+señal) sesga el corte porque
    # incluye eventos de señal en el percentil, produciendo S/√B artificialmente
    # optimista. En un análisis real el corte se define sobre fondo (MC QCD /
    # data sideband) sin acceso a la distribución de señal.
    qcd_scores_all = s_dir[qcd]
    for pct in percs:
        thr = np.percentile(qcd_scores_all, pct)   # ← solo QCD
        ns = np.sum(s_dir[sig] >= thr)
        nb = np.sum(s_dir[qcd] >= thr)
        sig_vals.append(ns / np.sqrt(nb + 1e-6))
        eff_s.append(ns / (N_s + 1e-6))
        eff_b.append(nb / (N_b + 1e-6))
    best_sig_idx = np.argmax(sig_vals)



    qcd_scores = s_dir[qcd]
    qcd_masses = masses[qcd]
    bins_m = np.linspace(40, 200, 41)
    h_ref, _ = np.histogram(qcd_masses, bins=bins_m, density=True)
    fpr_cuts = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005]
    colors_ms = ["black", "blue", "orange", "red", "purple", "magenta"]
    lbls_ms = ["100% QCD", "50% QCD", "10% QCD", "5% QCD", "1% QCD", "0.5% QCD"]
    jsd_vals = {}

    fig = plt.figure(figsize=(26, 13))
    gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.40, wspace=0.35)

    # Panel 0: ROC
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(fpr_raw, tpr_raw, color="gray", lw=1.5, ls="--", label=f"Raw AUC={auc_raw:.3f}")
    if s_tau21_ddt is not None and auc_tau21_only is not None:
        fpr_t, tpr_t, _ = roc_curve(labels, s_tau21_ddt)
        ax0.plot(fpr_t, tpr_t, color="cornflowerblue", lw=1.8, ls="-.",
                 label=f"τ₂₁ DDT AUC={auc_tau21_only:.3f}")
    if s_n2_ddt is not None and auc_n2_only is not None:
        fpr_n, tpr_n, _ = roc_curve(labels, s_n2_ddt)
        ax0.plot(fpr_n, tpr_n, color="mediumseagreen", lw=1.8, ls=":",
                 label=f"N2 DDT AUC={auc_n2_only:.3f}")
    ax0.plot(fpr_ddt, tpr_ddt, color="darkorange", lw=2.5, label=f"Combined AUC={auc:.3f}")
    if boot_stats is not None:
        fg = boot_stats["fpr_grid"]
        ax0.fill_between(fg, boot_stats["tpr"]["p16"], boot_stats["tpr"]["p84"],
                         color="darkorange", alpha=0.2, label="68% CI")
    ax0.scatter([fpr_ddt[idx50]], [tpr_ddt[idx50]], s=60, color="gray", zorder=4,
                label=f"TPR=50% Rej={rej50:.0f}", alpha=0.8)
    ax0.scatter([fpr_sic_max], [tpr_sic_max], s=80, color="red", zorder=5,
                label=f"Max SIC (TPR={tpr_sic_max:.2f})")
    ax0.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    ax0.set_xlabel("FPR"); ax0.set_ylabel("TPR"); ax0.set_title("ROC")
    ax0.legend(fontsize=7); ax0.grid(True, alpha=0.3)

    # Panel 1: Rejection
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(tpr_ddt, rej_ddt, color="steelblue", lw=2.5)
    ax1.set_yscale("log")
    ax1.axhline(100,  color="gray", ls=":",  lw=1)
    ax1.axhline(1000, color="gray", ls="--", lw=1)
    # WP convencional TPR=50% — referencia secundaria (gris)
    ax1.axvline(0.5, color="gray", ls=":", lw=1.2, alpha=0.6)
    ax1.scatter([tpr_ddt[idx50]], [rej50], s=60, color="gray", zorder=4,
                label=f"Rej@TPR=50%={rej50:.0f}", alpha=0.8)
    # WP óptimo = máximo SIC — punto principal (rojo)
    ax1.axvline(tpr_sic_max, color="red", ls="--", lw=1.2, alpha=0.7)
    ax1.scatter([tpr_sic_max], [rej_sic_max], s=90, color="red", zorder=5,
                label=f"Max SIC: TPR={tpr_sic_max:.2f}, Rej={rej_sic_max:.0f}")
    ax1.set_xlabel("TPR"); ax1.set_ylabel("1/FPR"); ax1.set_title("Rejection")
    ax1.legend(fontsize=7); ax1.grid(True, alpha=0.3, which="both")
    ax1.set_xlim(0, 1)
    fin = rej_ddt[np.isfinite(rej_ddt)]
    if len(fin):
        ax1.set_ylim(1, fin.max() * 2)

    # Panel 2: SIC
    ax2 = fig.add_subplot(gs[0, 2])
    with np.errstate(divide="ignore", invalid="ignore"):
        sic_ddt = tpr_ddt / np.sqrt(fpr_ddt)
    ax2.plot(fpr_ddt, sic_ddt, color="forestgreen", lw=2.5, label="SIC")
    if boot_stats is not None:
        fg = boot_stats["fpr_grid"]
        ax2.fill_between(fg, boot_stats["sic"]["p16"], boot_stats["sic"]["p84"],
                         color="forestgreen", alpha=0.2, label="68% CI")
    ax2.set_xlim(0, 1)
    fin_s = sic_ddt[np.isfinite(sic_ddt)]
    if len(fin_s):
        ax2.set_ylim(0, np.nanpercentile(fin_s, 99) * 1.3)
    ax2.set_xlabel("FPR"); ax2.set_ylabel("SIC"); ax2.set_title("SIC")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    # Panel 3: Score distribution
    ax3 = fig.add_subplot(gs[0, 3])
    bins_s = np.linspace(np.percentile(s_dir, 1), np.percentile(s_dir, 99), 60)
    ax3.hist(s_dir[qcd], bins=bins_s, density=True, histtype="step", lw=2, color="blue", label="QCD")
    ax3.hist(s_dir[sig], bins=bins_s, density=True, histtype="step", lw=2, color="red", label="WW")
    ax3.set_xlabel("DDT Score"); ax3.set_ylabel("Density"); ax3.set_title("Score distribution")
    ax3.legend(); ax3.grid(True, alpha=0.3)

    # Panel 4: Residual correlation
    ax4 = fig.add_subplot(gs[0, 4])
    r_f, _ = pearsonr(s_ddt[qcd], rho[qcd])
    ax4.hexbin(rho[qcd], s_ddt[qcd], gridsize=50, cmap="plasma", mincnt=1, bins="log")
    ax4.set_xlabel("ρ"); ax4.set_ylabel("DDT score")
    ax4.set_title(f"Decorrelation (r={r_f:+.4f})"); ax4.grid(True, alpha=0.3)

    # Panel 5: Significance — S/√B relativo (proporcional a significancia física)
    # El umbral se define sobre QCD únicamente (corregido vs versión anterior).
    # Los valores absolutos dependen del tamaño de muestra; la figura muestra
    # mejora relativa entre discriminantes. Ver caption LaTeX para aclaración.
    ax5 = fig.add_subplot(gs[1, 0])
    ax5b = ax5.twinx()

    # Calcular curvas de baselines para comparación directa en el mismo panel
    sig_vals_t21, sig_vals_n2 = [], []
    if s_tau21_ddt is not None:
        qcd_t21 = s_tau21_ddt[qcd]
        for pct in percs:
            thr = np.percentile(qcd_t21, pct)
            ns  = np.sum(s_tau21_ddt[sig] >= thr)
            nb  = np.sum(s_tau21_ddt[qcd] >= thr)
            sig_vals_t21.append(ns / np.sqrt(nb + 1e-6))
    if s_n2_ddt is not None:
        qcd_n2 = s_n2_ddt[qcd]
        for pct in percs:
            thr = np.percentile(qcd_n2, pct)
            ns  = np.sum(s_n2_ddt[sig] >= thr)
            nb  = np.sum(s_n2_ddt[qcd] >= thr)
            sig_vals_n2.append(ns / np.sqrt(nb + 1e-6))

    # Baselines (líneas finas)
    if sig_vals_t21:
        ax5.plot(100 - percs, sig_vals_t21, color="cornflowerblue",
                 lw=1.4, ls="-.", label="τ₂₁ DDT", alpha=0.85)
    if sig_vals_n2:
        ax5.plot(100 - percs, sig_vals_n2, color="mediumseagreen",
                 lw=1.4, ls=":",  label="N₂ DDT",  alpha=0.85)

    # Discriminante combinado (línea principal)
    ax5.plot(100 - percs, sig_vals, color="darkorange", lw=2.5,
             label="Combined DDT2D")
    ax5.scatter([100 - percs[best_sig_idx]], [sig_vals[best_sig_idx]],
                s=90, color="darkorange", zorder=5,
                label=f"Max S/√B @ {100 - percs[best_sig_idx]:.1f}% QCD")

    # Eficiencias en eje derecho
    ax5b.plot(100 - percs, eff_s, color="green",  lw=1.3, ls="--",
              label="ε_sig", alpha=0.75)
    ax5b.plot(100 - percs, eff_b, color="red",    lw=1.3, ls=":",
              label="ε_bkg", alpha=0.75)
    ax5b.set_ylim(0, 1)

    ax5.set_xlabel("% QCD retained  (← tighter cut)")
    ax5.set_ylabel("S/√B  (relative, MC sample)")
    ax5b.set_ylabel("Efficiency")
    ax5.set_title("Significance")
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0, 50)
    ax5.set_ylim(bottom=0)
    l1, lb1 = ax5.get_legend_handles_labels()
    l2, lb2 = ax5b.get_legend_handles_labels()
    ax5.legend(l1 + l2, lb1 + lb2, fontsize=7, loc="upper right")

    # Panel 6: Mass sculpting RAW
    ax6 = fig.add_subplot(gs[1, 1])
    qcd_s_raw = s_raw_dir[qcd]
    for fpr_c, color, lbl in zip(fpr_cuts, colors_ms, lbls_ms):
        thr = np.percentile(qcd_s_raw, (1.0 - fpr_c) * 100)
        m_pass = qcd_masses[qcd_s_raw >= thr]
        h_cut, _ = np.histogram(m_pass, bins=bins_m, density=True)
        jsd = compute_jsd(h_ref, h_cut) if fpr_c < 1.0 else None
        lbl_s = f"{lbl} JSD={jsd:.3f}" if jsd else lbl
        ax6.hist(m_pass, bins=bins_m, density=True, histtype="step", lw=2, color=color, alpha=0.85, label=lbl_s)
    ax6.axvspan(65, 95, color="gray", alpha=0.15, label="W window")
    ax6.set_xlabel("SDMass [GeV]"); ax6.set_ylabel("Density"); ax6.set_title("Sculpting — RAW")
    ax6.legend(loc="upper right", fontsize=7); ax6.grid(True, alpha=0.3)

    # Panel 7: Mass sculpting DDT
    ax7 = fig.add_subplot(gs[1, 2])
    print(f"\n  {'FPR':>6}  |  {'JSD raw':>8}  |  {'JSD DDT':>8}  |  {'68% CI':>16}  |  {'N evts':>8}")
    print("  " + "-" * 65)
    for fpr_c, color, lbl in zip(fpr_cuts, colors_ms, lbls_ms):
        thr = np.percentile(qcd_scores, (1.0 - fpr_c) * 100)
        mask_p = qcd_scores >= thr
        m_pass = qcd_masses[mask_p]
        h_cut, _ = np.histogram(m_pass, bins=bins_m, density=True)
        if fpr_c < 1.0:
            jsd = compute_jsd(h_ref, h_cut)
            jsd_vals[f"{fpr_c * 100:.1f}% QCD"] = jsd
            thr_r = np.percentile(qcd_s_raw, (1.0 - fpr_c) * 100)
            h_r, _ = np.histogram(qcd_masses[qcd_s_raw >= thr_r], bins=bins_m, density=True)
            jsd_r = compute_jsd(h_ref, h_r)
            ci_str = "n/a"
            if jsd_boot_bands is not None and fpr_c in jsd_boot_bands:
                p16, p50, p84 = jsd_boot_bands[fpr_c]
                if np.isfinite(p16):
                    ci_str = f"[{p16:.3f}, {p84:.3f}]"
            print(f"  {fpr_c * 100:>5.1f}%  |  {jsd_r:>8.4f}  |  {jsd:>8.4f}  |  {ci_str:>16}  |  {mask_p.sum():>8d}")
            lbl_s = f"{lbl} JSD={jsd:.3f}"
        else:
            lbl_s = lbl
        ax7.hist(m_pass, bins=bins_m, density=True, histtype="step", lw=2, color=color, alpha=0.85, label=lbl_s)
    ax7.axvspan(65, 95, color="gray", alpha=0.15, label="W window")
    ax7.set_xlabel("SDMass [GeV]"); ax7.set_ylabel("Density"); ax7.set_title("Sculpting — DDT")
    ax7.legend(loc="upper right", fontsize=7); ax7.grid(True, alpha=0.3)

    # Panel 8: log(C) vs τ₂₁
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.hexbin(tau21[qcd], C_log[qcd], gridsize=50, cmap="Blues", mincnt=1, bins="log", alpha=0.8)
    ax8.hexbin(tau21[sig], C_log[sig], gridsize=50, cmap="Reds", mincnt=1, bins="log", alpha=0.6)
    ax8.set_xlabel("τ₂₁"); ax8.set_ylabel("log(Chamfer)"); ax8.set_title("log(C) vs τ₂₁")
    ax8.grid(True, alpha=0.3)

    # Panel 9: Summary text
    ax9 = fig.add_subplot(gs[1, 4])
    ax9.axis("off")
    t21_line = f"AUC τ₂₁ DDT   = {auc_tau21_only:.4f}\n" if auc_tau21_only is not None else ""
    n2_line = f"AUC N2 DDT    = {auc_n2_only:.4f}\n" if auc_n2_only is not None else ""
    delta = ""
    if auc_tau21_only is not None:
        delta += f"ΔAUC(comb−τ₂₁) = {auc - auc_tau21_only:+.4f}\n"
    if auc_n2_only is not None:
        delta += f"ΔAUC(comb−N2)  = {auc - auc_n2_only:+.4f}\n"
    delta += f"ΔAUC(comb−AE)  = {auc - auc_ae_only:+.4f}\n"
    txt = (
        f"AUC combined     = {auc:.4f}\n{t21_line}{n2_line}"
        f"AUC AE only      = {auc_ae_only:.4f}\nAUC raw          = {auc_raw:.4f}\n\n"
        f"{delta}\nRej@TPR=50%      = {rej50:.1f}  (ref)\n"
        f"Rej@MaxSIC       = {rej_sic_max:.0f}  (TPR={tpr_sic_max:.2f})\n"
        f"Max S/√B         = {max(sig_vals):.2f}\n"
        f"  @ {100 - percs[best_sig_idx]:.1f}% QCD\n"
        f"  [relative, MC sample]"
    )
    ax9.text(0.02, 0.98, txt, transform=ax9.transAxes, fontsize=9,
             va="top", ha="left", family="monospace",
             bbox=dict(boxstyle="round", facecolor="whitesmoke", alpha=0.8))

    fig.suptitle(
        f"SONIC v4 — AUC={auc:.3f}  |  AUC(AE)={auc_ae_only:.3f}  |  "
        f"Rej@MaxSIC={rej_sic_max:.0f} (TPR={tpr_sic_max:.2f})  |  "
        f"Rej@50%={rej50:.0f}  |  JSD(1%)={jsd_vals.get('1.0% QCD', 999):.3f}",
        fontsize=12, fontweight="bold",
    )
    save_path = f"{save_prefix}_panel_sonic.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n[*] Dashboard saved to '{save_path}'")

    # ---- Console summary ----
    print(f"\n{'=' * 65}")
    print(f"  AUC (DDT2D combined)  = {auc:.4f}")
    if auc_tau21_only is not None:
        print(f"  AUC (τ₂₁ DDT solo)   = {auc_tau21_only:.4f}")
    if auc_n2_only is not None:
        print(f"  AUC (N2 DDT solo)    = {auc_n2_only:.4f}")
    print(f"  AUC (AE solo)        = {auc_ae_only:.4f}")
    print(f"  Rej@TPR=50%          = {rej50:.1f}  (referencia convencional)")
    print(f"  Rej@MaxSIC           = {rej_sic_max:.0f}  (TPR={tpr_sic_max:.3f}, FPR={fpr_sic_max:.4f})")
    print(f"  Max S/√B             = {max(sig_vals):.2f}  @ {100-percs[best_sig_idx]:.1f}% QCD  [relativo, MC sample]")
    print(f"  JSD by cut (DDT2D):")
    for lbl, jsd in jsd_vals.items():
        ok = "✓" if jsd < 0.04 else "[!]"
        extra = ""
        if jsd_boot_bands is not None:
            try:
                fpr_key = float(lbl.split("%")[0]) / 100.0
                if fpr_key in jsd_boot_bands and np.isfinite(jsd_boot_bands[fpr_key][0]):
                    p16, p50, p84 = jsd_boot_bands[fpr_key]
                    # FIX: reportar p50 del bootstrap ademas del CI.
                    # El valor puntual 'jsd' se calcula sobre el dataset completo;
                    # el CI bootstrap [p16,p84] se calcula sobre resamples.
                    # En colas extremas (0.5% QCD) hay muy pocos jets: el valor
                    # puntual puede diferir del p50 bootstrap porque cada resample
                    # selecciona jets distintos de la cola. Mostrar ambos permite
                    # distinguir varianza estadistica real de artefacto de N pequeno.
                    # Si jsd esta fuera de [p16,p84] es senal de que N_jets en la
                    # cola es demasiado pequeno para estimar JSD de forma robusta.
                    outside = " (*)" if (jsd < p16 or jsd > p84) else ""
                    extra = f"   boot_p50={p50:.4f}  CI=[{p16:.4f}, {p84:.4f}]{outside}"
            except Exception:
                pass
        print(f"    {lbl:20s}: {jsd:.4f}  {ok}{extra}")
    print(f"{'=' * 65}")

    return rej50, sig_vals[best_sig_idx], jsd_vals
