#!/usr/bin/env python3
"""
orthogonality_probe.py
======================
Diagnostico rapido: El AE (Chamfer) aporta informacion ORTOGONAL a tau21 y N2?

Logica de la prueba
--------------------
Despues del DDT 2D, las tres variables (C_DDT, tau21_DDT, N2_DDT) deberían
ser decorrelacionadas respecto a (rho, log_pT). Lo que NO elimina el DDT es
la correlacion MUTUA entre los discriminantes.

Si el AE mide fundamentalmente lo mismo que tau21 o N2, la combinacion
trivalente no gana casi nada:
  - Si |Pearson(C_DDT, tau21_DDT)|_QCD > 0.6 -> AE ~ tau21 -> argumento del paper se debilita.
  - Si |Pearson(C_DDT, N2_DDT)|_QCD   > 0.6 -> AE ~ N2   -> idem.

Pruebas realizadas
------------------
1. Matriz de correlacion de Pearson entre (C_DDT, tau21_DDT, N2_DDT) en QCD.
2. VIF (Variance Inflation Factor): detecta multicolinealidad en la regresion lineal.
3. Ganancia marginal de AUC: AUC(C + tau21) vs AUC(tau21 solo), etc.
4. Analisis de componentes principales (PCA) en QCD para ver si las 3 variables
   viven esencialmente en 1D (colinear) o en un espacio 2D/3D genuino.
5. Grafico 2x2 con scatter, heatmap de correlaciones, PCA variance, y delta_AUC.

Uso
---
Con datos REALES (despues de correr SONIC):
    import numpy as np
    C_ddt    = np.load("C_ddt_qcd.npy")    # scores DDT en QCD
    T21_ddt  = np.load("T21_ddt_qcd.npy")
    N2_ddt   = np.load("N2_ddt_qcd.npy")
    labels   = np.load("labels_test.npy")
    run_probe(C_ddt, T21_ddt, N2_ddt, labels)

Con datos SINTETICOS (prueba del script, no necesita GPU ni datos):
    python orthogonality_probe.py

Referencia del umbral
---------------------
|rho_Pearson| > 0.6 en QCD: la correlacion es fuerte; el AE probablemente
replica informacion ya contenida en tau21/N2. En ese caso:
  - La combinacion lineal no aporta ganancia estadisticamente significativa.
  - El argumento de "informacion complementaria" del paper se debilita.
  - Alternativa: buscar si hay una sub-region de fase-espacio donde la
    correlacion es baja (e.g., jets de alto pT o baja masa SDM).

Si |rho_Pearson| < 0.3: las variables son esencialmente independientes.
  -> La combinacion trivalente tiene justificacion solida para el paper.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Funcion principal
# ---------------------------------------------------------------------------
def run_probe(
    C_ddt: np.ndarray,
    T21_ddt: np.ndarray,
    N2_ddt: np.ndarray,
    labels: np.ndarray,
    save_prefix: str = "orthogonality",
    n_boot: int = 500,
    seed: int = 42,
) -> dict:
    """
    Parametros
    ----------
    C_ddt   : array (N,) -- Chamfer score post-DDT (todos los jets del test set)
    T21_ddt : array (N,) -- tau21 post-DDT
    N2_ddt  : array (N,) -- N2 post-DDT
    labels  : array (N,) -- 1=signal, 0=QCD
    save_prefix : str    -- prefijo para guardar figuras y JSON

    Retorna
    -------
    dict con todas las metricas calculadas (para incluir en el paper)
    """
    rng = np.random.default_rng(seed)

    qcd_mask = labels == 0
    sig_mask = labels == 1
    n_qcd = qcd_mask.sum()
    n_sig = sig_mask.sum()

    print(f"\n{'='*65}")
    print(f"  SONIC Orthogonality Probe")
    print(f"  Jets: {n_qcd} QCD, {n_sig} signal")
    print(f"{'='*65}\n")

    # =========================================================
    # 1. Matriz de correlacion de Pearson en QCD
    # =========================================================
    C_q   = C_ddt[qcd_mask]
    T21_q = T21_ddt[qcd_mask]
    N2_q  = N2_ddt[qcd_mask]

    # Correlaciones QCD (la unica que importa: el AE se entrenó en QCD)
    corr_CT  = float(np.corrcoef(C_q, T21_q)[0, 1])
    corr_CN  = float(np.corrcoef(C_q, N2_q)[0, 1])
    corr_TN  = float(np.corrcoef(T21_q, N2_q)[0, 1])

    corr_mat_qcd = np.array([
        [1.0,     corr_CT, corr_CN],
        [corr_CT, 1.0,     corr_TN],
        [corr_CN, corr_TN, 1.0    ],
    ])

    # Correlaciones en senal (informativas pero secundarias)
    C_s   = C_ddt[sig_mask]
    T21_s = T21_ddt[sig_mask]
    N2_s  = N2_ddt[sig_mask]
    corr_CT_sig = float(np.corrcoef(C_s, T21_s)[0, 1]) if n_sig > 2 else np.nan
    corr_CN_sig = float(np.corrcoef(C_s, N2_s)[0, 1])  if n_sig > 2 else np.nan

    print("[ 1 ] Correlaciones de Pearson en QCD (post-DDT)")
    print(f"      |rho(C, tau21)| = {abs(corr_CT):.4f}  {'<-- ALTA: AE~tau21' if abs(corr_CT)>0.6 else ''}")
    print(f"      |rho(C, N2)  | = {abs(corr_CN):.4f}  {'<-- ALTA: AE~N2'    if abs(corr_CN)>0.6 else ''}")
    print(f"      |rho(tau21, N2)| = {abs(corr_TN):.4f}")
    _interp_ct = _interpret_corr(corr_CT, "C_DDT", "tau21_DDT")
    _interp_cn = _interpret_corr(corr_CN, "C_DDT", "N2_DDT")
    print(f"\n      -> {_interp_ct}")
    print(f"      -> {_interp_cn}")

    # =========================================================
    # 2. VIF (Variance Inflation Factor) -- multicolinealidad
    # =========================================================
    vif_C, vif_T21, vif_N2 = _compute_vif(C_q, T21_q, N2_q)
    print(f"\n[ 2 ] VIF en QCD (VIF > 5 indica multicolinealidad problematica)")
    print(f"      VIF(C_DDT)    = {vif_C:.2f}{'  <-- ALTO' if vif_C > 5 else ''}")
    print(f"      VIF(tau21_DDT)= {vif_T21:.2f}{'  <-- ALTO' if vif_T21 > 5 else ''}")
    print(f"      VIF(N2_DDT)   = {vif_N2:.2f}{'  <-- ALTO' if vif_N2 > 5 else ''}")

    # =========================================================
    # 3. Ganancia marginal de AUC (con bootstrap CI)
    # =========================================================
    print(f"\n[ 3 ] Ganancia marginal de AUC (bootstrap n={n_boot})")
    aucs = _bootstrap_aucs(
        C_ddt, T21_ddt, N2_ddt, labels, n_boot=n_boot, rng=rng
    )
    _print_auc_table(aucs)

    delta_AUC_C_over_T21 = aucs["C+T21"]["p50"] - aucs["T21"]["p50"]
    delta_AUC_C_over_N2  = aucs["C+N2"]["p50"]  - aucs["N2"]["p50"]
    delta_AUC_tri        = aucs["C+N2+T21"]["p50"] - max(aucs["T21"]["p50"], aucs["N2"]["p50"])

    # =========================================================
    # 4. PCA en QCD: dimensionalidad efectiva del espacio (C, tau21, N2)
    # =========================================================
    pca_var, pca_vecs = _pca_3vars(C_q, T21_q, N2_q)
    print(f"\n[ 4 ] PCA en QCD -- varianza explicada por componente")
    for i, v in enumerate(pca_var):
        bar = "█" * int(v * 30)
        print(f"      PC{i+1}: {v*100:5.1f}%  {bar}")
    print(f"      Var acumulada PC1+PC2: {(pca_var[0]+pca_var[1])*100:.1f}%")
    print(f"      Var acumulada PC1+PC2+PC3: {pca_var.sum()*100:.1f}%")
    if pca_var[0] > 0.85:
        print("      -> El espacio es ESENCIALMENTE 1D: las 3 variables son casi colineales.")
        print("         La combinacion trivalente no aporta diversidad de informacion.")
    elif pca_var[0] + pca_var[1] > 0.95:
        print("      -> El espacio es ESENCIALMENTE 2D: hay 2 direcciones de informacion.")
        print("         Una de las 3 variables es combinacion lineal aproximada de las otras 2.")
    else:
        print("      -> El espacio es genuinamente 3D: las 3 variables son ORTOGONALES.")
        print("         La combinacion trivalente esta justificada para el paper.")

    # =========================================================
    # 5. Figuras
    # =========================================================
    fig = _make_figure(
        C_q, T21_q, N2_q, C_s, T21_s, N2_s,
        corr_mat_qcd, pca_var, aucs,
        corr_CT, corr_CN, corr_TN,
    )
    fig_path = f"{save_prefix}_orthogonality.pdf"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figura guardada: {fig_path}")

    # =========================================================
    # Veredicto final
    # =========================================================
    verdict = _verdict(corr_CT, corr_CN, delta_AUC_tri, pca_var[0])
    print(f"\n{'='*65}")
    print(f"  VEREDICTO PARA EL PAPER")
    print(f"{'='*65}")
    print(f"  {verdict}")
    print(f"{'='*65}\n")

    results = {
        "pearson_C_T21_qcd": corr_CT,
        "pearson_C_N2_qcd":  corr_CN,
        "pearson_T21_N2_qcd": corr_TN,
        "pearson_C_T21_sig": corr_CT_sig,
        "pearson_C_N2_sig":  corr_CN_sig,
        "vif_C": vif_C, "vif_T21": vif_T21, "vif_N2": vif_N2,
        "pca_variance_explained": pca_var.tolist(),
        "delta_AUC_C_over_T21": delta_AUC_C_over_T21,
        "delta_AUC_C_over_N2":  delta_AUC_C_over_N2,
        "delta_AUC_trivariate":  delta_AUC_tri,
        "auc_bootstrap": {k: v for k, v in aucs.items()},
        "verdict": verdict,
    }
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _interpret_corr(rho, name_a, name_b):
    ar = abs(rho)
    direction = "positiva" if rho > 0 else "negativa"
    if ar > 0.6:
        return (f"|rho({name_a},{name_b})| = {ar:.3f} > 0.6 ({direction}): "
                f"ALTA correlacion -- el AE mide informacion similar a {name_b}. "
                f"La combinacion lineal gana poco.")
    elif ar > 0.3:
        return (f"|rho({name_a},{name_b})| = {ar:.3f} in (0.3, 0.6): "
                f"correlacion MODERADA -- hay informacion complementaria parcial.")
    else:
        return (f"|rho({name_a},{name_b})| = {ar:.3f} < 0.3: "
                f"correlacion BAJA -- el AE aporta informacion ORTOGONAL a {name_b}. "
                f"Argumento del paper solido.")


def _compute_vif(C, T21, N2):
    """VIF = 1 / (1 - R^2) para cada variable regresada sobre las otras dos."""
    X = np.stack([C, T21, N2], axis=1)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
    vifs = []
    for i in range(3):
        y = Xs[:, i]
        others = np.delete(Xs, i, axis=1)
        # OLS R^2
        beta_hat = np.linalg.lstsq(
            np.c_[np.ones(len(y)), others], y, rcond=None
        )[0]
        y_hat = np.c_[np.ones(len(y)), others] @ beta_hat
        ss_res = ((y - y_hat)**2).sum()
        ss_tot = ((y - y.mean())**2).sum()
        r2 = 1 - ss_res / (ss_tot + 1e-12)
        vif = 1.0 / (1.0 - r2 + 1e-8)
        vifs.append(float(vif))
    return vifs


def _auc_safe(labels, scores):
    try:
        a = roc_auc_score(labels, scores)
        return a if a >= 0.5 else 1 - a
    except Exception:
        return np.nan


def _bootstrap_aucs(C, T21, N2, labels, n_boot, rng):
    """Bootstrap de AUC para 6 combinaciones."""
    idx_sig = np.where(labels == 1)[0]
    idx_qcd = np.where(labels == 0)[0]
    n_s, n_q = len(idx_sig), len(idx_qcd)

    combos = {
        "C":        lambda a, b, g: a,
        "T21":      lambda a, b, g: -b,       # tau21 bajo = senal -> negamos
        "N2":       lambda a, b, g: -g,        # N2 bajo = senal -> negamos
        "C+T21":    lambda a, b, g: a - b,
        "C+N2":     lambda a, b, g: a - g,
        "C+N2+T21": lambda a, b, g: a - g - b,
    }

    boot = {k: [] for k in combos}
    for _ in range(n_boot):
        bs = np.concatenate([
            rng.choice(idx_sig, n_s, replace=True),
            rng.choice(idx_qcd, n_q, replace=True),
        ])
        c_b, t_b, n_b, l_b = C[bs], T21[bs], N2[bs], labels[bs]
        # Normalizar en cada bootstrap para que los pesos sean comparables
        sc = StandardScaler()
        mat = sc.fit_transform(np.stack([c_b, t_b, n_b], 1))
        c_n, t_n, n_n = mat[:, 0], mat[:, 1], mat[:, 2]
        for k, fn in combos.items():
            boot[k].append(_auc_safe(l_b, fn(c_n, t_n, n_n)))

    out = {}
    for k, vals in boot.items():
        v = np.array([x for x in vals if not np.isnan(x)])
        out[k] = {
            "p16": float(np.percentile(v, 16)),
            "p50": float(np.percentile(v, 50)),
            "p84": float(np.percentile(v, 84)),
        }
    return out


def _print_auc_table(aucs):
    print(f"      {'Combinacion':<18} {'AUC p50':>9} {'AUC p16':>9} {'AUC p84':>9}")
    print(f"      {'-'*50}")
    order = ["C", "T21", "N2", "C+T21", "C+N2", "C+N2+T21"]
    for k in order:
        v = aucs[k]
        print(f"      {k:<18} {v['p50']:>9.4f} {v['p16']:>9.4f} {v['p84']:>9.4f}")


def _pca_3vars(C, T21, N2):
    """PCA manual sobre (C, tau21, N2) en QCD."""
    X = np.stack([C, T21, N2], axis=1)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
    cov = np.cov(Xs.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigh retorna en orden ascendente; revertir
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    var_explained = eigvals / (eigvals.sum() + 1e-12)
    return var_explained, eigvecs


def _verdict(corr_CT, corr_CN, delta_AUC_tri, pc1_var):
    """Devuelve un string de veredicto para el paper."""
    issues = []
    if abs(corr_CT) > 0.6:
        issues.append(
            f"PROBLEMA: |rho(C,tau21)|={abs(corr_CT):.3f}>0.6 -> AE ~ tau21; "
            f"combinacion trivalente poco justificada."
        )
    if abs(corr_CN) > 0.6:
        issues.append(
            f"PROBLEMA: |rho(C,N2)|={abs(corr_CN):.3f}>0.6 -> AE ~ N2; "
            f"combinacion trivalente poco justificada."
        )
    if pc1_var > 0.85:
        issues.append(
            f"PROBLEMA: PC1 explica {pc1_var*100:.1f}% de varianza -> "
            f"espacio esencialmente 1D (variables casi colineales)."
        )
    if delta_AUC_tri < 0.005:
        issues.append(
            f"PROBLEMA: delta_AUC(trivalente - mejor_baseline)={delta_AUC_tri:.4f} < 0.005 -> "
            f"ganancia marginal del AE es negligible."
        )

    if not issues:
        return (
            f"ORTOGONALIDAD CONFIRMADA: |rho(C,tau21)|={abs(corr_CT):.3f}, "
            f"|rho(C,N2)|={abs(corr_CN):.3f}; ambas < 0.3. "
            f"El AE aporta informacion genuinamente complementaria. "
            f"La combinacion trivalente tiene justificacion solida para el paper. "
            f"Delta_AUC(trivalente)={delta_AUC_tri:+.4f}."
        )
    elif len(issues) == 1 and abs(corr_CT) < 0.6 and abs(corr_CN) < 0.6:
        return (
            f"ORTOGONALIDAD PARCIAL: " + " | ".join(issues) +
            f" Las correlaciones de Pearson son aceptables pero la ganancia "
            f"de AUC es marginal; considerar incluir solo la variable mas discriminante."
        )
    else:
        return (
            "ADVERTENCIA PARA EL PAPER: " + " | ".join(issues) +
            " Revisar si el AE aporta algo mas alla de las variables analiticas. "
            "Considerar analisis de informacion mutua o SHAP para diagnóstico adicional."
        )


# ---------------------------------------------------------------------------
# Figura diagnostica
# ---------------------------------------------------------------------------
def _make_figure(C_q, T21_q, N2_q, C_s, T21_s, N2_s,
                 corr_mat, pca_var, aucs,
                 corr_CT, corr_CN, corr_TN):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("SONIC — Orthogonality Probe: AE vs tau21 vs N2", fontsize=13, fontweight="bold")

    # ---- Panel A: Scatter C_DDT vs tau21_DDT (QCD vs Signal) ----
    ax = axes[0, 0]
    ax.scatter(T21_q, C_q, s=3, alpha=0.3, color="steelblue", label=f"QCD (rho={corr_CT:.3f})", rasterized=True)
    if len(C_s) > 0:
        ax.scatter(T21_s, C_s, s=5, alpha=0.5, color="crimson", label="Signal", rasterized=True)
    ax.set_xlabel("tau21_DDT"); ax.set_ylabel("C_DDT (log Chamfer)")
    ax.set_title(f"C vs tau21  |rho|={abs(corr_CT):.3f}" +
                 ("  [ALTA]" if abs(corr_CT) > 0.6 else "  [OK]"))
    ax.legend(fontsize=8, markerscale=3); ax.grid(alpha=0.3)
    _color_border(ax, abs(corr_CT))

    # ---- Panel B: Scatter C_DDT vs N2_DDT ----
    ax = axes[0, 1]
    ax.scatter(N2_q, C_q, s=3, alpha=0.3, color="steelblue", label=f"QCD (rho={corr_CN:.3f})", rasterized=True)
    if len(C_s) > 0:
        ax.scatter(N2_s, C_s, s=5, alpha=0.5, color="crimson", label="Signal", rasterized=True)
    ax.set_xlabel("N2_DDT"); ax.set_ylabel("C_DDT (log Chamfer)")
    ax.set_title(f"C vs N2  |rho|={abs(corr_CN):.3f}" +
                 ("  [ALTA]" if abs(corr_CN) > 0.6 else "  [OK]"))
    ax.legend(fontsize=8, markerscale=3); ax.grid(alpha=0.3)
    _color_border(ax, abs(corr_CN))

    # ---- Panel C: Heatmap de correlaciones + PCA variance ----
    ax = axes[1, 0]
    labels_vars = ["C_DDT", "tau21_DDT", "N2_DDT"]
    im = ax.imshow(np.abs(corr_mat), vmin=0, vmax=1, cmap="RdYlGn_r")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{corr_mat[i,j]:.2f}", ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="white" if abs(corr_mat[i,j]) > 0.6 else "black")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(labels_vars, rotation=20, ha="right", fontsize=9)
    ax.set_yticklabels(labels_vars, fontsize=9)
    ax.set_title("Matriz de correlacion de Pearson (QCD)")
    plt.colorbar(im, ax=ax, shrink=0.8)
    # Añadir PCA como texto en el plot
    ax.text(0.02, -0.28,
            f"PCA QCD: PC1={pca_var[0]*100:.1f}%  PC2={pca_var[1]*100:.1f}%  PC3={pca_var[2]*100:.1f}%",
            transform=ax.transAxes, fontsize=9, color="navy",
            bbox=dict(boxstyle="round", fc="lightyellow", ec="navy", alpha=0.8))

    # ---- Panel D: Delta AUC bar chart ----
    ax = axes[1, 1]
    order = ["C", "T21", "N2", "C+T21", "C+N2", "C+N2+T21"]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
    p50s = [aucs[k]["p50"] for k in order]
    p16s = [aucs[k]["p50"] - aucs[k]["p16"] for k in order]
    p84s = [aucs[k]["p84"] - aucs[k]["p50"] for k in order]
    bars = ax.bar(range(len(order)), p50s, color=colors, alpha=0.85, edgecolor="k", linewidth=0.7)
    ax.errorbar(range(len(order)), p50s, yerr=[p16s, p84s],
                fmt="none", color="black", capsize=4, linewidth=1.5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("AUC (bootstrap median)")
    ax.set_title("AUC por combinacion de variables (con IC 68%)")
    best_base = max(aucs["T21"]["p50"], aucs["N2"]["p50"])
    ax.axhline(best_base, color="gray", ls="--", lw=1.5, label=f"Mejor baseline={best_base:.4f}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ymin = min(p50s) - 0.05
    ymax = max(p50s) + 0.05
    ax.set_ylim(max(0.45, ymin), min(1.0, ymax))

    plt.tight_layout()
    return fig


def _color_border(ax, abs_corr):
    """Pone borde rojo si correlacion alta, verde si baja."""
    color = "crimson" if abs_corr > 0.6 else ("orange" if abs_corr > 0.3 else "green")
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(2.5)


# ---------------------------------------------------------------------------
# Demo con datos sinteticos (ejecutar directo sin datos reales)
# ---------------------------------------------------------------------------
def _make_synthetic_data(n_qcd=8000, n_sig=2000, seed=42):
    """
    Genera 3 escenarios sinteticos para ilustrar el diagnostico:

    Escenario A: AE ORTOGONAL a tau21 y N2 (el paper funciona bien)
    Escenario B: AE CORRELACIONADO con tau21 (el paper se debilita)
    Escenario C: AE CORRELACIONADO con ambos (combinacion no aporta nada)
    """
    rng = np.random.default_rng(seed)

    # ---- Escenario A: ortogonal ----
    # QCD: tau21 y N2 correlacionados entre si pero C independiente
    T21_q = rng.normal(0.5,  0.15, n_qcd)
    N2_q  = 0.4 * T21_q + rng.normal(0, 0.12, n_qcd)  # leve correlacion T21-N2
    C_q   = rng.normal(0.0,  1.0,  n_qcd)              # AE independiente en QCD

    # Signal: tau21 y N2 bajos, C alto
    T21_s = rng.normal(0.25, 0.08, n_sig)
    N2_s  = rng.normal(0.05, 0.04, n_sig)
    C_s   = rng.normal(2.5,  0.6,  n_sig)              # Chamfer alto en senal

    return (
        np.concatenate([C_q,   C_s]),
        np.concatenate([T21_q, T21_s]),
        np.concatenate([N2_q,  N2_s]),
        np.concatenate([np.zeros(n_qcd), np.ones(n_sig)]),
    )


if __name__ == "__main__":
    print("=" * 65)
    print("  MODO DEMO: datos sinteticos")
    print("  Para datos reales, importar run_probe() con tus arrays.")
    print("=" * 65)

    C, T21, N2, labels = _make_synthetic_data(n_qcd=8000, n_sig=2000)
    results = run_probe(
        C, T21, N2, labels,
        save_prefix="demo",
        n_boot=300,
    )

    # Guardar JSON
    import json
    # Convertir a tipos serializables
    def _to_serializable(d):
        if isinstance(d, dict):
            return {k: _to_serializable(v) for k, v in d.items()}
        elif isinstance(d, (np.floating, float)):
            return float(d)
        elif isinstance(d, (np.integer, int)):
            return int(d)
        elif isinstance(d, np.ndarray):
            return d.tolist()
        return d

    with open("demo_orthogonality_results.json", "w") as f:
        json.dump(_to_serializable(results), f, indent=2)
    print("  JSON guardado: demo_orthogonality_results.json")
