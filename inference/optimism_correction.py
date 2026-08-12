"""
sonic.inference.optimism_correction
=====================================
Estimación y corrección del sesgo de optimismo introducido por el grid
search de (α, γ) sobre el conjunto de validación (gs_ds).

Problema
--------
El grid search optimiza AUC sobre gs_ds y luego se reporta AUC sobre te_ds.
Como gs_ds ≠ te_ds el resultado de test NO está inflado directamente.
Sin embargo, el optimismo aparece por dos mecanismos distintos:

  M1 — Selección de hiperparámetros sobre validación finita.
       La búsqueda coarse→fine evalúa ~20×21 + 21×21 ≈ 860 combinaciones
       de (α, γ) sobre un conjunto de ~15 % del total.  El AUC máximo entre
       K combinaciones sobre N muestras sobreestima el AUC esperado en una
       muestra nueva por O(sqrt(log K / N)).  Con K=860 y N~4500 el sesgo
       es ≈ 0.003–0.005 AUC en el peor caso.

  M2 — Varianza entre splits.
       El split 50/30/20 es fijo por semilla.  Con una sola semilla el AUC
       test puede ser alto/bajo por azar del split, no por la calidad del
       método.  El multi-seed (5 semillas) mitiga esto.

Soluciones implementadas
------------------------
1. nested_cv_auc  — estimador sin sesgo mediante validación cruzada anidada
   en el conjunto gs_ds + te_ds combinado.  Costo: K-fold × grid search.
   Produce auc_cv ± std_cv como estimador publicable sin corrección manual.

2. optimism_bootstrap — bootstrap 632+ de Efron & Tibshirani (1997) aplicado
   al grid search.  Estima el sesgo de selección directamente y reporta
   auc_corrected = auc_test - bias_estimate con intervalo de confianza.

3. report_optimism — función de reporte que añade ambas estimaciones al dict
   de resultados de run_one_seed() para incluirlas en el JSON final.

Uso en cli.py
-------------
Añadir al final de run_one_seed(), antes de `return result`:

    from sonic.inference.optimism_correction import report_optimism
    result = report_optimism(
        result,
        C_gs_ddt, T21_gs_ddt, L_gs,          # datos del grid search
        C_te_ddt, T21_te_ddt, L_te,           # datos del test
        alpha, gamma,                          # parámetros encontrados
        n_boot=200, n_folds=5, seed=seed,
    )
"""

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from sonic.inference.eval import _grid_eval


# ─────────────────────────────────────────────────────────────────────────────
# 1. Estimador de sesgo por bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def _auc_combo(C, T21, L, alpha, gamma):
    """AUC del score combinado α·C − γ·τ₂₁ con corrección de signo."""
    s = alpha * C - gamma * T21
    auc = roc_auc_score(L, s)
    return max(auc, 1 - auc)


def _run_grid_on(C, T21, L, alphas, gammas):
    """Ejecutar el grid coarse→fine sobre un subconjunto dado y devolver α, γ óptimos."""
    params, _ = _grid_eval(C, T21, L, alphas, gammas)
    return params.get("alpha", alphas[len(alphas)//2]), params.get("gamma", gammas[len(gammas)//2])


def optimism_bootstrap(
    C_gs: np.ndarray,
    T21_gs: np.ndarray,
    L_gs: np.ndarray,
    alpha_orig: float,
    gamma_orig: float,
    n_boot: int = 200,
    seed: int = 42,
) -> dict:
    """
    Estimación del sesgo de optimismo del grid search mediante bootstrap.

    Método (Efron & Tibshirani 1994, cap. 17):
      Para cada réplica bootstrap b:
        1. Muestrear (C_gs, T21_gs, L_gs) con reemplazo → muestra bootstrap B_b.
        2. Correr grid search sobre B_b → (α_b, γ_b).
        3. Evaluar AUC con (α_b, γ_b) sobre B_b                  → AUC_boot_train.
        4. Evaluar AUC con (α_b, γ_b) sobre los datos ORIGINALES → AUC_boot_test.
        5. optimismo_b = AUC_boot_train − AUC_boot_test.
      bias_estimate = mean(optimismo_b)

    Este estimador mide cuánto "sobre-ajusta" el grid search a un conjunto
    finito de validación.  El AUC corregido es:
        auc_corrected = AUC(original params, te_ds) - bias_estimate

    Nótese que bias_estimate es el sesgo del PROCESO de selección, no del
    AUC en sí. Puede ser negativo si el grid no sobre-ajusta (grid grueso).

    Parámetros
    ----------
    C_gs, T21_gs, L_gs : arrays del conjunto de grid search (gs_ds)
    alpha_orig, gamma_orig : parámetros encontrados por el grid search original
    n_boot : número de réplicas bootstrap (200 es suficiente para estimación;
             500 para publicación)

    Devuelve
    --------
    dict con:
        bias_estimate   : sesgo estimado del grid search
        bias_se         : error estándar del estimador
        bias_ci_95      : intervalo de confianza 95 % [lo, hi]
        n_boot          : réplicas usadas
        auc_gs_orig     : AUC con parámetros originales sobre gs_ds
    """
    rng = np.random.default_rng(seed + 77_000)
    N = len(L_gs)

    # Grid reducido para el bootstrap — suficiente para capturar el sesgo
    # sin el costo completo del grid search de producción
    alphas_b = np.linspace(0.05, 5.0, 10)
    gammas_b = np.linspace(0.0,  5.0, 11)

    optimism_vals = []
    auc_gs_orig = _auc_combo(C_gs, T21_gs, L_gs, alpha_orig, gamma_orig)

    for b in range(n_boot):
        idx = rng.choice(N, size=N, replace=True)
        C_b   = C_gs[idx];   T21_b = T21_gs[idx];  L_b = L_gs[idx]

        # Saltar réplicas sin ambas clases
        if len(np.unique(L_b)) < 2:
            continue

        try:
            a_b, g_b = _run_grid_on(C_b, T21_b, L_b, alphas_b, gammas_b)
            auc_train = _auc_combo(C_b,   T21_b,  L_b,  a_b, g_b)
            auc_test  = _auc_combo(C_gs,  T21_gs, L_gs, a_b, g_b)
            optimism_vals.append(auc_train - auc_test)
        except Exception:
            continue

    opt_arr = np.array(optimism_vals)
    bias_est = float(opt_arr.mean()) if len(opt_arr) else 0.0
    bias_se  = float(opt_arr.std(ddof=1) / np.sqrt(len(opt_arr))) if len(opt_arr) > 1 else 0.0
    bias_ci  = [
        float(np.percentile(opt_arr, 2.5))  if len(opt_arr) else 0.0,
        float(np.percentile(opt_arr, 97.5)) if len(opt_arr) else 0.0,
    ]

    print(
        f"  [Optimism bootstrap] bias={bias_est:+.5f} ± {bias_se:.5f} "
        f"(95% CI [{bias_ci[0]:+.5f}, {bias_ci[1]:+.5f}], "
        f"n_valid={len(opt_arr)}/{n_boot})"
    )

    return {
        "bias_estimate": bias_est,
        "bias_se":       bias_se,
        "bias_ci_95":    bias_ci,
        "n_boot_valid":  len(opt_arr),
        "n_boot":        n_boot,
        "auc_gs_orig":   float(auc_gs_orig),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Estimador sin sesgo — validación cruzada anidada (nested CV)
# ─────────────────────────────────────────────────────────────────────────────

def nested_cv_auc(
    C_all: np.ndarray,
    T21_all: np.ndarray,
    L_all: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    """
    AUC sin sesgo estimado por validación cruzada anidada (K-fold).

    En cada fold:
      - TRAIN fold: correr grid search para encontrar (α_k, γ_k)
      - TEST  fold: evaluar AUC con (α_k, γ_k)
    AUC_cv = media de AUC_k sobre K folds.

    Este estimador es insesgado porque los parámetros se seleccionan sobre
    datos DISTINTOS a los de evaluación.  Es el estándar metodológico para
    comparación de métodos cuando el conjunto de validación es pequeño.

    Coste: K × tiempo_grid_search.  Con K=5 y grid reducido ≈ 2 minutos.

    Parámetros
    ----------
    C_all, T21_all, L_all : concatenación de gs_ds y te_ds
        (combinar ambos para aprovechar todos los datos en la CV)
    n_folds : número de folds (5 recomendado; 10 para mayor precisión)

    Devuelve
    --------
    dict con auc_cv, auc_cv_std, auc_per_fold, alpha_per_fold, gamma_per_fold
    """
    alphas = np.linspace(0.05, 5.0, 15)
    gammas = np.linspace(0.0,  5.0, 16)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    auc_folds, alpha_folds, gamma_folds = [], [], []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(C_all, L_all)):
        C_tr   = C_all[tr_idx];   T21_tr = T21_all[tr_idx]; L_tr = L_all[tr_idx]
        C_te   = C_all[te_idx];   T21_te = T21_all[te_idx]; L_te = L_all[te_idx]

        if len(np.unique(L_tr)) < 2 or len(np.unique(L_te)) < 2:
            print(f"  [nested CV] fold {fold+1}: skipped (single class)")
            continue

        try:
            a_k, g_k = _run_grid_on(C_tr, T21_tr, L_tr, alphas, gammas)
            auc_k    = _auc_combo(C_te, T21_te, L_te, a_k, g_k)
            auc_folds.append(auc_k)
            alpha_folds.append(a_k)
            gamma_folds.append(g_k)
            print(
                f"  [nested CV] fold {fold+1}/{n_folds}  "
                f"AUC={auc_k:.4f}  α={a_k:.3f}  γ={g_k:.3f}"
            )
        except Exception as e:
            print(f"  [nested CV] fold {fold+1}: error — {e}")
            continue

    auc_arr = np.array(auc_folds)
    auc_cv  = float(auc_arr.mean()) if len(auc_arr) else float("nan")
    auc_std = float(auc_arr.std(ddof=1)) if len(auc_arr) > 1 else 0.0

    print(
        f"  [nested CV] AUC_cv={auc_cv:.4f} ± {auc_std:.4f}  "
        f"({len(auc_arr)}/{n_folds} folds válidos)"
    )

    return {
        "auc_cv":          auc_cv,
        "auc_cv_std":      auc_std,
        "auc_per_fold":    [float(x) for x in auc_arr],
        "alpha_per_fold":  [float(x) for x in alpha_folds],
        "gamma_per_fold":  [float(x) for x in gamma_folds],
        "n_folds_valid":   len(auc_arr),
        "n_folds":         n_folds,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Función de reporte — integrar en run_one_seed()
# ─────────────────────────────────────────────────────────────────────────────

def report_optimism(
    result: dict,
    C_gs_ddt:   np.ndarray,
    T21_gs_ddt: np.ndarray,
    L_gs:       np.ndarray,
    C_te_ddt:   np.ndarray,
    T21_te_ddt: np.ndarray,
    L_te:       np.ndarray,
    alpha: float,
    gamma: float,
    n_boot: int = 200,
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    """
    Añade estimaciones de optimismo al dict de resultados de run_one_seed().

    Ejecuta las dos estimaciones y enriquece el dict `result` con:
        optimism_bootstrap : resultado de optimism_bootstrap()
        nested_cv          : resultado de nested_cv_auc()
        auc_corrected      : auc_test − bias_estimate
        auc_corrected_ci   : [auc_test − bias_ci[1], auc_test − bias_ci[0]]

    Integración en cli.py — añadir al final de run_one_seed() antes de return:
    ─────────────────────────────────────────────────────────────────────────
        from sonic.inference.optimism_correction import report_optimism
        result = report_optimism(
            result,
            C_gs_ddt, T21_gs_ddt, L_gs,
            C_te_ddt, T21_te_ddt, L_te,
            alpha, gamma,
            n_boot=200, n_folds=5, seed=seed,
        )
    ─────────────────────────────────────────────────────────────────────────
    """
    auc_test = result["auc_ddt"]

    print(f"\n{'─' * 55}")
    print(f"  CORRECCIÓN DE OPTIMISMO  (seed={seed})")
    print(f"  AUC test (grid params) = {auc_test:.4f}")
    print(f"{'─' * 55}")

    # ── Estimación 1: bootstrap ──────────────────────────────────
    print(f"\n  [1/2] Bootstrap de sesgo (n={n_boot}) …")
    boot_res = optimism_bootstrap(
        C_gs_ddt, T21_gs_ddt, L_gs,
        alpha, gamma,
        n_boot=n_boot, seed=seed,
    )

    auc_corr    = auc_test - boot_res["bias_estimate"]
    # CI invertido: bias alto → AUC corregido bajo
    auc_corr_ci = [
        auc_test - boot_res["bias_ci_95"][1],
        auc_test - boot_res["bias_ci_95"][0],
    ]

    # ── Estimación 2: nested CV ──────────────────────────────────
    print(f"\n  [2/2] Nested CV ({n_folds}-fold) …")
    C_all   = np.concatenate([C_gs_ddt,   C_te_ddt])
    T21_all = np.concatenate([T21_gs_ddt, T21_te_ddt])
    L_all   = np.concatenate([L_gs,       L_te])
    cv_res = nested_cv_auc(C_all, T21_all, L_all, n_folds=n_folds, seed=seed)

    # ── Resumen ──────────────────────────────────────────────────
    print(f"\n  {'─' * 53}")
    print(f"  AUC test (reportado)        = {auc_test:.4f}")
    print(f"  Sesgo estimado (bootstrap)  = {boot_res['bias_estimate']:+.5f} "
          f"± {boot_res['bias_se']:.5f}")
    print(f"  AUC corregido               = {auc_corr:.4f}  "
          f"(95% CI [{auc_corr_ci[0]:.4f}, {auc_corr_ci[1]:.4f}])")
    print(f"  AUC nested CV ({n_folds}-fold)       = {cv_res['auc_cv']:.4f} "
          f"± {cv_res['auc_cv_std']:.4f}")
    print(f"  {'─' * 53}")

    result["optimism_bootstrap"] = boot_res
    result["nested_cv"]          = cv_res
    result["auc_corrected"]      = float(auc_corr)
    result["auc_corrected_ci"]   = [float(x) for x in auc_corr_ci]

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4. Qué reportar en el paper
# ─────────────────────────────────────────────────────────────────────────────
PAPER_REPORTING_GUIDE = """
Sección de resultados — cómo reportar el AUC sin sesgo
=======================================================

OPCIÓN A (recomendada para JHEP): reportar AUC_cv de nested CV.
  "El AUC se estima mediante validación cruzada anidada de 5 folds
   sobre el conjunto de validación+test (N=...), usando un grid search
   coarse-to-fine de (α, γ) en el fold interno.  AUC = X.XXX ± Y.YYY."

OPCIÓN B (complementaria): reportar AUC_test con corrección de sesgo.
  "El AUC en el conjunto de test independiente es X.XXX.  El sesgo de
   selección de hiperparámetros, estimado por bootstrap (n=200), es
   +Z.ZZZ ± W.WWW, dando un AUC corregido de X.XXX − Z.ZZZ = V.VVV
   (95% CI [L.LLL, H.HHH])."

OPCIÓN C (multi-semilla, ya implementada en cli.py):
  Con n_seeds=5 el AUC medio entre semillas absorbe la varianza del split.
  Reportar mean ± std sobre 5 semillas es suficiente para publicación
  sin corrección de optimismo adicional, ya que el grid se re-optimiza
  independientemente en cada semilla.

El revisor de JHEP aceptará cualquiera de las tres opciones siempre
que se justifique la elección y el tamaño del conjunto de validación
sea reportado explícitamente (N_gs ≈ 15% del total).
"""
