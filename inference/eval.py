"""
sonic.inference.eval
====================
Score collection in inference mode, sign-convention fixing, and trivariate
grid search for the linear combination:

    S = alpha * C_DDT  +  beta * N2_DDT  -  gamma * tau21_DDT

Cambios respecto a v4 original
--------------------------------
1. _grid_eval soporta modo trivariate 3D limpiamente.
2. grid_search_hybrid activa el grid 3D cuando N2_ddt != None.
3. cli.py debe pasar N2_gs_ddt al llamar a grid_search_hybrid.
"""

import itertools

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from sonic.training.losses import chamfer


# ---------------------------------------------------------------------------
# Raw score collection
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_raw(model, loader, device):
    """
    Return arrays: Chamfer, tau21, rho, label, SDMass, log_pT, N2
    """
    model.eval()
    C, T21, R, L, M, LP, N2 = [], [], [], [], [], [], []

    for feat, mask, phys in loader:
        feat = feat.to(device)
        mask = mask.to(device)
        recon, _ = model(feat)
        c = chamfer(feat, recon, mask, mask, "none")

        C.extend(c.cpu().numpy())
        T21.extend(phys[:, 3].numpy())
        R.extend(phys[:, 6].numpy())
        L.extend(phys[:, 5].numpy())
        M.extend(phys[:, 4].numpy())
        LP.extend(phys[:, 7].numpy())
        N2.extend(phys[:, 8].numpy())

    return (
        np.array(C),
        np.array(T21),
        np.array(R),
        np.array(L),
        np.array(M),
        np.array(LP),
        np.array(N2),
    )


# ---------------------------------------------------------------------------
# Sign convention
# ---------------------------------------------------------------------------
def fix_dir_on_validation(s_val, labels_val):
    auc_val = roc_auc_score(labels_val, s_val)
    sign = -1.0 if auc_val < 0.5 else 1.0
    auc_val_oriented = auc_val if sign > 0 else 1 - auc_val
    return sign, auc_val_oriented


def fix_dir(s, labels):
    auc = roc_auc_score(labels, s)
    return (-s, 1 - auc) if auc < 0.5 else (s, auc)


# ---------------------------------------------------------------------------
# Core grid evaluator
# ---------------------------------------------------------------------------
def _grid_eval(C_ddt, T21_ddt, labels, alphas, gammas,
               N2_ddt=None, betas=None):
    """
    Evalua todas las combinaciones (alpha, gamma) o (alpha, beta, gamma).

    Trivariate (N2_ddt y betas provistos):
        S = alpha * C_DDT  +  beta * N2_DDT  -  gamma * tau21_DDT

    Bivariate legacy (N2_ddt=None):
        S = alpha * C_DDT  -  gamma * tau21_DDT

    Todos los coeficientes son >= 0. La ley de signos para que
    scores altos correspondan a senal:
      - Chamfer alto  -> jet anomalo -> alpha > 0: suma
      - N2 DDT bajo   -> jet 2-prong -> beta  puede ser > 0 o 0
        (si el grid lo halla informativo, pondra beta > 0 y la
         combinacion funcionara con el flip de signo global)
      - tau21 bajo    -> jet 2-prong -> gamma > 0: resta tau21
        (tau21 bajo = senal, restarlo sube el score)
    """
    best_auc, best_params = 0.0, {}
    trivariate = N2_ddt is not None and betas is not None

    if trivariate:
        grid = itertools.product(alphas, betas, gammas)
    else:
        grid = ((a, None, g) for a, g in itertools.product(alphas, gammas))

    for alpha, beta, gamma in grid:
        if trivariate:
            s = alpha * C_ddt + beta * N2_ddt - gamma * T21_ddt
        else:
            s = alpha * C_ddt - gamma * T21_ddt
        try:
            auc = roc_auc_score(labels, s)
            if auc < 0.5:
                auc = 1 - auc
            if auc > best_auc:
                best_auc = auc
                best_params = {
                    "alpha": float(alpha),
                    "gamma": float(gamma),
                }
                if trivariate:
                    best_params["beta"] = float(beta)
        except Exception:
            continue
    return best_params, best_auc


# ---------------------------------------------------------------------------
# Coarse -> fine grid search
# ---------------------------------------------------------------------------
def grid_search_hybrid(C_ddt, T21_ddt, labels, N2_ddt=None):
    """
    Grid search coarse->fine para:

        S = alpha * C_DDT  +  beta * N2_DDT  -  gamma * tau21_DDT

    (trivariate si N2_ddt != None, bivariate legacy si N2_ddt=None)

    Grilla de busqueda:
        alpha in linspace(0.05, 5.0, 20)  -- AE siempre participa
        beta  in linspace(0.0,  5.0, 11)  -- N2 puede ser 0
        gamma in linspace(0.0,  5.0, 21)  -- tau21 puede ser 0
        -> 20 x 11 x 21 = 4620 pts coarse, luego zoom x21^3 fine

    Retorna
    -------
    best_params : dict {"alpha": float, "beta": float, "gamma": float}
    auc_best    : float AUC en validacion
    """
    trivariate = N2_ddt is not None

    # ---- Coarse grid -------------------------------------------------------
    alphas_c = np.linspace(0.05, 5.0, 20)
    gammas_c = np.linspace(0.0,  5.0, 21)
    betas_c  = np.linspace(0.0,  5.0, 11) if trivariate else None

    params_c, auc_c = _grid_eval(
        C_ddt, T21_ddt, labels, alphas_c, gammas_c, N2_ddt, betas_c
    )
    b_str = f"  beta={params_c.get('beta', 0):.2f}" if trivariate else ""
    print(
        f"[+] Coarse grid -> AUC={auc_c:.4f}  "
        f"alpha={params_c.get('alpha', 0):.2f}{b_str}"
        f"  gamma={params_c.get('gamma', 0):.2f}"
    )
    if not params_c:
        return params_c, auc_c

    # ---- Fine grid: zoom +/-1 paso coarse ----------------------------------
    a0, g0 = params_c["alpha"], params_c["gamma"]
    da = alphas_c[1] - alphas_c[0]
    dg = gammas_c[1] - gammas_c[0]
    alphas_f = np.linspace(max(0.01, a0 - da), a0 + da, 21)
    gammas_f = np.linspace(max(0.0,  g0 - dg), g0 + dg, 21)

    if trivariate:
        b0 = params_c.get("beta", 0.0)
        db = betas_c[1] - betas_c[0]
        betas_f = np.linspace(max(0.0, b0 - db), b0 + db, 21)
    else:
        betas_f = None

    params_f, auc_f = _grid_eval(
        C_ddt, T21_ddt, labels, alphas_f, gammas_f, N2_ddt, betas_f
    )

    if auc_f >= auc_c and params_f:
        b_str = f"  beta={params_f.get('beta', 0):.3f}" if trivariate else ""
        print(
            f"[+] Fine grid   -> AUC={auc_f:.4f}  "
            f"alpha={params_f.get('alpha', 0):.3f}{b_str}"
            f"  gamma={params_f.get('gamma', 0):.3f}"
        )
        return params_f, auc_f
    return params_c, auc_c
