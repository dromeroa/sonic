"""
sonic.utils.compute
===================
Numerically intensive helper functions: CSV column parsing, jet‑axis
reconstruction, and the N₂ energy‑correlation function.
"""

import itertools
import numpy as np


# ---------------------------------------------------------------------------
# Fast semicolon‑delimited column parser
# ---------------------------------------------------------------------------
def fast_parse(df_col, max_p: int) -> np.ndarray:
    """Parse a Pandas Series of semicolon‑separated floats into a (N, max_p) array."""
    n_rows = len(df_col)
    out = np.zeros((n_rows, max_p), dtype=np.float32)
    for idx, val in enumerate(df_col):
        if not isinstance(val, str):
            continue
        if val.startswith('"') and val.endswith('"'):
            val = val[1:-1]
        parts = val.split(";")
        n = min(len(parts), max_p)
        if n > 0:
            try:
                out[idx, :n] = [float(x) for x in parts[:n]]
            except ValueError:
                pass
    return out


# ---------------------------------------------------------------------------
# Jet axis from constituent pT / η / φ
# ---------------------------------------------------------------------------
def jet_axis(pt: np.ndarray, eta: np.ndarray, phi: np.ndarray):
    """Return (jet_eta, jet_phi) arrays each of shape (N, 1)."""
    s = np.sum(pt, axis=1, keepdims=True) + 1e-8
    we = np.sum(pt * eta, axis=1, keepdims=True) / s
    wx = np.sum(pt * np.cos(phi), axis=1, keepdims=True)
    wy = np.sum(pt * np.sin(phi), axis=1, keepdims=True)
    return we, np.arctan2(wy, wx)


# ---------------------------------------------------------------------------
# N₂ energy‑correlation function (with combination‑index cache)
# ---------------------------------------------------------------------------
_COMBO_CACHE: dict = {}


def _get_combo_indices(n_c: int):
    """Memoised (i, j, k) triplet indices for n_c constituents."""
    if n_c not in _COMBO_CACHE:
        combo = np.array(list(itertools.combinations(range(n_c), 3)))
        _COMBO_CACHE[n_c] = (combo[:, 0], combo[:, 1], combo[:, 2])
    return _COMBO_CACHE[n_c]


def compute_n2(
    pt: np.ndarray,
    eta: np.ndarray,
    phi: np.ndarray,
    masks: np.ndarray,
    beta: float = 1.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """Compute the N₂^(β) ECF ratio for each jet."""
    n_jets = pt.shape[0]
    n2 = np.zeros(n_jets, dtype=np.float32)
    for j in range(n_jets):
        m = masks[j] > 0
        n_c = int(m.sum())
        if n_c < 3:
            continue

        pt_j = pt[j, m].astype(np.float64)
        eta_j = eta[j, m].astype(np.float64)
        phi_j = phi[j, m].astype(np.float64)
        z = pt_j / (pt_j.sum() + eps)

        deta = eta_j[:, None] - eta_j[None, :]
        dphi = (phi_j[:, None] - phi_j[None, :] + np.pi) % (2 * np.pi) - np.pi
        theta = np.sqrt(deta ** 2 + dphi ** 2)

        iu, ju = np.triu_indices(n_c, k=1)
        e2 = np.sum(z[iu] * z[ju] * theta[iu, ju] ** beta)
        if e2 <= eps:
            continue

        ci, cj, ck = _get_combo_indices(n_c)
        th_ij = theta[ci, cj]
        th_ik = theta[ci, ck]
        th_jk = theta[cj, ck]
        min_prod = np.minimum(
            np.minimum(th_ij * th_ik, th_ij * th_jk), th_ik * th_jk
        )
        e3 = np.sum(z[ci] * z[cj] * z[ck] * (min_prod ** beta))
        n2[j] = e3 / (e2 ** 2 + eps)

    return n2.astype(np.float32)
