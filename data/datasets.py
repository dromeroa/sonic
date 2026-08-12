"""
sonic.data.datasets
===================
PyTorch Dataset that reads CMS‑format CSV files, computes 6‑feature jet
representations and physics arrays, and caches the tensors to disk.
"""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from sonic.data.prep import PrepV12
from sonic.utils.compute import fast_parse, jet_axis, compute_n2


class JetDatasetV12(Dataset):
    """
    Columns stored in ``self.phys`` (axis 1):
        0: jmass   1: jpt    2: nc     3: tau21   4: sdm
        5: label   6: rho    7: log_jpt   8: n2
    """

    N_FEAT = 8

# ---- 8‑feature per-constituent representation ----
# feat[:,  0]: Δη normalizado (wrt jet axis)
# feat[:,  1]: Δφ normalizado (wrt jet axis)
# feat[:,  2]: log(1 + pt_frac × 100)   — pT relativo suavizado
# feat[:,  3]: pt_frac                   — fracción de pT
# feat[:,  4]: Δη wrt leading constituent
# feat[:,  5]: Δφ wrt leading constituent
# feat[:,  6]: ΔR wrt jet axis
# feat[:,  7]: log(pt_abs) normalizado   — escala energética absoluta


    def __init__(
        self,
        file_path: str,
        prep: PrepV12,
        max_p: int = 30,
        force: bool = False,
        max_rows: int | None = None,
    ):
        rows_suffix = f"_{max_rows}" if max_rows is not None else "_full"
        self.cache = file_path.replace(".csv", f"_cache_v12_sonic_n2{rows_suffix}.pt")
        if force and os.path.exists(self.cache):
            os.remove(self.cache)

        if os.path.exists(self.cache) and not force:
            self.feat, self.mask, self.phys = torch.load(self.cache, weights_only=False)
            if max_rows and len(self.feat) > max_rows:
                self.feat = self.feat[:max_rows]
                self.mask = self.mask[:max_rows]
                self.phys = self.phys[:max_rows]
            return

        # ---- read CSV ----
        cols_to_use = ["PF_Pt", "PF_Eta", "PF_Phi", "SDMass", "Tau1", "Tau2"]
        header_df = pd.read_csv(file_path, nrows=0)
        if "is_signal" in header_df.columns:
            cols_to_use.append("is_signal")
        df = pd.read_csv(file_path, usecols=cols_to_use, nrows=max_rows)

        pt = fast_parse(df["PF_Pt"], max_p)
        eta = fast_parse(df["PF_Eta"], max_p)
        phi = fast_parse(df["PF_Phi"], max_p)

        px, py = pt * np.cos(phi), pt * np.sin(phi)
        pz, e = pt * np.sinh(eta), pt * np.cosh(eta)
        jm2 = e.sum(1) ** 2 - px.sum(1) ** 2 - py.sum(1) ** 2 - pz.sum(1) ** 2
        jmass = np.sqrt(np.maximum(0, jm2))
        jpt = np.sqrt(px.sum(1) ** 2 + py.sum(1) ** 2)
        sdm = df["SDMass"].values.astype(np.float32)

        valid = (jpt > 250.0) & (sdm >= 40.0) & (sdm <= 200.0)
        pt, eta, phi = pt[valid], eta[valid], phi[valid]
        jmass, jpt, sdm = jmass[valid], jpt[valid], sdm[valid]
        df = df[valid]

        tau1 = df["Tau1"].values.astype(np.float32) + 1e-8
        tau2 = df["Tau2"].values.astype(np.float32)
        tau21 = tau2 / tau1

        if "is_signal" in df.columns:
            sig = df["is_signal"].values.astype(np.float32)
        elif "signal" in os.path.basename(file_path).lower():
            sig = np.ones(len(df), dtype=np.float32)
        else:
            sig = np.zeros(len(df), dtype=np.float32)

        nc = np.sum(pt > 0, 1).astype(np.float32)
        rho = np.log(np.clip(sdm ** 2 / (jpt ** 2 + 1e-6), 1e-6, None)).astype(np.float32)
        log_jpt = np.log(jpt + 1e-6).astype(np.float32)

        print(
            f"    [*] Computing N2 (β=1) for {len(pt)} jets from "
            f"'{os.path.basename(file_path)}' (cached once) …"
        )
        n2 = compute_n2(pt, eta, phi, pt > 0)

        phys_arr = np.stack(
            [jmass, jpt, nc, tau21, sdm, sig, rho, log_jpt, n2], axis=1
        ).astype(np.float32)

        # ---- 6‑feature representation ----
        je, jp = jet_axis(pt, eta, phi)
        deta = eta - je
        dphi = (phi - jp + np.pi) % (2 * np.pi) - np.pi
        masks = (pt > 0).astype(np.float32)

        pt_sum = (pt * masks).sum(axis=1, keepdims=True) + 1e-8
        pt_frac = pt * masks / pt_sum
        logpt_rel = np.log1p(pt_frac * 100) * masks

        lead_idx = np.argmax(pt * masks, axis=1)
        N = pt.shape[0]
        lead_eta = eta[np.arange(N), lead_idx]
        lead_phi = phi[np.arange(N), lead_idx]
        deta_lead = (eta - lead_eta[:, None]) * masks
        dphi_lead = ((phi - lead_phi[:, None] + np.pi) % (2 * np.pi) - np.pi) * masks

        dn = (deta - prep.feat_mean[0]) / (prep.feat_std[0] + 1e-8) * masks
        dp = (dphi - prep.feat_mean[1]) / (prep.feat_std[1] + 1e-8) * masks

        # Feature 6: delta_R wrt jet axis — señal directa de estructura 2-prong
        delta_r = np.sqrt(deta**2 + ((phi - jp + np.pi) % (2*np.pi) - np.pi)**2) * masks

        # Feature 7: log pT absoluto normalizado — escala energética del constituyente
        # FIX — data leakage corregido:
        # Antes: mean/std se calculaban sobre el archivo actual → estadísticos
        # distintos para QCD vs señal → canal 7 con escalas incomparables entre
        # train/val/test. Ahora se usan los valores fijados en PrepV12.fit()
        # sobre el split QCD de entrenamiento, igual que feat channels 0 y 1.
        prep.check_log_pt_abs()
	# [FIX-3] Constituentes de padding (pt=0, mask=0) deben contribuir con 0
        # al feature normalizado. La versión anterior calculaba log(0+1e-6)≈-13.8,
        # lo multiplicaba por 0 (correcto), pero la resta de log_pt_abs_mean ocurría
        # sobre ese valor antes de aplicar la máscara final, lo que hacía que el
        # z-score del padding dependiera numéricamente del valor de la media.
        # Solución: sustituir pt=0 por 1.0 antes del log → log(1.0)=0, z-score=
        # (0 - mean)/std que luego se zeroa con *masks. Más limpio y exacto.
        log_pt_abs      = np.log(np.where(masks > 0, pt + 1e-6, 1.0))
        log_pt_abs_norm = ((log_pt_abs - prep.log_pt_abs_mean) / prep.log_pt_abs_std) * masks

        feat_arr = np.stack(
            [dn, dp, logpt_rel, pt_frac, deta_lead, dphi_lead,
             delta_r, log_pt_abs_norm],
            axis=2
        ).astype(np.float32)

        self.feat = torch.from_numpy(feat_arr)
        self.mask = torch.from_numpy(masks)
        self.phys = torch.from_numpy(phys_arr)

        # [FIX-2] Shuffle con semilla fija antes de guardar en cache.
        # Sin esto, cargar desde cache con max_rows<len(cache) devuelve los
        # primeros N jets en orden de archivo (sesgado si el CSV está ordenado
        # por pT, evento, etc.). Con shuffle, [:max_rows] es siempre una
        # muestra aleatoria reproducible del dataset completo.
        perm = torch.randperm(len(self.feat), generator=torch.Generator().manual_seed(42))
        self.feat = self.feat[perm]
        self.mask = self.mask[perm]
        self.phys = self.phys[perm]
        torch.save((self.feat, self.mask, self.phys), self.cache)


    def __len__(self):
        return len(self.feat)

    def __getitem__(self, idx):
        return self.feat[idx], self.mask[idx], self.phys[idx]
