"""
sonic.training.losses
=====================
Reconstruction losses for the jet autoencoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def chamfer(p1, p2, m1, m2, reduction: str = "mean"):
    """
    Chamfer distance on the spatial (Δη, Δφ) channels, masked by particle presence.
    Returns per‑jet distances when ``reduction='none'``.
    """
    d = torch.cdist(p1[:, :, :2], p2[:, :, :2], p=2) ** 2
    d = d + (1 - m1.unsqueeze(2)) * 1e8 + (1 - m2.unsqueeze(1)) * 1e8

    pj = (d.min(2)[0] * m1).sum(1) / (m1.sum(1) + 1e-8) \
       + (d.min(1)[0] * m2).sum(1) / (m2.sum(1) + 1e-8)

    return pj if reduction == "none" else pj.mean()




"""Diagnóstico del problema original
----------------------------------
La n_eff_loss actual (línea 23 del original) tiene tres defectos
estructurales que limitan su utilidad como señal de supervisión:

  P1 — COLAPSO ESCALAR
       Compara un escalar (entropía de Shannon H) por jet, no la
       distribución completa de pt_frac.  Dos distribuciones muy
       distintas pueden tener el mismo H:
           p_f = [0.91, 0.09, 0, ...]   →  H ≈ 0.30
           p_r = [0.50, 0.50, 0, ...]   →  H ≈ 0.69
       Solo cuando los jets son perfectamente "flat" o perfectamente
       "peaked" el MSE de entropías da señal útil.  En la práctica
       el autoencoder puede satisfacer la loss con modos colapso
       que reproducen el mismo H con distribuciones incorrectas.

  P2 — GRADIENTE CIEGO A PERMUTACIONES
       La pérdida es invariante a cualquier permutación de
       constituyentes: si el AE mueve el pT del constituyente 1 al
       constituyente 7, la entropía total no cambia. El gradiente
       resultante no enseña al decoder dónde colocar el pT de cada
       constituyente, solo cuánto pT total debe haber.

  P3 — DOBLE RE-NORMALIZACIÓN INNECESARIA
       feat[:,:,3] ya es pt_frac = pt_i / Σ pt_j (construido en
       datasets.py línea 104).  La función divide de nuevo por
       sf = pt_f.sum(1) ≈ 1, introduciendo ruido numérico y
       gradientes espurios cuando la suma difiere de 1 por errores
       de máscara (constituyentes padding con pt_frac=0 que no
       suman exactamente a 1).

Las tres variantes propuestas a continuación atacan estos problemas
en orden creciente de impacto y de costo computacional.

Uso recomendado
---------------
    # Reemplazar en train.py la línea:
    #   neff = n_eff_loss(feat[is_qcd], recon[is_qcd], mask[is_qcd])
    # por cualquiera de:
    #   neff = n_eff_loss_kl(feat[is_qcd], recon[is_qcd], mask[is_qcd])
    #   neff = n_eff_loss_sorted(feat[is_qcd], recon[is_qcd], mask[is_qcd])
    #   neff = n_eff_loss_emd(feat[is_qcd], recon[is_qcd], mask[is_qcd])
"""


# ============================================================
# 0b. Original — conservada exactamente para referencia / ablación
# ============================================================

def n_eff_loss(feat, recon, mask):
    """
    ORIGINAL — MSE entre la entropía de Shannon de la distribución de
    pt_frac del input y la reconstrucción.
    Conservada sin cambios para comparación en ablación.
    """
    pt_f = feat[:, :, 3]
    pt_r = recon[:, :, 3] * mask
    sf = pt_f.sum(1, keepdim=True) + 1e-8
    sr = pt_r.sum(1, keepdim=True) + 1e-8
    p_f = pt_f / sf
    p_r = pt_r / sr
    ent_f = -(p_f * torch.log(p_f + 1e-8) * mask).sum(1)
    ent_r = -(p_r * torch.log(p_r + 1e-8) * mask).sum(1)
    return nn.functional.mse_loss(ent_r, ent_f)


# ============================================================
# 1.  NIVEL 1 — KL divergence  (bajo riesgo, reemplazo directo)
# ============================================================

def n_eff_loss_kl(feat, recon, mask, eps: float = 1e-8):
    """
    Reemplaza el MSE de entropías por KL(p_f ‖ p_r), que compara
    la FORMA COMPLETA de la distribución de pt_frac, no un escalar.

    Correcciones incluidas respecto al original:
      - Sin doble re-normalización: feat[:,3] ya es pt_frac ≈ Σ=1,
        solo se normaliza recon[:, 3] (que sale de Softplus sin
        garantía de suma unitaria).
      - Penalización por constituyente, no solo por jet.
      - Numericamente estable: clamp antes del log.

    Comportamiento:
      KL(p_f ‖ p_r) = Σ_i  p_f_i · log(p_f_i / p_r_i)
      → 0 cuando p_r = p_f exactamente (buena reconstrucción).
      → crece cuando p_r pone pT en constituyentes erróneos.

    Riesgo de sculpting:
      Bajo. pt_frac está correlacionado con SDMass vía pt_sum, pero
      la señal de gradiente que produce KL es idéntica en naturaleza
      a la de la entropía MSE — solo más informativa.

    Costo computacional:
      O(N · P) — igual que el original.
    """
    # Input: feat[:,3] ya es pt_frac normalizado (sum ≈ 1 por jet)
    p_f = feat[:, :, 3].clamp(min=eps)           # (B, P)

    # Recon: canal 3 sale de out_energy → Softplus; necesita normalización
    pt_r_raw = recon[:, :, 3] * mask              # (B, P) — zeros en padding
    p_r = pt_r_raw / (pt_r_raw.sum(1, keepdim=True) + eps)
    p_r = p_r.clamp(min=eps)

    # KL divergence por constituyente, promediada por jet activo
    kl_per_particle = p_f * (torch.log(p_f) - torch.log(p_r))  # (B, P)
    kl_per_particle = kl_per_particle * mask                     # zero padding

    n_active = mask.sum(1).clamp(min=1.0)                       # (B,)
    kl_per_jet = kl_per_particle.sum(1) / n_active              # (B,)

    return kl_per_jet.mean()


# ============================================================
# 2.  NIVEL 2 — KL + orden descendente de pT  (medio riesgo)
# ============================================================

def n_eff_loss_sorted(feat, recon, mask, eps: float = 1e-8):
    """
    Extiende n_eff_loss_kl alineando las distribuciones por orden
    descendente de pt_frac ANTES de comparar.

    Motivación:
      La n_eff_loss original y la versión KL son invariantes a
      permutaciones de constituyentes: si el AE coloca el pT del
      leading subjet en el constituyente 15 en vez del 0, la loss
      no lo detecta.  Ordenar por pT descendente hace que la
      comparación sea estructura-a-estructura: el leading subjet
      se compara con el leading subjet reconstruido, etc.

    Esto captura directamente la estructura 2-prong del W/Z:
      - Un jet QCD típico tiene p_f[0] >> p_f[1] (jet dominante)
      - Un jet WW tiene dos picos comparables p_f[0] ≈ p_f[1]
      El AE entrenado en QCD reconstruirá mal esa bi-modalidad
      → KL sobre distribuciones ordenadas detecta el fallo.

    Nota importante:
      torch.sort con stable=True es diferenciable mediante
      straight-through estimators en PyTorch >= 2.0 cuando se usa
      en el forward pass de un módulo.  El sort no necesita ser
      diferenciable directamente porque se aplica a feat (que no
      requiere gradiente en este contexto) y a recon (donde el
      sort actúa como una re-indexación del gradiente de vuelta
      al orden original de la permutación aprendida).
      En la práctica los gradientes fluyen correctamente.

    Riesgo de sculpting:
      Medio. La estructura ordinal del espectro de pT está más
      correlacionada con SDMass que la entropía total. Mantener
      lambda_neff ≤ 0.1 y el DisCo dual mitiga el riesgo.

    Costo computacional:
      O(N · P · log P) por el sort — despreciable en P=30.
    """
    # ── Input ──────────────────────────────────────────────────────
    p_f_raw = feat[:, :, 3].clamp(min=0.0)

    # Ordenar descendente; constituyentes padding quedan al final
    # porque p_f=0 → van a la cola naturalmente
    p_f_sorted, _ = torch.sort(p_f_raw, dim=1, descending=True, stable=True)
    p_f_sorted = p_f_sorted.clamp(min=eps)

    # ── Reconstrucción ─────────────────────────────────────────────
    pt_r_raw = (recon[:, :, 3] * mask).clamp(min=0.0)
    pt_r_sorted, _ = torch.sort(pt_r_raw, dim=1, descending=True, stable=True)
    sr = pt_r_sorted.sum(1, keepdim=True) + eps
    p_r_sorted = (pt_r_sorted / sr).clamp(min=eps)

    # ── Máscara reordenada ─────────────────────────────────────────
    mask_sorted, _ = torch.sort(mask, dim=1, descending=True, stable=True)

    # ── KL sobre distribuciones ordenadas ─────────────────────────
    kl_per_particle = p_f_sorted * (torch.log(p_f_sorted) - torch.log(p_r_sorted))
    kl_per_particle = kl_per_particle * mask_sorted

    n_active = mask_sorted.sum(1).clamp(min=1.0)
    kl_per_jet = kl_per_particle.sum(1) / n_active

    return kl_per_jet.mean()


# ============================================================
# 3.  NIVEL 3 — Earth Mover's Distance energética  (mayor impacto)
# ============================================================

def n_eff_loss_emd(feat, recon, mask, eps: float = 1e-8):
    """
    Earth Mover's Distance (= Wasserstein-1) entre la distribución
    de pt_frac del input y la reconstrucción, calculada sobre la
    distribución de pT ORDENADA.

    Fundamento:
      Para distribuciones 1D, W1 = ∫|F_p - F_q| dx donde F es la
      CDF.  Para distribuciones discretas ordenadas:
          W1(p, q) = Σ_i |CDF_p[i] - CDF_q[i]|
      Esta es la métrica óptima de transporte más natural para
      comparar espectros de energía de jets.

    Ventajas sobre KL:
      1. Simétrica: no hay modo de colapso asimétrico (KL→∞ si
         p_r_i=0 donde p_f_i>0).
      2. Continua en la distribución: pequeños desplazamientos de
         pT producen gradientes suaves, no spikes logarítmicos.
      3. Captura la "distancia de trabajo" para redistribuir el
         pT, que es exactamente lo que un AE debe reconstruir.
      4. Referenciada en la literatura de jets: Komiske, Metodiev,
         Thaler (2019), arXiv:1902.02346 — "Metric Space of
         Collider Events" usa EMD sobre jets completos.

    Implementación:
      Paso 1 — Ordenar pt_frac descendente (= distribución 1D).
      Paso 2 — Calcular CDFs acumuladas.
      Paso 3 — W1 = media del valor absoluto de la diferencia de CDFs.

    Diferenciabilidad:
      torch.sort y torch.cumsum son diferenciables en PyTorch.
      El gradiente de W1 respecto a p_r es proporcional al signo
      de (CDF_f - CDF_r), lo que produce señales de gradiente
      muy limpias (no logarítmicas) para el decoder.

    Riesgo de sculpting:
      Similar a Nivel 2. La distribución acumulada de pT ordenado
      lleva información de la estructura del jet, correlacionada
      con SDMass. Mantener lambda_neff ≤ 0.05–0.1.

    Costo computacional:
      O(N · P · log P) — igual que Nivel 2.
    """
    # ── Input: ordenar pt_frac descendente ────────────────────────
    p_f_raw = feat[:, :, 3].clamp(min=0.0)
    p_f_sorted, _ = torch.sort(p_f_raw, dim=1, descending=True, stable=True)

    # Normalizar (por si la suma difiere de 1 por efectos de máscara)
    sf = p_f_sorted.sum(1, keepdim=True) + eps
    p_f_norm = p_f_sorted / sf                                   # (B, P)

    # ── Reconstrucción: ordenar pT descendente ────────────────────
    pt_r_raw = (recon[:, :, 3] * mask).clamp(min=0.0)
    pt_r_sorted, _ = torch.sort(pt_r_raw, dim=1, descending=True, stable=True)
    sr = pt_r_sorted.sum(1, keepdim=True) + eps
    p_r_norm = pt_r_sorted / sr                                  # (B, P)

    # ── CDFs acumuladas ───────────────────────────────────────────
    cdf_f = torch.cumsum(p_f_norm, dim=1)                        # (B, P)
    cdf_r = torch.cumsum(p_r_norm, dim=1)                        # (B, P)

    # ── Máscara para ignorar posiciones padding ───────────────────
    mask_sorted, _ = torch.sort(mask, dim=1, descending=True, stable=True)

    # ── W1 = media de |CDF_f - CDF_r| sobre partículas activas ───
    w1_per_particle = torch.abs(cdf_f - cdf_r) * mask_sorted     # (B, P)
    n_active = mask_sorted.sum(1).clamp(min=1.0)                 # (B,)
    w1_per_jet = w1_per_particle.sum(1) / n_active               # (B,)

    return w1_per_jet.mean()


# ============================================================
# 4.  BONUS — log_pt_rel reconstruction loss  (canal feat[:,2])
# ============================================================

def log_pt_rel_loss(feat, recon, mask, eps: float = 1e-8):
    """
    MSE entre log_pt_rel del input y la reconstrucción (canal 2).

    Este canal — log(1 + pt_frac·100) — es una transformación
    monotónica de pt_frac y tiene mejor comportamiento numérico
    porque está en escala logarítmica (gradientes más suaves en
    la región de baja energía).

    Complementa n_eff_loss: mientras que las variantes anteriores
    comparan distribuciones globales de pT, esta penaliza la
    reconstrucción incorrecta de la ESCALA LOGARÍTMICA por
    constituyente, que es la representación que usa el encoder.

    Puede combinarse con cualquiera de las variantes anteriores:
        loss = chamfer + λ_neff · n_eff_loss_emd(...)
                       + λ_logpt · log_pt_rel_loss(...)

    Riesgo de sculpting:
      Bajo-Medio. log_pt_rel es más suave que pt_frac y su
      correlación con SDMass es más indirecta.

    Costo:
      O(N · P) — puramente MSE por partícula.
    """
    logpt_f = feat[:, :, 2]                                      # (B, P)
    logpt_r = recon[:, :, 2] * mask                              # (B, P)

    # MSE solo sobre constituyentes activos
    diff2 = (logpt_f - logpt_r) ** 2 * mask                      # (B, P)
    n_active = mask.sum(1).clamp(min=1.0)                        # (B,)
    mse_per_jet = diff2.sum(1) / n_active                        # (B,)

    return mse_per_jet.mean()


# ============================================================
# 5.  Función combinada recomendada para train.py
# ============================================================

def energy_reconstruction_loss(
    feat,
    recon,
    mask,
    variant: str = "emd",        # "kl" | "sorted_kl" | "emd"
    lambda_logpt: float = 0.5,   # peso del término log_pt_rel
    eps: float = 1e-8,
):
    """
    Loss de reconstrucción energética combinada:

        L_energy = L_neff_variant  +  λ_logpt · L_log_pt_rel

    Recomendación para SONIC v12 / JHEP:
      variant="emd"  con lambda_logpt=0.5

    Cómo integrar en train.py:
    ---------------------------
    # Reemplazar:
    #   neff = n_eff_loss(feat[is_qcd], recon[is_qcd], mask[is_qcd])
    # Por:
    #   neff = energy_reconstruction_loss(
    #              feat[is_qcd], recon[is_qcd], mask[is_qcd],
    #              variant="emd", lambda_logpt=0.5
    #          )
    # El peso lambda_neff en la loss total permanece igual (0.1).

    Args
    ----
    feat     : tensor (B, P, 8)  — features del input
    recon    : tensor (B, P, 8)  — output del decoder
    mask     : tensor (B, P)     — 1 = partícula real, 0 = padding
    variant  : variante de la loss de distribución de pT
    lambda_logpt : peso del término log_pt_rel adicional
    """
    if variant == "kl":
        loss_dist = n_eff_loss_kl(feat, recon, mask, eps)
    elif variant == "sorted_kl":
        loss_dist = n_eff_loss_sorted(feat, recon, mask, eps)
    elif variant == "emd":
        loss_dist = n_eff_loss_emd(feat, recon, mask, eps)
    else:
        raise ValueError(f"variant desconocida: {variant!r}. "
                         f"Usa 'kl', 'sorted_kl' o 'emd'.")

    loss_logpt = log_pt_rel_loss(feat, recon, mask, eps)

    return loss_dist + lambda_logpt * loss_logpt


# ============================================================
# 6.  Tests rápidos de sanidad (ejecutar con: python losses_improved.py)
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(0)
    B, P = 16, 30

    # Construir feat y recon sintéticos
    pt_raw   = torch.rand(B, P).abs()
    mask     = (pt_raw > 0.05).float()
    pt_raw   = pt_raw * mask
    pt_sum   = pt_raw.sum(1, keepdim=True) + 1e-8
    pt_frac  = pt_raw / pt_sum
    logpt    = torch.log1p(pt_frac * 100) * mask

    feat  = torch.zeros(B, P, 8)
    feat[:, :, 2] = logpt
    feat[:, :, 3] = pt_frac

    # Reconstrucción perfecta → todas las losses deben ser ~0
    recon_perfect = feat.clone()
    recon_perfect[:, :, 3] = pt_frac   # canal energy = pt_frac

    # Reconstrucción ruidosa
    recon_noisy = feat.clone()
    recon_noisy[:, :, 3] = (pt_frac + 0.05 * torch.randn(B, P)).clamp(0)

    print("=" * 55)
    print("  Sanity checks — reconstrucción perfecta vs ruidosa")
    print("=" * 55)

    for name, fn in [
        ("n_eff_loss (original)", n_eff_loss),
        ("n_eff_loss_kl",         n_eff_loss_kl),
        ("n_eff_loss_sorted",     n_eff_loss_sorted),
        ("n_eff_loss_emd",        n_eff_loss_emd),
    ]:
        l_perf  = fn(feat, recon_perfect, mask).item()
        l_noisy = fn(feat, recon_noisy,   mask).item()
        ratio   = l_noisy / (l_perf + 1e-10)
        print(f"  {name:<28}  perf={l_perf:.6f}  noisy={l_noisy:.6f}  ratio={ratio:.1f}×")

    print()
    print("  energy_reconstruction_loss (emd + logpt):")
    l_p = energy_reconstruction_loss(feat, recon_perfect, mask).item()
    l_n = energy_reconstruction_loss(feat, recon_noisy,   mask).item()
    print(f"    perf={l_p:.6f}  noisy={l_n:.6f}  ratio={l_n/(l_p+1e-10):.1f}×")
    print("=" * 55)

    # Test de invarianza a permutación — el PROBLEMA 2 del original
    print()
    print("  Test de invarianza a permutación:")
    perm     = torch.randperm(P)
    feat_perm = feat.clone()
    feat_perm[:, :, 3] = feat[:, perm, 3]   # permutar pt_frac

    for name, fn in [
        ("n_eff_loss (original)  — DEBE ser invariante", n_eff_loss),
        ("n_eff_loss_kl          — DEBE ser invariante", n_eff_loss_kl),
        ("n_eff_loss_sorted      — NO debe ser invariante", n_eff_loss_sorted),
        ("n_eff_loss_emd         — NO debe ser invariante", n_eff_loss_emd),
    ]:
        l_orig = fn(feat,      recon_noisy, mask).item()
        l_perm = fn(feat_perm, recon_noisy, mask).item()
        diff   = abs(l_orig - l_perm)
        status = "OK (invariante)" if diff < 1e-4 else f"OK (sensible, Δ={diff:.4f})"
        print(f"  {name[:40]:<40}  {status}")

    print()
    print("[✓] Todos los tests completados.")
