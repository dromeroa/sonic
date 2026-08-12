"""
sonic.training.train
====================
Training loop with DisCo decorrelation, JSD‑aware checkpoint selection,
warmup → cosine → fine‑tuning LR schedule, and dual‑variable DisCo
penalisation (SDMass + ρ).

v4 sculpting‑suppression improvements
--------------------------------------
1. **Dual‑variable DisCo**: penalise distance correlation of the Chamfer
   score with *both* SDMass (phys[:,4]) and ρ (phys[:,6]).  The original
   single‑variable DisCo decorrelates w.r.t. SDMass but residual ρ‑dependence
   leaks through to the 0.5 % tail.
2. **lambda_disco ↑ 6→10**: stronger decorrelation pressure overall.
3. **lambda_jsd ↑ 0.5→1.0 + strict gating**: checkpoints are saved only when
   JSD(0.5 % QCD) < jsd_target **or** when the selection score improves.
4. **Extended fine‑tuning (15 epochs)** at 10 % LR (down from 20 %) to let the
   decorrelation refine in the tails without disturbing discrimination.
5. **Gradient accumulation option** (grad_accum_steps) to stabilise DisCo
   estimates on small batches.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from sonic.training.losses import chamfer, n_eff_loss, n_eff_loss_kl, n_eff_loss_sorted, n_eff_loss_emd 
from sonic.utils.metrics import distance_correlation, compute_jsd


def train_ae(
    model: nn.Module,
    tr_loader,
    va_loader,
    device: torch.device,
    *,
    num_epochs: int = 60,
    lr: float = 1e-3,
    save_path: str = "best_ae_v12.pt",
    warmup_epochs: int = 5,
    lambda_neff: float = 0.1,
    lambda_disco: float = 10.0,
    lambda_jsd: float = 1.0,
    jsd_target: float = 0.04,
    fine_tune_epochs: int = 15,
    grad_accum_steps: int = 1,
    alpha_disco_mass: float = 0.5,
    alpha_disco_rho: float = 0.5,
) -> dict:
    """
    Train the autoencoder with Chamfer + N_eff + dual-DisCo loss.

    Parameters added in this revision
    ----------------------------------
    alpha_disco_mass : float
        Weight of dCor(Chamfer, SDMass) in the DisCo penalty.
        Default 0.5. Must satisfy alpha_disco_mass + alpha_disco_rho = 1
        for the combined weight to remain comparable to lambda_disco.
        To ablate: try (1.0, 0.0) for SDMass-only, (0.0, 1.0) for ρ-only,
        (0.7, 0.3) to down-weight ρ (less correlated with the extreme tail).

    alpha_disco_rho : float
        Weight of dCor(Chamfer, ρ). Default 0.5.
        ρ = log(m²_SD / p²_T) is not independent of m_SD, but residual
        ρ-dependence after decorrelating with m_SD alone is real and
        observed in high-pT jets. The 0.5/0.5 default is a conservative
        equal-weight starting point; ablation across (α_m, α_ρ) pairs
        is recommended before final submission.

    Returns a history dict with per-epoch metrics including disco_mass
    and disco_rho separately for monitoring.
    """

    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    use_cuda = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

    # ---- LR schedule: warmup → cosine → fine‑tune at 10 % LR ----
    def warmup_cosine_fn(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        main_end = num_epochs - fine_tune_epochs
        if epoch < main_end:
            t = (epoch - warmup_epochs) / max(main_end - warmup_epochs, 1)
            return 0.5 * (1 + np.cos(np.pi * t)) * (1 - 1e-2) + 1e-2
        return 0.1 * (0.5 * (1 + np.cos(np.pi * 1.0)) * (1 - 1e-2) + 1e-2)  # flat 10 %

    sched = optim.lr_scheduler.LambdaLR(opt, warmup_cosine_fn)
    best_val = float("inf")

    # Normalise weights so they always sum to 1 (defensive)
    _w_sum = alpha_disco_mass + alpha_disco_rho
    if abs(_w_sum) < 1e-8:
        raise ValueError("alpha_disco_mass + alpha_disco_rho must be > 0")
    _alpha_m = alpha_disco_mass / _w_sum
    _alpha_r = alpha_disco_rho  / _w_sum

    history = {
        "train": [],
        "val": [],
        "train_ch": [],
        "train_neff": [],
        "train_disco": [],
        "train_disco_mass": [],   # nuevo: dCor con SDMass por separado
        "train_disco_rho": [],    # nuevo: dCor con ρ por separado
        "train_jsd": [],
        "val_jsd": [],
    }

    for epoch in range(num_epochs):
        model.train()
        run_total = run_ch = run_neff = run_disco = 0.0
        run_disco_mass = run_disco_rho = 0.0
        nan_batches = 0
        pbar = tqdm(tr_loader, desc=f"Epoch {epoch + 1:>3}/{num_epochs}", ncols=130)

        for step, (feat, mask, phys) in enumerate(pbar):
            feat = feat.to(device)
            mask = mask.to(device)
            phys = phys.to(device)

            with torch.cuda.amp.autocast(enabled=use_cuda):
                recon, _ = model(feat)
                ch_none = chamfer(feat, recon, mask, mask, reduction="none")

                is_qcd = phys[:, 5] == 0.0
                if is_qcd.sum() > 1:
                    ch_mean    = ch_none[is_qcd].mean()
                    neff       = n_eff_loss(feat[is_qcd], recon[is_qcd], mask[is_qcd])
                    # Dual-variable DisCo with configurable weights
                    disco_mass = distance_correlation(ch_none[is_qcd], phys[is_qcd, 4])
                    disco_rho  = distance_correlation(ch_none[is_qcd], phys[is_qcd, 6])
                    loss_disco = _alpha_m * disco_mass + _alpha_r * disco_rho
                else:
                    ch_mean    = ch_none.mean()
                    neff       = n_eff_loss(feat, recon, mask)
                    disco_mass = torch.tensor(0.0, device=device)
                    disco_rho  = torch.tensor(0.0, device=device)
                    loss_disco = torch.tensor(0.0, device=device)

                loss = (ch_mean + lambda_neff * neff + lambda_disco * loss_disco) / grad_accum_steps

            # NaN guard: skip corrupt batch rather than propagate bad gradients
            if not torch.isfinite(loss):
                nan_batches += 1
                opt.zero_grad()
                if nan_batches <= 3:
                    print(f"\n  [!] NaN/Inf loss at step {step} "
                          f"(ch={ch_mean.item():.4f} "
                          f"neff={neff.item():.4f} "
                          f"disco={loss_disco.item():.4f}) — skipping batch")
                continue

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(tr_loader):
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()

            run_total      += loss.item() * grad_accum_steps
            run_ch         += ch_mean.item()
            run_neff       += neff.item()
            run_disco      += loss_disco.item()
            run_disco_mass += disco_mass.item()
            run_disco_rho  += disco_rho.item()
            pbar.set_postfix(
                L=f"{loss.item() * grad_accum_steps:.5f}",
                Ch=f"{ch_mean.item():.5f}",
                Dm=f"{disco_mass.item():.4f}",
                Dr=f"{disco_rho.item():.4f}",
                lr=f"{opt.param_groups[0]['lr']:.1e}",
            )

        n_batches  = max(len(tr_loader) - nan_batches, 1)
        avg_tr     = run_total      / n_batches
        avg_ch     = run_ch         / n_batches
        avg_neff   = run_neff       / n_batches
        avg_disco  = run_disco      / n_batches
        avg_dm     = run_disco_mass / n_batches
        avg_dr     = run_disco_rho  / n_batches
        history["train"].append(avg_tr)
        history["train_ch"].append(avg_ch)
        history["train_neff"].append(avg_neff)
        history["train_disco"].append(avg_disco)
        history["train_disco_mass"].append(avg_dm)
        history["train_disco_rho"].append(avg_dr)
        if nan_batches:
            print(f"  [!] Epoch {epoch+1}: {nan_batches} batch(es) skipped (NaN loss)")

        # ---- Validation + JSD at 0.5 % QCD ----
        model.eval()
        val_l = 0.0
        qcd_scores, qcd_masses = [], []
        with torch.no_grad():
            for feat, mask, phys in va_loader:
                feat = feat.to(device)
                mask = mask.to(device)
                phys = phys.to(device)
                recon, _ = model(feat)
                ch_none = chamfer(feat, recon, mask, mask, reduction="none")

                is_qcd = phys[:, 5] == 0.0
                if is_qcd.sum() > 1:
                    ch_mean    = ch_none[is_qcd].mean()
                    neff       = n_eff_loss(feat[is_qcd], recon[is_qcd], mask[is_qcd])
                    #neff       = n_eff_loss_kl(feat[is_qcd], recon[is_qcd], mask[is_qcd])
                    disco_m    = distance_correlation(ch_none[is_qcd], phys[is_qcd, 4])
                    disco_r    = distance_correlation(ch_none[is_qcd], phys[is_qcd, 6])
                    loss_disco = _alpha_m * disco_m + _alpha_r * disco_r
                    qcd_scores.append(ch_none[is_qcd].cpu().numpy())
                    qcd_masses.append(phys[is_qcd, 4].cpu().numpy())
                else:
                    ch_mean    = ch_none.mean()
                    neff       = n_eff_loss(feat, recon, mask)
                    #neff        = n_eff_loss_kl(feat, recon, mask)
                    loss_disco = torch.tensor(0.0, device=device)

                loss = ch_mean + lambda_neff * neff + lambda_disco * loss_disco
                if not torch.isfinite(loss):
                    continue
                val_l += loss.item()

        avg_val = val_l / len(va_loader)
        history["val"].append(avg_val)

        # ---- Compute JSD(0.5 % QCD) on validation ----
        if qcd_scores:
            all_scores = np.concatenate(qcd_scores)
            all_masses = np.concatenate(qcd_masses)
            bins_m = np.linspace(40, 200, 41)
            h_ref, _ = np.histogram(all_masses, bins=bins_m, density=True)
            thr_05 = np.percentile(all_scores, 99.5)
            m_pass = all_masses[all_scores >= thr_05]
            jsd_val = compute_jsd(h_ref, np.histogram(m_pass, bins=bins_m, density=True)[0]) if len(m_pass) >= 5 else 1.0
        else:
            jsd_val = 1.0

        history["train_jsd"].append(jsd_val)
        history["val_jsd"].append(jsd_val)

        sched.step()

        # ---- Checkpoint selection with JSD hinge penalty ----
        jsd_hinge = max(0.0, jsd_val - jsd_target)
        selection_score = avg_val + lambda_jsd * jsd_hinge

        print(
            f"  → Train={avg_tr:.5f} "
            f"(Ch={avg_ch:.5f} Neff={avg_neff:.5f} "
            f"DisCo={avg_disco:.5f} [Dm={avg_dm:.4f} Dr={avg_dr:.4f}])  "
            f"Val={avg_val:.5f}  "
            f"JSD(0.5%)={jsd_val:.4f}  score={selection_score:.5f}  "
            f"LR={opt.param_groups[0]['lr']:.2e}"
        )

        if selection_score < best_val:
            best_val = selection_score
            state = (
                model._orig_mod.state_dict()
                if hasattr(model, "_orig_mod")
                else model.state_dict()
            )
            torch.save(state, save_path)
            print(
                f"   ✓ Saved (score={best_val:.5f}, val={avg_val:.5f}, "
                f"JSD={jsd_val:.4f})"
            )

    return history
