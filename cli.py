#!/usr/bin/env python3
"""
sonic.cli
=========
Command‑line entry point for the SONIC v4 jet‑tagging pipeline.

Usage (terminal)::

    python -m sonic.cli --epochs 60 --max-rows 30000

Usage (notebook)::

    from sonic.cli import main
    main(["--epochs", "60", "--max-rows", "30000"])
"""

import argparse
import glob
import itertools
import json
import os
import warnings
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from torch.utils.data import ConcatDataset, DataLoader

from sonic.utils.config import DEFAULTS, configure_system_resources, fijar_semilla, seed_worker
from sonic.utils.metrics import bootstrap_jsd, bootstrap_roc_sic
from sonic.data.prep import PrepV12
from sonic.data.datasets import JetDatasetV12
from sonic.models.jet_autoencoder import JetAE_V12
from sonic.models.ddt2d import DDTransform2D
from sonic.training.train import train_ae
from sonic.inference.eval import collect_raw, fix_dir_on_validation, grid_search_hybrid
from sonic.inference.plots import plot_training, plot_ddt2d_fit, plot_all

# Silence noisy but harmless warnings
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.*is deprecated.*")
warnings.filterwarnings("ignore", message=".*GradScaler.*is deprecated.*")
warnings.filterwarnings("ignore", message=".*Tight layout not applied.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*divide by zero encountered.*", category=RuntimeWarning)


# =====================================================================
# CLI argument parser
# =====================================================================
def build_parser() -> argparse.ArgumentParser:
    D = DEFAULTS
    p = argparse.ArgumentParser(
        description="SONIC Tagger v4: AE + Chebyshev DDT2D + DisCo + JSD gating"
    )
    p.add_argument("--max-rows", type=int, default=30000,
                   help="Max CSV rows (-1 = unlimited)")
    p.add_argument("--epochs", type=int, default=D["EPOCHS"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-seeds", type=int, default=1,
                   help="Independent seeds to run (>=5 for publication)")
    p.add_argument("--lambda-disco", type=float, default=D["LAMBDA_DISCO"],
                   help="DisCo penalty weight (default 10.0)")
    p.add_argument("--lambda-jsd", type=float, default=D["LAMBDA_JSD"],
                   help="JSD hinge penalty for checkpoint selection")
    p.add_argument("--jsd-target", type=float, default=D["JSD_TARGET"])
    p.add_argument("--train-pattern", type=str, default="signalM1000_last.csv")
    p.add_argument("--test-file", type=str, default="qcd_background.csv",
                   help="Signal file(s): comma‑separated or glob")
    p.add_argument("--n-boot", type=int, default=D["N_BOOT"])
    p.add_argument("--force-cache", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--static-graph", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--grad-accum", type=int, default=1,
                   help="Gradient accumulation steps (stabilises DisCo on small batches)")
    # ── Recursos de cómputo en nube ───────────────────────────────── [CLOUD]
    p.add_argument("--n-cores", type=int, default=10,
                   help="[CLOUD] Cores CPU disponibles (DataLoader workers + torch threads)")
    p.add_argument("--max-ram-gb", type=int, default=16,
                   help="[CLOUD] Límite de RAM en GB (default 16)")
    # ── Control de tamaño de muestra ─────────────────────────────── [CLOUD]
    p.add_argument("--n-signal", type=int, default=None,
                   help="[CLOUD] Jets de señal a usar (None = todos los disponibles)")
    p.add_argument("--n-background", type=int, default=None,
                   help="[CLOUD] Jets de QCD/fondo a usar (None = todos los disponibles)")
    return p


# =====================================================================
# Single‑seed run
# =====================================================================
def run_one_seed(
    seed: int,
    args,
    all_ds: ConcatDataset,
    qcd_indices: list,
    sig_indices: list,
    event_file_id: np.ndarray,
    all_files: list,
    device: torch.device,
    threads_per_seed: int = 0,   # 0 = no override (modo serie)
):
    D = DEFAULTS
    MAX_P = D["MAX_P"]
    BATCH = D["BATCH"]

    # [PERF] En modo paralelo, cada proceso hijo limita sus threads para evitar
    # contención: n_seeds procesos × threads_per_seed = n_cores totales.
    if threads_per_seed > 0:
        torch.set_num_threads(threads_per_seed)
        torch.set_num_interop_threads(max(1, threads_per_seed // 2))

    print(f"\n{'#' * 70}\n  SEED = {seed}\n{'#' * 70}")
    print(f"  [PERF] torch threads: {torch.get_num_threads()} | "
          f"interop: {torch.get_num_interop_threads()}")
    fijar_semilla(seed)

    # ---- Split 50 / 30 / 20 ----
    g = torch.Generator().manual_seed(seed)
    qcd_shuf = torch.tensor(qcd_indices)[torch.randperm(len(qcd_indices), generator=g)].tolist()
    sig_shuf = torch.tensor(sig_indices)[torch.randperm(len(sig_indices), generator=g)].tolist()

    nq = len(qcd_shuf)
    nq_tr, nq_va = int(0.50 * nq), int(0.30 * nq)
    qcd_tr = qcd_shuf[:nq_tr]
    qcd_va = qcd_shuf[nq_tr:nq_tr + nq_va]
    qcd_te = qcd_shuf[nq_tr + nq_va:]

    ns = len(sig_shuf)
    ns_tr, ns_va = int(0.50 * ns), int(0.30 * ns)
    sig_tr = sig_shuf[:ns_tr]
    sig_va = sig_shuf[ns_tr:ns_tr + ns_va]
    sig_te = sig_shuf[ns_tr + ns_va:]

    nq_vh = len(qcd_va) // 2
    qcd_va_ddt = qcd_va[:nq_vh]
    qcd_va_gs = qcd_va[nq_vh:]

    print(f"  QCD  → train {len(qcd_tr)} | val {len(qcd_va)} "
          f"(DDT {len(qcd_va_ddt)} + GS {len(qcd_va_gs)}) | test {len(qcd_te)}")
    print(f"  Sig  → train {len(sig_tr)} | val {len(sig_va)} | test {len(sig_te)}")

#    tr_ds = torch.utils.data.Subset(all_ds, qcd_tr + sig_tr)
    tr_ds = torch.utils.data.Subset(all_ds, qcd_tr)
    va_ds = torch.utils.data.Subset(all_ds, qcd_va + sig_va)
    ddt_ds = torch.utils.data.Subset(all_ds, qcd_va_ddt)
    gs_ds = torch.utils.data.Subset(all_ds, qcd_va_gs + sig_va)
    te_ds = torch.utils.data.Subset(all_ds, qcd_te + sig_te)

    g_train = torch.Generator().manual_seed(seed)
    # [CLOUD] Repartir cores: reservar 2 para cómputo PyTorch, el resto → DataLoader workers.
    # pin_memory=False porque no hay GPU; en CPU es overhead neto.
    auto_workers = max(1, args.n_cores - 2)                          # [CLOUD]
    n_workers = args.num_workers if args.num_workers > 0 else auto_workers  # [CLOUD]
    print(f"  [CLOUD] DataLoader workers: {n_workers}")              # [CLOUD]
    dl = dict(batch_size=BATCH, num_workers=n_workers, pin_memory=False)  # [CLOUD]
    if n_workers > 0:                                                 # [CLOUD]
        dl["worker_init_fn"] = seed_worker
        # persistent_workers=True omitido: en CPU con multi-seed causa
        # OSError/semaphore errors al destruir DataLoaders entre procesos.

    tr_loader = DataLoader(tr_ds, shuffle=True, generator=g_train, **dl)
    va_loader = DataLoader(va_ds, **dl)
    ddt_loader = DataLoader(ddt_ds, **dl)
    gs_loader = DataLoader(gs_ds, **dl)
    te_loader = DataLoader(te_ds, **dl)

    # ---- Model ----
    MODEL_PATH = f"best_ae_v12_sonic_seed{seed}.pt"
    model = JetAE_V12(n_part=MAX_P, latent=D["LATENT"], ch=D["CHANNELS"],
                      static_graph=args.static_graph).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {n_params:,} trainable params")

    if args.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"  [!] compile failed ({e}), continuing without")

    # ---- Train ----
    history = train_ae(
        model, tr_loader, va_loader, device,
        num_epochs=args.epochs, lr=D["LR"], save_path=MODEL_PATH,
        warmup_epochs=D["WARMUP_EP"], lambda_neff=D["LAMBDA_NEFF"],
        lambda_disco=args.lambda_disco, lambda_jsd=args.lambda_jsd,
        alpha_disco_mass=0.3,   # menos énfasis en SDMass
        alpha_disco_rho=0.7,    # más presión en ρ → ataca cola 0.5%
        jsd_target=args.jsd_target, fine_tune_epochs=D["FINE_TUNE_EPOCHS"],
        grad_accum_steps=args.grad_accum,
    )
    plot_training(history, save_path=f"training_history_sonic_seed{seed}.png")

    # ---- Load best checkpoint ----
    target = model._orig_mod if hasattr(model, "_orig_mod") else model
    target.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))

    # ---- Collect scores ----
    C_df, T21_df, R_df, L_df, M_df, LP_df, N2_df = collect_raw(model, ddt_loader, device)
    C_gs, T21_gs, R_gs, L_gs, M_gs, LP_gs, N2_gs = collect_raw(model, gs_loader, device)
    C_te, T21_te, R_te, L_te, M_te, LP_te, N2_te = collect_raw(model, te_loader, device)


    # línea 177 — renombrar las máscaras booleanas
    qcd_df_mask = L_df == 0
    qcd_gs_mask = L_gs == 0
    qcd_te_mask = L_te == 0          # ← ya no pisa qcd_te (lista de índices)

 #   qcd_df = L_df == 0; qcd_gs = L_gs == 0; qcd_te = L_te == 0

    C_log_df = np.log(C_df + 1e-7)
    C_log_gs = np.log(C_gs + 1e-7)
    C_log_te = np.log(C_te + 1e-7)

    # ---- DDT 2D fits (higher degree + finer bins for sculpting suppression) ----
    print(f"\n{'=' * 60}\n  DDT2D FITTING (loc+scale, deg 4×3, 16×8 bins)\n{'=' * 60}")

    ddt_C = DDTransform2D(name="log(Chamfer)", deg_rho=4, deg_pt=3, quantile=D["DDT_QUANT"])
    xc, yc, zc = ddt_C.fit(C_log_df[qcd_df_mask], R_df[qcd_df_mask], LP_df[qcd_df_mask])
    ddt_C.save(f"ddt_logchamfer_sonic_seed{seed}.json")

    ddt_T21 = DDTransform2D(name="Tau21", deg_rho=4, deg_pt=3, quantile=D["DDT_QUANT"])
    xt, yt, zt = ddt_T21.fit(T21_df[qcd_df_mask], R_df[qcd_df_mask], LP_df[qcd_df_mask])
    ddt_T21.save(f"ddt_tau21_sonic_seed{seed}.json")

    ddt_N2 = DDTransform2D(name="N2", deg_rho=4, deg_pt=3, quantile=D["DDT_QUANT"])
    xn, yn, zn = ddt_N2.fit(N2_df[qcd_df_mask], R_df[qcd_df_mask], LP_df[qcd_df_mask])
    ddt_N2.save(f"ddt_n2_sonic_seed{seed}.json")

    for d, n in [(ddt_C, "log(Chamfer)"), (ddt_T21, "Tau21"), (ddt_N2, "N2")]:
        if not d.fit_ok:
            print(f"  [!!!] DDT2D for {n} FAILED (null coefficients).")

    # ---- Transform ----
    C_gs_ddt = ddt_C.transform(C_log_gs, R_gs, LP_gs)
    T21_gs_ddt = ddt_T21.transform(T21_gs, R_gs, LP_gs)
    N2_gs_ddt = ddt_N2.transform(N2_gs, R_gs, LP_gs)

    # ---- Score clipping asimétrico — eliminación de artefactos de extrapolación DDT ----
    #
    # El fit Chebyshev extrapola fuera del dominio de entrenamiento en jets con ρ > -2
    # (baja estadística), produciendo predicciones de location/scale patológicas que
    # generan scores individuales extremos. El clip asimétrico corrige solo la dirección
    # donde la extrapolación es un artefacto, preservando la cola de señal legítima:
    #
    #   log(Chamfer) DDT: discriminante positivo para WW (jets compactos → score alto).
    #     - Cola baja  (score << 0): artefacto de extrapolación → clip en p0.1% de QCD.
    #     - Cola alta  (score >> 0): jets WW legítimos → NO clipear (np.inf).
    #     Razón: el clip simétrico anterior (p99.9% QCD ≈ 28.6) cortaba jets WW
    #     anómalos, bajando AUC(AE solo) de 0.768 → 0.740 y desplazando α→0.96.
    #
    #   τ₂₁ DDT: variable analítica bien comportada, sin asimetría de señal extrema.
    #     - Ambas colas clippeadas en [p0.1%, p99.9%] de QCD — simétrico es correcto.
    #
    #   N₂ DDT: discriminante negativo para WW (jets de 2-prong → score bajo).
    #     - Cola alta  (score >> 0): artefacto de extrapolación → clip en p99.9% de QCD.
    #     - Cola baja  (score << 0): jets WW legítimos → NO clipear (-np.inf).
    #
    # Todos los límites se calibran sobre QCD del pool grid-search (disjunto del
    # DDT-fit pool) para evitar fuga de información hacia el test set.
    _qcd_gs_C   = C_gs_ddt[qcd_gs_mask]
    _qcd_gs_T21 = T21_gs_ddt[qcd_gs_mask]
    _qcd_gs_N2  = N2_gs_ddt[qcd_gs_mask]

    # log(Chamfer): clip solo cola baja — cola alta libre (señal WW)
    clip_C_lo  = np.percentile(_qcd_gs_C,  0.1)
    clip_C_hi  = np.inf

    # τ₂₁: clip simétrico — variable analítica sin asimetría de señal
    clip_T21_lo, clip_T21_hi = np.percentile(_qcd_gs_T21, [0.1, 99.9])

    # N₂: clip solo cola alta — cola baja libre (señal WW de 2-prong)
    clip_N2_lo = -np.inf
    clip_N2_hi = np.percentile(_qcd_gs_N2, 99.9)

    print(f"  [CLIP] log(Chamfer) DDT  : [{clip_C_lo:.3f}, +inf]  (asimétrico: señal en cola alta)")
    print(f"  [CLIP] τ₂₁ DDT          : [{clip_T21_lo:.3f}, {clip_T21_hi:.3f}]")
    print(f"  [CLIP] N₂ DDT           : [-inf, {clip_N2_hi:.3f}]  (asimétrico: señal en cola baja)")

    C_gs_ddt   = np.clip(C_gs_ddt,   clip_C_lo,   clip_C_hi)
    T21_gs_ddt = np.clip(T21_gs_ddt, clip_T21_lo, clip_T21_hi)
    N2_gs_ddt  = np.clip(N2_gs_ddt,  clip_N2_lo,  clip_N2_hi)

    plot_ddt2d_fit(xc, yc, zc, ddt_C, R_gs[qcd_gs_mask], LP_gs[qcd_gs_mask],
                   C_log_gs[qcd_gs_mask], C_gs_ddt[qcd_gs_mask],
                   f"sonic_ddt_fit_logchamfer_seed{seed}.png")
    plot_ddt2d_fit(xt, yt, zt, ddt_T21, R_gs[qcd_gs_mask], LP_gs[qcd_gs_mask],
                   T21_gs[qcd_gs_mask], T21_gs_ddt[qcd_gs_mask],
                   f"sonic_ddt_fit_tau21_seed{seed}.png")
    plot_ddt2d_fit(xn, yn, zn, ddt_N2, R_gs[qcd_gs_mask], LP_gs[qcd_gs_mask],
                   N2_gs[qcd_gs_mask], N2_gs_ddt[qcd_gs_mask],
                   f"sonic_ddt_fit_n2_seed{seed}.png")

    # ---- Grid search ----
    best_params, auc_val = grid_search_hybrid(C_gs_ddt, T21_gs_ddt, L_gs, N2_ddt=N2_gs_ddt)
    alpha = best_params["alpha"]; beta = best_params.get("beta", 0.0); gamma = best_params["gamma"]

    s_gs_combo = alpha * C_gs_ddt + beta * N2_gs_ddt - gamma * T21_gs_ddt
    sign, auc_val_ori = fix_dir_on_validation(s_gs_combo, L_gs)
    print(f"  Sign convention: {sign:+.0f}  (AUC_val={auc_val_ori:.4f})")

    # ---- Test evaluation ----
    C_te_ddt   = np.clip(ddt_C.transform(C_log_te, R_te, LP_te),     clip_C_lo,   clip_C_hi)
    T21_te_ddt = np.clip(ddt_T21.transform(T21_te, R_te, LP_te),     clip_T21_lo, clip_T21_hi)
    N2_te_ddt  = np.clip(ddt_N2.transform(N2_te, R_te, LP_te),       clip_N2_lo,  clip_N2_hi)

    s_te_raw = sign * (alpha * C_log_te + beta * N2_te - gamma * T21_te)
    s_te_ddt = sign * (alpha * C_te_ddt + beta * N2_te_ddt - gamma * T21_te_ddt)
    auc_test = roc_auc_score(L_te, s_te_ddt)

    # ---- Guardar arrays para orthogonality_probe.py ----
    np.save(f"C_te_ddt_seed{seed}.npy",   C_te_ddt)
    np.save(f"T21_te_ddt_seed{seed}.npy", T21_te_ddt)
    np.save(f"N2_te_ddt_seed{seed}.npy",  N2_te_ddt)
    np.save(f"labels_te_seed{seed}.npy",  L_te)
    print(f"  [*] Arrays DDT guardados para orthogonality probe (seed={seed})")
    print(f"\n  AUC TEST (combined DDT) = {auc_test:.4f}")

    sign_ae, _ = fix_dir_on_validation(C_gs_ddt, L_gs)
    auc_ae = roc_auc_score(L_te, sign_ae * C_te_ddt)

    sign_t21, _ = fix_dir_on_validation(T21_gs_ddt, L_gs)
    auc_t21 = roc_auc_score(L_te, sign_t21 * T21_te_ddt)

    sign_n2, _ = fix_dir_on_validation(N2_gs_ddt, L_gs)
    auc_n2 = roc_auc_score(L_te, sign_n2 * N2_te_ddt)

    rho_after, _ = pearsonr(s_te_ddt[qcd_te_mask], R_te[qcd_te_mask])
    rho_pt_after, _ = pearsonr(s_te_ddt[qcd_te_mask], LP_te[qcd_te_mask])
    print(f"  Pearson(s, ρ)={rho_after:+.4f}   Pearson(s, logpT)={rho_pt_after:+.4f}")

    # ---- Signal mass‑point breakdown ----
    te_global_idx = np.array(qcd_te + sig_te)
    te_fids = event_file_id[te_global_idx]
    sig_fi = [i for i, f in enumerate(all_files) if "signal" in os.path.basename(f).lower()]
    sig_bd = {}
    if len(sig_fi) > 1:
        print("\n  Per‑mass‑point breakdown:")
        for fi in sig_fi:
            sel = (te_fids == fi) & (L_te == 1)
            n_s = int(sel.sum())
            fn = os.path.basename(all_files[fi])
            if n_s < 20:
                continue
            mask_e = sel | (L_te == 0)
            a = roc_auc_score(L_te[mask_e], s_te_ddt[mask_e])
            sig_bd[fn] = {"auc": float(a), "n_sig_test": n_s}
            print(f"    {fn:25s}: AUC={a:.4f}  n={n_s}")

    # ---- Bootstrap ----
    print(f"\n  Bootstrap (n={args.n_boot}) …")
    boot = bootstrap_roc_sic(s_te_ddt, L_te, n_boot=args.n_boot, seed=seed)

    bins_ms = np.linspace(40, 200, 41)
    fpr_cuts = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005]
    jsd_bb = bootstrap_jsd(s_te_ddt[qcd_te_mask], M_te[qcd_te_mask], fpr_cuts, bins_ms,
                           n_boot=args.n_boot, seed=seed)

    # ---- Plot ----
    rej50, sig_max, jsd_vals = plot_all(
        C_log_te, T21_te, s_te_raw, s_te_ddt, L_te, M_te, R_te, auc_test,
        boot_stats=boot, auc_ae_only=auc_ae,
        auc_tau21_only=auc_t21, s_tau21_ddt=sign_t21 * T21_te_ddt,
        auc_n2_only=auc_n2, s_n2_ddt=sign_n2 * N2_te_ddt,
        jsd_boot_bands=jsd_bb, save_prefix=f"sonic_seed{seed}",
    )

    # ---- Persist ----
    result = dict(
        version="v4_SONIC_production",
        seed=seed,
        auc_ddt=float(auc_test),
        auc_ae_only=float(auc_ae),
        auc_tau21_only=float(auc_t21),
        auc_n2_only=float(auc_n2),
        delta_auc_ae_over_tau21=float(auc_test - auc_t21),
        delta_auc_ae_over_n2=float(auc_test - auc_n2),
        rej_at_50tpr=float(rej50),
        max_significance=float(sig_max),
        jsd_1pct=float(jsd_vals.get("1.0% QCD", 999)),
        jsd_05pct=float(jsd_vals.get("0.5% QCD", 999)),
        jsd_bootstrap_ci={
            f"{k * 100:.1f}%_QCD": {"p16": v[0], "p50": v[1], "p84": v[2]}
            for k, v in jsd_bb.items()
        },
        signal_mass_breakdown=sig_bd,
        alpha=alpha, beta=beta, gamma=gamma,
        rho_corr_after=float(rho_after),
        lambda_disco_used=args.lambda_disco,
        lambda_jsd_used=args.lambda_jsd,
        jsd_target=args.jsd_target,
    )
    out_path = f"results_sonic_v4_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Results → {out_path}")
    return result


# =====================================================================
# Main orchestrator
# =====================================================================
def main(argv=None):
    """
    Entry point. Pass ``argv`` as a list of strings for notebook use,
    or ``None`` to read from ``sys.argv``.
    """
    # [CLOUD] Parsear primero para leer --n-cores y --max-ram-gb antes de configurar recursos
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print(f"[!] Ignored unknown args: {unknown}")

    # [CLOUD] Exportar n_cores ANTES de fijar_semilla para que OMP/MKL/OpenBLAS queden
    # sincronizados con el valor real de la CLI, no con el hardcode anterior de "14".
    os.environ["SONIC_N_CORES"] = str(args.n_cores)                  # [CLOUD]
    configure_system_resources(                                        # [CLOUD]
        max_ram_gb=args.max_ram_gb,                                    # [CLOUD]
        target_cores=args.n_cores,                                     # [CLOUD]
    )
    print(f"[CLOUD] {args.n_cores} cores | RAM cap {args.max_ram_gb} GB")  # [CLOUD]

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [args.seed + i for i in range(max(1, args.n_seeds))]

    D = DEFAULTS
    MAX_P = D["MAX_P"]
    MAX_ROWS = None if args.max_rows == -1 else args.max_rows

    print(f"[*] Device: {device}  |  Seeds: {seeds}")
    print(f"[*] λ_disco={args.lambda_disco}  λ_jsd={args.lambda_jsd}  jsd_target={args.jsd_target}")

    # ---- File discovery ----
    train_files = sorted(glob.glob(args.train_pattern))
    assert train_files, f"No training files matched '{args.train_pattern}'"

    test_pats = [p.strip() for p in args.test_file.split(",") if p.strip()]
    test_files = sorted(set(itertools.chain.from_iterable(glob.glob(p) for p in test_pats)))
    assert test_files, f"No test files matched '{args.test_file}'"
    all_files = train_files + test_files

    # ---- Preprocessor ----
    SCALER = D["SCALER_PATH"]
    prep = PrepV12(MAX_P)
    if os.path.exists(SCALER):
        prep.load(SCALER)
        print("[*] Scaler loaded")
    else:
        prep.fit(train_files)
        prep.save(SCALER)

    # ---- Global dataset ----
    datasets = []
    for f in all_files:
        is_sig = "signal" in os.path.basename(f).lower()
        mr = MAX_ROWS if is_sig else (MAX_ROWS * 3 if MAX_ROWS else None)
        datasets.append(JetDatasetV12(f, prep, MAX_P, force=args.force_cache, max_rows=mr))
    all_ds = ConcatDataset(datasets)

    event_file_id = np.concatenate([
        np.full(len(ds), i, dtype=np.int64) for i, ds in enumerate(all_ds.datasets)
    ])

    qcd_indices, sig_indices = [], []
    offset = 0
    for ds in all_ds.datasets:
        sig_m = ds.phys[:, 5] == 1.0
        qcd_m = ds.phys[:, 5] == 0.0
        sig_indices.extend((torch.where(sig_m)[0] + offset).tolist())
        qcd_indices.extend((torch.where(qcd_m)[0] + offset).tolist())
        offset += len(ds)
    print(f"[*] Global: {len(qcd_indices)} QCD, {len(sig_indices)} signal jets")

    # ── Submuestreo controlado de jets ───────────────────────────── [CLOUD]
    rng_sub = np.random.default_rng(42)

    if args.n_background is not None and args.n_background < len(qcd_indices):
        n_prev_qcd = len(qcd_indices)
        qcd_indices = rng_sub.choice(
            qcd_indices, size=args.n_background, replace=False
        ).tolist()
        print(f"[CLOUD] QCD submuestreado: {len(qcd_indices)} jets "
              f"(de {n_prev_qcd} disponibles)")                       # [CLOUD]

    if args.n_signal is not None and args.n_signal < len(sig_indices):
        n_prev_sig = len(sig_indices)
        sig_indices = rng_sub.choice(
            sig_indices, size=args.n_signal, replace=False
        ).tolist()
        print(f"[CLOUD] Señal submuestreada: {len(sig_indices)} jets "
              f"(de {n_prev_sig} disponibles)")                       # [CLOUD]

    print(f"[*] Jets efectivos: {len(qcd_indices)} QCD, {len(sig_indices)} señal")
    # ─────────────────────────────────────────────────────────────── [CLOUD]

    # ---- Run seeds ----
    # [PERF] Con 1 seed: directo, sin overhead de spawn.
    # Con N seeds: ProcessPoolExecutor reparte los cores equitativamente.
    # Cada proceso hijo recibe threads_per_seed = n_cores // n_parallel para
    # evitar contención (N procesos × T threads ≤ n_cores físicos).
    n_parallel = min(len(seeds), max(1, args.n_cores // 2))
    threads_per_seed = max(1, args.n_cores // n_parallel)

    if len(seeds) == 1:
        print(f"[PERF] Modo serie (1 seed) — {args.n_cores} threads disponibles")
        all_results = [
            run_one_seed(seeds[0], args, all_ds, qcd_indices, sig_indices,
                         event_file_id, all_files, device, threads_per_seed=0)
        ]
    else:
        print(f"[PERF] Modo paralelo: {len(seeds)} seeds × {n_parallel} procesos "
              f"| {threads_per_seed} torch-threads/seed")

        def _worker(seed):
            return run_one_seed(
                seed, args, all_ds, qcd_indices, sig_indices,
                event_file_id, all_files, device,
                threads_per_seed=threads_per_seed,
            )

        all_results_map = {}
        with ProcessPoolExecutor(max_workers=n_parallel) as pool:
            futures = {pool.submit(_worker, s): s for s in seeds}
            for fut in as_completed(futures):
                s = futures[fut]
                try:
                    all_results_map[s] = fut.result()
                    print(f"[PERF] Seed {s} completado ✓")
                except Exception as exc:
                    print(f"[!] Seed {s} falló: {exc}")
        # Restaurar orden original de seeds para el resumen
        all_results = [all_results_map[s] for s in seeds if s in all_results_map]

    # ---- Multi‑seed summary ----
    if len(seeds) > 1:
        print(f"\n{'=' * 70}\n  MULTI‑SEED SUMMARY ({len(seeds)} runs)\n{'=' * 70}")
        keys = ["auc_ddt", "auc_ae_only", "auc_tau21_only", "auc_n2_only",
                "rej_at_50tpr", "max_significance", "jsd_1pct", "jsd_05pct"]
        summary = {}
        for k in keys:
            vals = np.array([r[k] for r in all_results if r.get(k) is not None], dtype=np.float64)
            if len(vals) == 0:
                continue
            m, s = float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            summary[k] = {"mean": m, "std": s, "values": vals.tolist()}
            print(f"  {k:22s}: {m:.4f} ± {s:.4f}")
        with open("results_sonic_v4_multiseed_summary.json", "w") as f:
            json.dump({"seeds": seeds, "per_seed": all_results, "summary": summary}, f, indent=2)
        print("  → results_sonic_v4_multiseed_summary.json")


if __name__ == "__main__":
    main()
