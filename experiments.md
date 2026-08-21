# Experiments Log — IG-Occ

Every run recorded here. Numbers to beat (EmbodiedOcc++, ACM MM 2025):
**Embodied full 0.522 IoU / 0.436 mIoU** · Embodied-mini 0.529 / 0.437 ·
Local full 0.549 / 0.462 · Local-mini 0.557 / 0.482.

Convention: MINI = fast pipeline check; FULL = headline numbers. Log wall-clock + GPUs so
we can compare cost (a stated contribution — we remove ++'s 3× MC-dropout).

---

## Results table

| # | date | stage/PR | config | GPUs | key change | mono IoU/mIoU | embodied IoU/mIoU | wall-clock | notes |
|---|---|---|---|---|---|---|---|---|---|
| — | — | ++ paper | — | 8×A800 | reference | 0.549/0.462 (local) | 0.522/0.436 (full) | — | targets |
| 0a | _pending_ | PR0 repro | train_mono_mini_geo | ? | baseline mini mono | | — | | pipeline validation |
| 0b | _pending_ | PR0 repro | train_embodied_mini_unc | ? | baseline mini embodied | — | | | pipeline validation |
| 0c | _pending_ | PR0 repro | train_mono_geo (full) | 4 | baseline full mono | | — | | |
| 0d | _pending_ | PR0 repro | train_embodied_unc (full) | 4 | baseline full embodied | — | | | **PR0 baseline** |

---

## Per-run entries (newest first)

### [2026-08-18] mono-mini-local-smoke — validate full pipeline on 80-scene mini
- Config: train_mono_mini_geo_local_config.py | GPUs: 3 (0,1,2) | batch 1 → eff. batch 3
- Data: local SSD cache (data_local/occscannet), 503GB RAM page cache
- **Pipeline VALIDATED end-to-end** (data→model→CUDA ops→train→eval→ckpt).
- Epoch 1 eval: geo IoU = 0.267, sem mIoU = 0.149
  per-class IoU [0.941,0.000,0.297,0.229,0.148,0.159,0.069,0.218,0.170,0.000,0.165,0.184]
  (epoch-1 numbers, expected to climb over 20 epochs; not the final baseline)
- **Speed finding:** ~4.7s/iter and did NOT improve on epoch 2 (page cache warm) →
  bottleneck is COMPUTE, not I/O. Local cache gave 11→4.7s (2.3× — that part was I/O).
  Residual = Python-heavy forward, suspected GRM per-Gaussian loop in mono_refine_layer
  (16200-pt `for` + `.item()`, ×3 refine/forward). GPU util only 9-18%.
  → At 4.7s/iter FULL repro is infeasible; must vectorize the forward first
  (already on the PR roadmap; doubles as speedup + our contribution). Profiling with py-spy.
- **py-spy confirmed hotspot** = get_kappa_guided_weight / get_projected_normals in
  mono_refine_layer.py (two inner `for n in range(16200)` with .item()/int(), GIL-bound).
  Secondary: SyncBatchNorm all_reduce + EfficientNet-B7 forward.
- **FIX (2026-08-18): vectorized both GRM loops** (advanced indexing + clamp, numerically
  equivalent). Wall-clock **4.7s → 1.25s/iter = 3.7x faster**; loss unchanged/healthy on
  resumed run (equivalence holds). Residual gap (1.25s wall vs 0.15s compute) likely
  SyncBN sync — next optimization target if needed. TODO: same vectorization for the
  embodied stage's online_refine_layer (loops + MC-dropout) before FULL embodied repro.
- **RESULT (ran to epoch 20) — INVALIDATED by a mid-run GPU-count switch.**
  Curve: peaked at epoch 9 on 3 GPUs (geo 0.502 / sem 0.396, still climbing toward the
  Local-mini target 0.557/0.482), then DEGRADED monotonically once resumed on 1 GPU
  (ep10 0.480/0.367 → ep20 0.435/0.331). Root cause: SyncBatchNorm with single-GPU
  batch_size=1 computes BN stats over ONE sample → corrupts running mean/var, model
  degrades every epoch. Transition is exactly at the 3→1 GPU switch (ep9→10).
  **NOT a code bug** (pipeline/vectorization/equivalence all valid) — an operational error.
  **RULES (locked, see server.md): never change GPU count mid-run; never single-GPU
  batch_size=1 for this model (BN breaks); keep GPU count fixed, effective batch >= 2.**
  Action: re-run mono-mini from scratch on a FIXED >=2 GPU setup (new work-dir).

## Per-run entries (older)

### Template (copy for each run)
```
### [YYYY-MM-DD] <exp-id> — <one-line goal>
- Commit: <git sha>  | Config: <config file>  | GPUs: <n> (<which>)  | grad-accum: <k>
- Change vs previous: <what>
- Command: <torchrun ...>
- Result: mono IoU/mIoU = __ / __ ; embodied IoU/mIoU = __ / __
- Per-class mIoU (if notable): <ceiling/floor/wall/... deltas>
- Look-back vs first-time (PR3+): first __ / revisited __
- Wall-clock: __ ; peak mem/GPU: __ GB
- Verdict: <beat ++? regression? next action>
```

---

## Setup milestones (pre-training)
- [x] 2026-08-13 datasets downloaded to NAS (occscannet, scene_occ, depth ckpt)
- [x] 2026-08-14 data extracted; scannet.pt (291MB) obtained; git workflow live
- [x] 2026-08-14 preprocess.py import + torch.hub cache fixed
- [x] 2026-08-17 dpt.py infer_image edit applied (depth preprocess)
- [x] 2026-08-17 MINI normals/kappas/depthanything generated (80 scenes; 7311 npy each, consistent, no NaN)
- [x] 2026-08-17 pytorch3d 0.7.2 (py38_cu113_pyt1121 wheel + fvcore/iopath) installed; CUDA ops built (local-aggregate, deformable-aggregation-ext)
- [ ] PR0 MINI reproduce (mono → embodied)  <-- IN PROGRESS: mono_mini_geo on GPU0-2
- [ ] FULL preprocess (681 scenes)
- [ ] PR0 FULL reproduce → baseline row 0d filled
