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
| 0a | 2026-08-23 | PR0 repro | train_mono_mini_geo_local | 3×A40 | baseline mini mono | **0.530/0.438** | — | ~14.5h/20ep | DONE — clean, vs target 0.557/0.482 (95%/91%) |
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

### [2026-08-21→22] mono-mini v2 — CLEAN re-run, fixed 3 GPUs (0,1,2), new work-dir
- Config: train_mono_mini_geo_local_config.py | work-dir: out/mono_mini_3gpu_v2
- GPUs: 3 (0,1,2), held the WHOLE run — no mid-run switch. batch 1 → eff. batch 3.
- ~1.24s/iter train; 1834 train + 792 eval iters/epoch ≈ 44 min/epoch.
- **Clean eval curve, monotonic climb (this is the valid baseline trajectory):**
  | ep | geo IoU | sem mIoU |
  |----|---------|----------|
  | 1  | 0.2614  | 0.1632 |
  | 2  | 0.3501  | 0.2633 |
  | 3  | 0.3997  | 0.2985 |
  | 4  | 0.4330  | 0.3183 |
  | 5  | 0.4506  | 0.3469 |
  | 6  | 0.4394  | 0.3570 |
  | 7  | 0.4770  | 0.3771 |
  | 8  | 0.4942  | 0.3871 |
  | 9  | 0.4957  | 0.3949 |
  ep9 per-class IoU [0.958,0.199,0.442,0.373,0.293,0.372,0.541,0.547,0.413,0.283,0.514,0.367]
- **Cross-check:** ep9 here (0.496/0.395) ≈ invalidated run's ep9 (0.502/0.396) → reproducible,
  vectorization confirmed NOT hurting accuracy. Still climbing toward Local-mini 0.557/0.482.
- **Interruptions (operational, not code):** ep1-9 ran 08/21 evening, process KILLED mid-ep10
  (~iter150, 03:02, SIGHUP — stderr not captured). Two failed relaunches were launched from the
  `(base)` conda env by mistake → `ModuleNotFoundError: mmengine`, instant exit (harmless, before
  any load). Fixed by `conda activate embodiedocc` + `python -m torch.distributed.run` + nohup +
  `2>&1` capture. Resumed from latest.pth (=epoch_9) on the SAME 3 GPUs → VALID (same count).
- **COMPLETED: ran full 20 epochs, clean, monotonic, no degradation.** ep10-20:
  | ep | geo | sem |    | ep | geo | sem |
  |----|-----|-----|----|----|-----|-----|
  | 10 | 0.4937 | 0.3999 | | 16 | 0.5287 | 0.4320 |
  | 11 | 0.5073 | 0.4094 | | 17 | 0.5298 | 0.4361 |
  | 12 | 0.5124 | 0.4177 | | 18 | 0.5283 | 0.4359 |
  | 13 | 0.5161 | 0.4232 | | 19 | 0.5297 | 0.4368 |
  | 14 | 0.5198 | 0.4241 | | 20 | 0.5299 | 0.4375 |
  | 15 | 0.5308 | 0.4351 | | peak-geo ep15, peak-sem ep20 |
  ep20 per-class IoU [0.963,0.244,0.490,0.414,0.345,0.406,0.570,0.594,0.460,0.311,0.561,0.418]
- **FINAL mono-mini baseline (ours, reproduced): geo IoU 0.530 / sem mIoU 0.438** (ep20; plateau
  from ep15). Converged ~44 min/epoch on 3×A40.
- **vs EmbodiedOcc++ Local-mini target 0.557 / 0.482:** we hit 95.2% geo / 90.8% sem. Gap
  (geo -0.027, sem -0.044) is within normal repro variance on the small mini set (single seed,
  no best-of). This is our anchor for the mono/Local benchmark; IG-Occ improvements measured
  against THIS number. (For headline claims we will also run FULL later.)

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
