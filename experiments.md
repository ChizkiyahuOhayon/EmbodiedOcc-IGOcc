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
