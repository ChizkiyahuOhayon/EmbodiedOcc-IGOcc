# Research Plan — IG-Occ: Information-Gaussian Occupancy

**Target venue:** CVPR 2027 · **Baseline (must-beat):** EmbodiedOcc++ (ACM MM 2025) ·
**Compute:** 4× A40 (48 GB) · **Repo:** github.com/ChizkiyahuOhayon/EmbodiedOcc-IGOcc

> One-line thesis: *the online Gaussian memory should fuse re-observations by
> optimal triangulation (information-form / inverse-covariance addition), not by a
> per-Gaussian scalar blend.* One principled mechanism, single forward pass, no
> teacher, no Monte-Carlo dropout — faster **and** more accurate than EmbodiedOcc++.

---

## 1. Problem & task

Embodied 3D occupancy prediction (EmbodiedOcc, ICCV 2025): a monocular agent explores
an indoor scene; a **global 3D-Gaussian memory** is progressively updated as new posed
RGB frames arrive, then splatted to a semantic occupancy volume. Benchmark:
EmbodiedOcc-ScanNet. Metrics: Scene-Completion **IoU** + **mIoU** (12 classes).

## 2. Where the state of the art actually is (code-grounded)

| Method | Memory update rule | Uncertainty used | Cost |
|---|---|---|---|
| EmbodiedOcc | `G ← (1−θ)·ΔG + θ·G_old`, θ fixed at init | none (fixed scalar) | cheap |
| EmbodiedOcc++ (SUS) | `G ← r·ΔG + (1−r)·G_old`, r = semantic entropy | **semantic** entropy | **3× MC-dropout** |

**Confirmed by reading EmbodiedOcc2 source:**
- The update is STILL a per-Gaussian **isotropic scalar blend** applied to all 24 Gaussian
  dims — see `online_refine_layer.py:564` (`output*anchor_update_ratio + anchor`), and
  the same scalar gates scale (`:568`), opacity (`:600`), semantics (`:604`).
  ++ only changed *where the scalar comes from* (entropy vs init), not the *form* of fusion.
- The uncertainty is **semantic only** (entropy of class logits, `online_refine_layer.py:662`),
  estimated by **Monte-Carlo dropout ×3** (`:627-647`) — expensive, and it says nothing about
  *positional* confidence. A far, textureless wall with confident class but unreliable depth
  gets a LOW update ratio → its bad geometry is frozen. Backwards for geometry.
- GRM (plane reg.) contains **Python per-Gaussian loops** over 16200 points with `.item()` and
  integer pixel indexing (`mono/online_refine_layer.py:329-346, 405-433`) — slow, non-differentiable.

Result: EmbodiedOcc++ gains only ~+1 IoU/mIoU on the full embodied set (0.515→0.522 /
~0.425→0.436) and is **flat/slightly worse** on full local mIoU (0.464→0.462). **The
memory-fusion problem is unsolved — that is our opening.**

## 3. Core idea — IG-Occ

Replace the scalar blend with **information-form (inverse-covariance) fusion**, i.e. the
Best-Linear-Unbiased / Kalman update — exactly optimal multi-view triangulation:

```
Σ_after⁻¹ = Σ_before⁻¹ + Σ_new⁻¹                                   (information add)
m_after   = Σ_after (Σ_before⁻¹ m_before + Σ_new⁻¹ m_new)          (BLUE estimate)
```

`Σ_new` is an **anisotropic** covariance derived from the monocular depth posterior:
**large along the camera ray** (depth is intrinsically uncertain), **small laterally**
(pixel localization is good). This is the structure a monocular embodied task uniquely
has, and which **neither EmbodiedOcc nor ++ models**. Re-observing a Gaussian from a new
viewpoint whose ray is transverse to the old one shrinks the fused covariance — true
triangulation. The scalar blend cannot express this; it can only average.

### Why this is "simple, effective, elegant" (the mandate)
- **One** principled mechanism replaces a pile of heuristics (fixed θ, entropy r, MC-dropout,
  dual-threshold depth×kappa fusion). Not a stacked module — a *replacement*.
- **Theoretically optimal** (BLUE) → reviewer-defensible; ++ is engineering assembly.
- **Cheaper**: deletes the 3× MC-dropout; closed-form covariance; single forward pass.
- **Complementary** to GRM (geometric plane prior, kept), orthogonal axis to SUS (semantic).

### Differentiation (the reviewer-proof table)
| Axis | EmbodiedOcc θ | ++ SUS | **IG-Occ (ours)** |
|---|---|---|---|
| uncertainty type | fixed scalar | semantic entropy | **positional/geometric, anisotropic** |
| how obtained | hand-set | MC-dropout ×3 | **closed-form σ_d head, ×1** |
| fusion form | scalar blend | scalar blend | **information-form Σ⁻¹ add = triangulation** |
| multi-view optimal? | no | no | **yes** |
| cost | cheap | expensive | cheap |

## 4. Method — four lightweight components

1. **Along-ray aleatoric head** on the depth branch: predict per-pixel depth variance σ_d²
   (a 3-layer MLP; the only new trainable params).
2. **Closed-form anisotropic covariance init** for each new Gaussian: along-ray variance from
   σ_d², lateral variance from the angular pixel footprint at that depth → `Σ_new` in world frame.
3. **Information-form memory fusion** replacing the scalar blend at `online_refine_layer.py:564`
   (+ scale/opacity/semantics paths): store `Σ⁻¹` (or its Cholesky) per Gaussian in memory,
   accumulate on re-observation. Delete MC-dropout SUS.
4. **Uncertainty-aware Gaussian→voxel splatting**: modulate each Gaussian's contribution by its
   (fused) positional certainty so confident geometry dominates the volume.

Keep GRM but **vectorize** its Python per-point loops (grid_sample) — free speed + differentiability.

## 5. Headline experiment (self-writing narrative)

EmbodiedOcc's own Table 3 shows **"look-back" (re-observed) regions are WORSE than
first-time** — their fusion *hurts*. Our claim: information fusion makes **look-back > first-time**.
That single reversal is the paper's money figure. Report per-region (first-time vs revisited)
IoU/mIoU for EmbodiedOcc, ++, and IG-Occ.

## 6. Experimental plan (incremental PRs)

| PR | Content | Success signal |
|---|---|---|
| PR0 | Reproduce EmbodiedOcc++ on 4×A40 | match 0.522/0.436 (embodied full), 0.557/0.482 (local mini) |
| PR1 | σ_d aleatoric head + covariance channel plumbed into memory (no fusion yet) | no regression; σ_d correlates with depth error |
| PR2 | Information-form fusion replaces scalar blend; delete MC-dropout | **beat ++**; faster wall-clock |
| PR3 | Uncertainty-aware splatting + vectorized GRM + look-back experiment | look-back > first-time; final SOTA table |

Ablations: scalar-blend vs info-fusion; isotropic vs anisotropic Σ; with/without σ_d head;
MC-dropout cost vs ours. Always MINI first (fast loop), then FULL for headline numbers.

## 7. Targets

- **Primary:** exceed EmbodiedOcc++ embodied full **0.522 IoU / 0.436 mIoU**.
- Local mini 0.557/0.482; local full 0.549/0.462.
- Efficiency: ≤ ++ latency (we removed 3× MC-dropout), report it as a contribution.

## 8. Risks & mitigations
- *Anisotropic Σ hard to calibrate* → start isotropic-along-ray, add lateral term; supervise σ_d
  with depth-GT residual (aleatoric NLL).
- *Info-fusion unstable early in training* → warm-start from PR0 weights; clamp Σ⁻¹; fuse only
  refine_state dims first, extend to scale/opacity later.
- *4 GPUs shared* → validate on MINI with whatever GPUs are free; FULL runs when 4 A40 idle.
- *Reproduction drift on 4 vs 8 GPUs* → gradient accumulation to match effective batch 8.

## 9. Workflow
Local edit → push `origin` (EmbodiedOcc-IGOcc) → server `git pull` on 4×A40 → results →
logged in `experiments.md` → next PR. Server/data/code map in `server.md`.
