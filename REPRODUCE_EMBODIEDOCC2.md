# EmbodiedOcc++ (EmbodiedOcc2) Reproduction Runbook — 4×A40

Target numbers to match (from the ACM MM 2025 paper):
- **Embodied full**: 0.522 IoU / 0.436 mIoU (Table 2)
- Embodied-mini: 0.529 / 0.437
- Local full (Occ-ScanNet): 0.549 / 0.462 · Local-mini: 0.557 / 0.482 (Table 1)

This is our **must-beat baseline**. We build our method (IG-Occ) on THIS repo.

Server: `smbu@ubuntu-server` · repo target `~/dy/EmbodiedOcc2` · `$NAS=/home/smbu/dy/nas/embodiedocc` · 4×A40 (48 GB).

> ⚠️ EmbodiedOcc++ needs THREE things the original EmbodiedOcc did not:
> (1) a surface-normal model + `scannet.pt` ckpt, (2) a `preprocess.py` pass that
> writes `normals/*.npy` + `kappas/*.npy`, (3) `pytorch3d`. All covered below.

---

## Part A — Clone repo + vendored sub-repos

```bash
cd ~/dy
# main repo (proxy fallback if github is slow: prefix https://ghfast.top/)
git clone --recursive https://github.com/PKUHaoWang/EmbodiedOcc2.git
cd EmbodiedOcc2

# vendored INTO repo root (paths are resolved via os.getcwd())
git clone https://github.com/DepthAnything/Depth-Anything-V2.git
git clone https://github.com/lukemelas/EfficientNet-PyTorch.git
git clone https://github.com/baegwangbin/surface_normal_uncertainty.git   # NEW for GRM
```

Then edit **Depth-Anything-V2/metric_depth/depth_anything_v2/dpt.py** — replace `infer_image`
in class `DepthAnythingV2` with:
```python
def infer_image(self, image, h_, w_, input_size=518):
    depth = self.forward(image)
    depth = F.interpolate(depth[:, None], (h_, w_), mode="bilinear", align_corners=True)[0, 0]
    return depth
```

---

## Part B — Conda env (same as EmbodiedOcc + pytorch3d)

```bash
conda create -n embodiedocc python=3.8.19 -y && conda activate embodiedocc
pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 --index-url https://download.pytorch.org/whl/cu113
pip install openmim==0.3.9
mim install mmcv==2.0.1 && mim install mmdet==3.0.0 && mim install mmsegmentation==1.2.2 && mim install mmdet3d==1.1.1
pip install spconv-cu114==2.3.6 timm vtk==9.0.1
pip install -r requirements.txt

# NEW: pytorch3d (knn_points in the refine layers). For torch1.12.1+cu113 / py3.8,
# build from source pinned to a compatible tag if the wheel route fails:
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.2"

# Custom CUDA ops (build on a GPU node; nvcc must match torch CUDA)
cd model/encoder/gaussianformer/ops && pip install -e . && cd -
cd model/head/gaussian_occ_head/ops/localagg && pip install -e . && cd -
```

---

## Part C — Checkpoints

Two DIFFERENT models:

| ckpt | put at | source | used by |
|---|---|---|---|
| `finetune_scannet_depthanythingv2.pth` (1.1 G, we have it) | `checkpoints/` | YkiWu/EmbodiedOcc (already on NAS) | depth branch + preprocess depth |
| `scannet.pt` (normal/kappa NNET) | `checkpoints/` | `surface_normal_uncertainty` repo (Bae ICCV'21) release / its Google-Drive link | preprocess normals+kappas |

```bash
cd ~/dy/EmbodiedOcc2 && mkdir -p checkpoints
ln -s $NAS/checkpoints/finetune_scannet_depthanythingv2.pth checkpoints/finetune_scannet_depthanythingv2.pth
# scannet.pt: download from surface_normal_uncertainty (its README links a scannet.pt);
#   place at checkpoints/scannet.pt  (preprocess.py:27 expects exactly this path)
```

---

## Part D — Data symlinks

Occ-ScanNet and EmbodiedOcc-ScanNet are already downloaded+extracted on NAS
(see ENVIRONMENT.md). Configs use RELATIVE `./data/occscannet` etc., so symlink:
```bash
cd ~/dy/EmbodiedOcc2 && mkdir -p data
ln -s $NAS/occscannet data/occscannet
ln -s $NAS/scene_occ  data/scene_occ
# expected: data/occscannet/{posed_images, gathered_data, *_final.txt}
#           data/scene_occ/{global_occ_package, streme_occ_new_package, *_online.txt}
```

> If the `*_final.txt` / `*_online.txt` split files are missing from the datasets,
> they ship with THIS repo's `data/` — check `~/dy/EmbodiedOcc2/data/` and copy/link.

---

## Part E — Hardcoded paths

GOOD NEWS: the `/data1/.../model_true.pth` in `model/depthbranch/depthbranch.py:26`
is **DEAD CODE**. `GaussianDepthBranch` is never instantiated (configs pass only
`flag_depthbranch=True`, which builds an EfficientNet-B7 backbone in the segmentor;
they never pass `depthbranch=dict(...)`, and the segmentor's depthanything load block
is commented out). The `load_state_dict(model_true.pth)` lives in `__init__`, which
never runs → we do NOT need model_true.pth and do NOT edit depthbranch.py.

The model's depth comes from PRECOMPUTED `depthanything/*.npy`, produced by
`preprocess.py` DepthPrior (**vitb**, max_depth=20, loads
`finetune_scannet_depthanythingv2.pth['model']` with `module.` prefix stripped —
preprocess.py:74-83). That checkpoint is the ONLY live use of the depth ckpt.
→ sanity-check it is a vitb checkpoint before the long preprocess run (see Part F).

config `load_from` (both `train_embodied_unc_config.py:18` and
`train_embodied_mini_unc_config.py:18`) = `None` → set to the Stage-1 mono checkpoint
AFTER Stage 1 finishes.

---

## Part F — Preprocess: generate normals + kappas (NEW, required by GRM)

`preprocess.py` runs the NNET normal model over posed images and writes
`normals/*.npy` + `kappas/*.npy` as siblings of `posed_images/`
(dataset reads them via regex `posed_images→normals` / `→kappas`).

```bash
cd ~/dy/EmbodiedOcc2
python preprocess.py --features all      # writes data/occscannet/{normals,kappas}
```
Verify a sample exists and shapes look right:
```bash
find data/occscannet/normals -name '*.npy' | head    # normal maps
find data/occscannet/kappas  -name '*.npy' | head    # curvature maps
```
> TODO to confirm on first run: whether the ONLINE (embodied) dataset needs
> normals/kappas generated for `scene_occ` images too, or reuses occscannet's.
> `preprocess.py` output dirs (lines ~208-209) point at `./data/occscannet/*`.
> If the embodied dataloader errors on a missing normals path, extend preprocess
> to that image root.

Also run the repo's own preprocess if the readme asks (`python preprocess.py`):
follow readme.md "preprocess dataset for efficient training".

---

## Part G — Reproduce (MINI first, then FULL)

Config names differ from the original repo (note `_geo_` / `_unc_`):

### G1. MINI (validate the whole pipeline + our loop fast)
```bash
conda activate embodiedocc && cd ~/dy/EmbodiedOcc2

# Stage 1: local/mono (mini) -> workdir checkpoint
torchrun --nproc_per_node=4 train_mono.py --py-config config/train_mono_mini_geo_config.py

# set train_embodied_mini_unc_config.py:18  load_from -> workdir/<mono_mini>/latest.pth
# Stage 2: embodied (mini) — designed for 4 GPUs
torchrun --nproc_per_node=4 train_embodied.py --py-config config/train_embodied_mini_unc_config.py
```

### G2. FULL (headline numbers; author used 8 GPUs → we use 4)
```bash
torchrun --nproc_per_node=4 train_mono.py --py-config config/train_mono_geo_config.py
# set train_embodied_unc_config.py:18  load_from -> workdir/<mono>/latest.pth
torchrun --nproc_per_node=4 train_embodied.py --py-config config/train_embodied_unc_config.py
```

Notes:
- `batch_size=1`/GPU fixed. Author 8 GPU → effective batch halves on 4×A40. If reproduced
  mIoU lands >~0.5 below target, add gradient accumulation to match effective batch 8.
- A40 48 GB: memory not the constraint; expect ~1.5–2× wall-clock vs their 8-GPU runs.
- Report back embodied **IoU / mIoU** (+ mono IoU/mIoU). We log it as the PR0 baseline.

---

## Setup checklist
- [ ] clone EmbodiedOcc2 + Depth-Anything-V2 + EfficientNet-PyTorch + surface_normal_uncertainty
- [ ] edit dpt.py infer_image
- [ ] conda env + torch + MMLab + spconv + **pytorch3d** + CUDA ops
- [ ] checkpoints/finetune_scannet_depthanythingv2.pth (symlink) + **scannet.pt** (download)
- [ ] symlink data/occscannet + data/scene_occ; confirm split txt files present
- [ ] fix depthbranch.py:26 (with `['model']`)
- [ ] `preprocess.py --features all` → normals/ + kappas/ ; verify samples
- [ ] MINI mono → MINI embodied (pipeline validated)
- [ ] FULL mono → FULL embodied → record 0.522/0.436-level baseline (PR0)
