# DATA_SPEC — exact data layout for EmbodiedOcc2 (write code against THIS)

Two datasets, two stages. All paths are repo-relative (dirs under `data/` are symlinks to NAS).
Verified 2026-08-23 by `check_embodied_data.py` + reading the dataset classes. When a tensor
shape below is marked (verify), dump it with the one-liner in §4 before hard-coding it.

Common voxel / label conventions (both stages):
- Local voxel grid: **60 × 60 × 36**, voxel size **0.08 m**, local scene box **4.8 × 4.8 × 2.88 m**.
- Classes: `cls_dims = 13`. label **0 = ignore/unknown**, **1–11 = objects**, **12 = empty**
  (`empty_idx = 12`). In the online loader raw target uses 0=empty,255=unknown → remapped:
  `target[target==0]=12; target[target==255]=0`.
- Global scene box / pc_range: `[-51.2,-51.2,-5.0, 51.2,51.2,3.0]`.
- Gaussians: `num_anchor = 16200`, `num_anchor_init (num_pts) = 8100`. embed_dims `_dim_ = 96`.
- Image size fed to model: **480 × 640** (H×W). Depth-Anything branch input resized to 480×480,
  multiple-of-14, aspect kept.

---

## 1. MONO stage — `Scannet_Scene_OpenOccupancy_Dataset` (Local benchmark)
- File: `dataset/dataset_scannet_occ_openocc.py`
- Root: **`data/occscannet`** (config-overridable via `data_path`; local-cache config uses
  `data_local/occscannet`).
- Split lists (per phase): `data_tag='mini'` → **`{phase}_mini_final.txt`**;
  `data_tag='base'` → `{phase}_train/test.txt`. Each line = `scenexxxx_xx/xxxxx` frame id
  (mono is per-FRAME).
- MINI set: **80 scenes shared** across train_mini_final + test_mini_final; **7311 frames** per
  feature dir (normals / kappas / depthanything each have 7311 .npy).
- Per-frame files (root = `data/occscannet`):
  | file | path | content |
  |------|------|---------|
  | rgb  | `posed_images/{scene}/{idx}.jpg` | RGB |
  | depth| `posed_images/{scene}/{idx}.png` | uint16 depth, `Image.convert('I;16')`, `/1000`→meters |
  | gathered | `gathered_data/{scene}/{idx}.pkl` | keys: `cam_pose`(4×4 cam2world), `intrinsic`(3×3 or 4×4), `voxel_origin`, `target_1_4`(60×60×36) |
  | normals | `normals/{scene}/{idx}.npy` | surface normals, **(3,H,W)** (verify) |
  | kappas  | `kappas/{scene}/{idx}.npy`  | normal-confidence κ, **(1,H,W) or (H,W)** (verify) |
  | depthanything | `depthanything/{scene}/{idx}.npy` | DepthAnything-V2 pred depth (verify shape) |

## 2. EMBODIED stage — `Scannet_Online_SceneOcc_Dataset` (Embodied benchmark)
- File: `dataset/dataset_scannet_online_occ.py`
- Root: **`data/scene_occ`** (HARD-CODED as `self.occscannet_root`, not config-overridable;
  per-frame image features are read from `data/occscannet` — also hard-coded).
- Split lists: `data_tag='mini'` → **`{phase}_mini_online.txt`**; `base` → `{phase}_online.txt`.
  Each line = `scenexxxx_xx` (embodied is per-SCENE; the loader loops all frames of the scene).
- MINI set (verified): **train 64 scenes / 1920 frames · test 16 scenes / 480 frames**.
- Per-SCENE file: `data/scene_occ/global_occ_package/{scene}.pkl`, keys:
  `scene_dim`(x,y,z dims), `global_labels`(x,y,z), `global_pts`(x,y,z,3), `global_mask`(bool x,y,z),
  `valid_img_paths` (list of ABS paths carrying author prefix
  `/data1/code/wyq/gaussianindoor/indoor-gaussian-scannet/` → stripped via `os.path.relpath`,
  yielding `posed_images/{scene}/{idx}.jpg`-style relative paths). Frames sorted by int(idx).
- Per-FRAME occ label: `data/scene_occ/streme_occ_new_package/{phase}/{scene}_{idx}_new.pkl`,
  keys: `local_label`(60×60×36, then `np.transpose(1,0,2)`), `mask_in_global`.
- Per-FRAME image features REUSE the mono dirs under `data/occscannet` (derived by regex on the
  rgb path, replacing `posed_images…​.jpg` → `normals/kappas/depthanything …​.npy`):
  `normals/`, `kappas/`, `depthanything/`, and `gathered_data/{scene}/{idx}.pkl`
  (same keys as mono; supplies `cam_pose`, `intrinsic`, `voxel_origin`).
- Two occ heads: **local** `GaussianOccHeadLocal` (60×60×36) and **global**
  `GaussianOccHeadGlobal` (**200×220×90**, grid 0.08, pc_min [-51.2,-51.2,-5.0]).

## 3. Config flags that change what data is used
- MONO local cfg `train_mono_mini_geo_local_config.py`: `use_normal_constraint=True`,
  `use_fusion_guidance=True`, `use_kappa_guidance` present (kappa used).
- EMBODIED cfg `train_embodied_mini_unc_config.py`: `use_normal_constraint=True`,
  `use_fusion_guidance=True`, **`use_kappa_guidance=False`**, `with_unc=True` (MC-dropout SUS),
  `fusion_strategy='product'`, `threshold=0.3`, `num_frames=1`. Loader `num_workers=0`,
  `batch_size=1`. `load_from` = mono ckpt (strict=False partial load).
- IG-Occ target: replace the `with_unc` MC-dropout SUS + the scalar-confidence memory blend in
  `online_refine_layer.py` with information-form (inverse-covariance) fusion.

## 4. Ground-truth shape dump (run before hard-coding shapes in new code)
```python
# python - <<'PY'   (from repo root, embodiedocc env)
import numpy as np, pickle, glob
s = sorted(glob.glob('data/occscannet/normals/*/*.npy'))[0]
idx = s.replace('normals','{}'); base = s.split('/'); scene, fid = base[-2], base[-1][:-4]
for k in ['normals','kappas','depthanything']:
    a = np.load(s.replace('normals',k), allow_pickle=True)
    print(k, getattr(a,'shape',None), getattr(a,'dtype',None))
g = pickle.load(open(f'data/occscannet/gathered_data/{scene}/{fid}.pkl','rb'))
print('gathered keys', list(g.keys()))
for kk in g: 
    v=g[kk]; print(' ',kk, getattr(v,'shape',type(v).__name__))
PY
```
Fill the (verify) shapes above with this output when we start writing IG-Occ.
