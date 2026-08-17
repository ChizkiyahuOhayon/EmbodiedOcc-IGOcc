"""Numerical-equivalence check for the GRM vectorization (mono_refine_layer.py).

Proves the vectorized get_projected_normals / get_kappa_guided_weight produce the
EXACT same output as the original per-point Python loops, on random inputs that mimic
what project_points_to_image returns (valid points are in-bounds; invalid points may be
out-of-bounds). Run: python check_grm_equiv.py   (expects max diff 0.0)
"""
import torch

torch.manual_seed(0)
N, H, W = 16200, 480, 640
# hyperparams (arbitrary but fixed, as in the layer)
kappa_low, kappa_high = 3.0, 30.0
min_w, max_w = 0.2, 1.0

# --- synthesize projection outputs like project_points_to_image ---
# pixel coords: most in-bounds, some out-of-bounds; valid_mask true only when in-bounds
px = torch.empty(N).uniform_(-50, W + 50)
py = torch.empty(N).uniform_(-50, H + 50)
pixel_coords = torch.stack([px, py], dim=-1)          # [N,2] float
in_bounds = (px >= 0) & (px < W) & (py >= 0) & (py < H)
# also randomly drop some in-bounds points (as depth<=0 would), like the real valid_mask
valid_mask = in_bounds & (torch.rand(N) > 0.1)         # [N] bool

kappa_map = torch.rand(1, 1, H, W) * 40.0              # [B,1,H,W]
normals   = torch.randn(1, 3, H, W)                    # [B,3,H,W]


# ---------- ORIGINAL loop implementations (verbatim logic) ----------
def normals_loop():
    batch = []
    for n in range(N):
        if valid_mask[n]:
            x = int(pixel_coords[n, 0]); y = int(pixel_coords[n, 1])
            batch.append(normals[0, :, y, x])
        else:
            batch.append(torch.zeros(3))
    return torch.stack(batch).unsqueeze(0)             # [1,N,3]


def kappa_loop():
    batch = []
    for n in range(N):
        if valid_mask[n]:
            x = int(pixel_coords[n, 0]); y = int(pixel_coords[n, 1])
            kv = kappa_map[0, 0, y, x].item()
            if kv <= kappa_low:
                w = min_w
            elif kv >= kappa_high:
                w = max_w
            else:
                r = (kv - kappa_low) / (kappa_high - kappa_low)
                w = min_w + r * (max_w - min_w)
        else:
            w = min_w
        batch.append(w)
    return torch.tensor(batch).unsqueeze(0)            # [1,N]


# ---------- NEW vectorized implementations (as committed) ----------
def normals_vec():
    x = pixel_coords[:, 0].long().clamp(0, W - 1)
    y = pixel_coords[:, 1].long().clamp(0, H - 1)
    normal = normals[0, :, y, x].transpose(0, 1)
    normal = normal * valid_mask.unsqueeze(-1).to(normal.dtype)
    return normal.unsqueeze(0)


def kappa_vec():
    x = pixel_coords[:, 0].long().clamp(0, W - 1)
    y = pixel_coords[:, 1].long().clamp(0, H - 1)
    kv = kappa_map[0, 0, y, x]
    r = ((kv - kappa_low) / (kappa_high - kappa_low)).clamp(0.0, 1.0)
    w = min_w + r * (max_w - min_w)
    w = torch.where(valid_mask, w, torch.full_like(w, float(min_w)))
    return w.unsqueeze(0)


nd = (normals_loop() - normals_vec()).abs().max().item()
kd = (kappa_loop()  - kappa_vec()).abs().max().item()
print(f"get_projected_normals   max|loop - vec| = {nd:.3e}")
print(f"get_kappa_guided_weight max|loop - vec| = {kd:.3e}")
print("EQUIVALENT" if max(nd, kd) < 1e-6 else "!!! MISMATCH !!!")
