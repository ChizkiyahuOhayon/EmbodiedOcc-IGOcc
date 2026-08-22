# Embodied (online) mini training — inherits the baseline config and only sets load_from
# to OUR reproduced mono-mini checkpoint (strict=False partial load initializes the shared
# backbone/encoder of the online model). Dataset paths (data/scene_occ, data/occscannet) are
# hardcoded in the dataset class, so nothing else to override here.
#
# RULES (see server.md 7b): SyncBatchNorm here too -> fixed GPU count, >=2 GPUs, never switch
# mid-run. Launch with an explicit --work-dir.
_base_ = ['./train_embodied_mini_unc_config.py']

load_from = 'out/mono_mini_3gpu_v2/epoch_20.pth'
