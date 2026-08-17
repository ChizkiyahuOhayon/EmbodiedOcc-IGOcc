# Same as train_mono_mini_geo_config.py but reads the MINI working set from the
# LOCAL SSD cache (data_local/occscannet) instead of the NAS-symlinked data/occscannet.
# Populate the cache first with: bash cache_mini_local.sh
_base_ = ['./train_mono_mini_geo_config.py']

data_path = './data_local/occscannet'
train_dataset_config = dict(data_path=data_path)
val_dataset_config = dict(data_path=data_path)

# Loader perf: data is on local SSD + 503GB RAM page cache, so feed the GPUs harder.
# persistent_workers keeps workers alive across the 20 epochs; prefetch deepens buffering.
train_loader_config = dict(num_workers=12, persistent_workers=True, prefetch_factor=4)
val_loader_config = dict(num_workers=4, persistent_workers=True, prefetch_factor=4)
