# Same as train_mono_mini_geo_config.py but reads the MINI working set from the
# LOCAL SSD cache (data_local/occscannet) instead of the NAS-symlinked data/occscannet.
# Populate the cache first with: bash cache_mini_local.sh
_base_ = ['./train_mono_mini_geo_config.py']

data_path = './data_local/occscannet'
train_dataset_config = dict(data_path=data_path)
val_dataset_config = dict(data_path=data_path)
