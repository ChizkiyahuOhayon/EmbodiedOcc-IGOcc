#!/usr/bin/env bash
# Cache the MINI (80-scene) working set from NAS -> local SSD for fast training.
# The dataset reads 6-7 small files per frame over CIFS; that I/O dominates wall-clock.
# Copying the mini set to local disk once makes every epoch after the first read locally.
# Resumable (rsync). Local free needed: ~26 GB. Run from repo root.
set -euo pipefail

SRC=data/occscannet          # symlinked to NAS
DST=data_local/occscannet    # local SSD
SUBS=(posed_images gathered_data normals kappas depthanything)

mkdir -p "$DST"

# 1) scene list = union of the two mini _final splits (train==test scenes, 80 total)
scenes=$(grep -hoE 'scene[0-9]{4}_[0-9]{2}' \
           "$SRC/train_mini_final.txt" "$SRC/test_mini_final.txt" | sort -u)
n=$(echo "$scenes" | wc -l)
echo "caching $n scenes -> $DST"

# 2) split txts (dataset reads {data_path}/{phase}_mini_final.txt)
cp -f "$SRC"/train_mini_final.txt "$SRC"/test_mini_final.txt "$DST"/

# 3) per-scene copy of every dir the dataset reads
for sub in "${SUBS[@]}"; do mkdir -p "$DST/$sub"; done
i=0
for s in $scenes; do
  i=$((i+1))
  echo "[$i/$n] $s"
  for sub in "${SUBS[@]}"; do
    if [ -d "$SRC/$sub/$s" ]; then
      mkdir -p "$DST/$sub/$s"
      rsync -a "$SRC/$sub/$s/" "$DST/$sub/$s/"
    else
      echo "  [WARN] missing $SRC/$sub/$s"
    fi
  done
done

echo "done. local cache size:"
du -sh "$DST"
