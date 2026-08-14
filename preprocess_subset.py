"""Preprocess ONLY the scenes referenced by given split .txt files.

EmbodiedOcc2's preprocess.py globs all 1513 posed_images scenes (151300 frames);
Occ-ScanNet's full split touches only 681 scenes (mini: 80). This runs the same
NNET normal/kappa + DepthAnything depth models but restricted to the scenes in the
splits you pass, writing normals/ kappas/ depthanything/ (resumable: skips existing).

Handles both split formats:
  - final txt  : gathered_data/sceneXXXX_XX/NNNNN.pkl  (frame-level)
  - online txt : sceneXXXX_XX                          (scene-level)

Usage (run from repo root, needs checkpoints/scannet.pt + finetune_scannet_depthanythingv2.pth):
  # mini (validate pipeline fast):
  python preprocess_subset.py --features all \
      --splits data/occscannet/train_mini_final.txt data/occscannet/test_mini_final.txt \
               data/scene_occ/train_mini_online.txt data/scene_occ/test_mini_online.txt
  # full:
  python preprocess_subset.py --features all \
      --splits data/occscannet/train_final.txt data/occscannet/test_final.txt \
               data/scene_occ/train_online.txt data/scene_occ/test_online.txt
"""
import os
import re
import glob
import argparse
import torch
from preprocess import NormalKappaPrior, DepthPrior, process_scene

SCENE_RE = re.compile(r'scene\d+_\d+')


def load_scene_set(txt_files):
    scenes = set()
    for tf in txt_files:
        with open(tf) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                m = SCENE_RE.search(line)
                if m:
                    scenes.add(m.group(0))
    return scenes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', default='all',
                    choices=['depth', 'normal', 'kappa', 'all'])
    ap.add_argument('--splits', nargs='+', required=True,
                    help='split .txt files whose scenes should be processed')
    ap.add_argument('--input_base', default='./data/occscannet/posed_images')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    scenes = sorted(load_scene_set(args.splits))
    print(f'{len(scenes)} distinct scenes to process from {len(args.splits)} split file(s)')

    models = {}
    if args.features in ['normal', 'kappa', 'all']:
        print('Initializing normal/kappa model...')
        models['normal_kappa'] = NormalKappaPrior(device).to(device).eval()
    if args.features in ['depth', 'all']:
        print('Initializing depth model...')
        models['depth'] = DepthPrior(device).to(device).eval()

    output_dirs = {
        'depth': './data/occscannet/depthanything',
        'normal': './data/occscannet/normals',
        'kappa': './data/occscannet/kappas',
    }
    for ft, od in output_dirs.items():
        if args.features in [ft, 'all']:
            os.makedirs(od, exist_ok=True)

    missing = 0
    for i, scene in enumerate(scenes):
        sf = os.path.join(args.input_base, scene)
        if not os.path.isdir(sf):
            print(f'  [WARN] missing posed_images dir: {sf}')
            missing += 1
            continue
        print(f'[{i + 1}/{len(scenes)}] {scene}')
        process_scene(sf, models, output_dirs, args.features)

    print(f'\nDone. processed {len(scenes) - missing} scenes, {missing} missing.')


if __name__ == '__main__':
    main()
