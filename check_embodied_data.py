"""Fast data-completeness probe for the EMBODIED (online) mini set.

Reads data/scene_occ/{phase}_mini_online.txt, opens each scene's global_occ_package pkl to
get its frame list, and checks (os.path.exists only, no heavy compute) that every per-frame
file the online dataloader needs is present:
  - rgb (posed_images/*.jpg) + depth (*.png)
  - normals / kappas / depthanything  (.npy, under data/occscannet, via regex on rgb path)
  - gathered_data/{scene}/{idx}.pkl   (under data/occscannet)
  - streme_occ_new_package/{phase}/{scene}_{idx}_new.pkl (under data/scene_occ)
Run from repo root:  python check_embodied_data.py
"""
import os, pickle, re, sys

root = 'data/scene_occ'
prefix = '/data1/code/wyq/gaussianindoor/indoor-gaussian-scannet/'


def check(phase):
    lst = f'{root}/{phase}_mini_online.txt'
    if not os.path.exists(lst):
        print(f'[{phase}] MISSING split list: {lst}')
        return
    scenes = [l.strip() for l in open(lst) if l.strip()]
    print(f'[{phase}] {len(scenes)} scenes from {lst}')
    miss = {k: 0 for k in ['pkg', 'streme', 'rgb', 'depth_png', 'normals', 'kappas', 'depthany', 'gathered']}
    ex_missing = []
    total_frames = 0
    for name in scenes:
        pkg = f'{root}/global_occ_package/{name}.pkl'
        if not os.path.exists(pkg):
            miss['pkg'] += 1
            ex_missing.append(pkg)
            continue
        d = pickle.load(open(pkg, 'rb'))
        img_paths = [os.path.relpath(p, prefix) for p in d['valid_img_paths']]
        for rgb in img_paths:
            total_frames += 1
            img_idx = rgb.split('/')[-1].split('.')[0]
            depth_png = rgb.replace('.jpg', '.png')
            normal = re.sub(r'posed_images(.*?)\.jpg$', r'normals\1.npy', rgb)
            kappa = re.sub(r'posed_images(.*?)\.jpg$', r'kappas\1.npy', rgb)
            depthany = re.sub(r'posed_images(.*?)\.jpg$', r'depthanything\1.npy', rgb)
            gathered = f'data/occscannet/gathered_data/{name}/{img_idx}.pkl'
            streme = f'{root}/streme_occ_new_package/{phase}/{name}_{img_idx}_new.pkl'
            for key, p in [('rgb', rgb), ('depth_png', depth_png), ('normals', normal),
                           ('kappas', kappa), ('depthany', depthany),
                           ('gathered', gathered), ('streme', streme)]:
                if not os.path.exists(p):
                    miss[key] += 1
                    if len(ex_missing) < 12:
                        ex_missing.append(p)
    print(f'  total frames: {total_frames}')
    print(f'  MISSING counts: {miss}')
    if any(miss.values()):
        print(f'  first missing examples:')
        for p in ex_missing[:12]:
            print(f'    {p}')
    else:
        print('  >>> ALL PRESENT <<<')
    return miss


if __name__ == '__main__':
    t = check('train')
    v = check('test')
    ok = t is not None and v is not None and not any(t.values()) and not any(v.values())
    print('\nRESULT:', 'EMBODIED MINI DATA COMPLETE' if ok else 'GAPS FOUND (see above)')
    sys.exit(0 if ok else 1)
