# -*- coding: utf-8 -*-
"""Bung CUB_200_2011.tgz thanh layout ImageFolder ma Fly-CL cho doi.

    data/cub/train/<lop>/<anh>.jpg
    data/cub/test/<lop>/<anh>.jpg

Fly-CL (`datasets/load_dataset.py`) va repo nay (`data_loader.load_raw`) deu doc
CUB bang `datasets.ImageFolder(f"{root}/cub/train/")`, nhung khong repo nao co
buoc tao ra hai thu muc do. Ban goc chi co `images/` phang cung
`train_test_split.txt` rieng.

Dung DUNG phep chia chinh thuc trong `train_test_split.txt` (5994 train / 5794
test), khong tu chia ngau nhien - neu khong thi con so khong so duoc voi bat ky
bai nao.

    python scripts/prepare_cub.py            # bung roi chia
    python scripts/prepare_cub.py --link     # dung hardlink thay vi copy
"""

import argparse
import os
import shutil
import tarfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='./data')
    p.add_argument('--tgz', default='./data/CUB_200_2011.tgz')
    p.add_argument('--link', action='store_true',
                   help='hardlink thay vi copy - nhanh hon, tiet kiem 1.1 GB')
    a = p.parse_args()

    src = os.path.join(a.root, 'CUB_200_2011')
    if not os.path.isdir(src):
        print(f"[cub] bung {a.tgz}")
        with tarfile.open(a.tgz, 'r:gz') as t:
            t.extractall(path=a.root)
    print(f"[cub] thu muc goc: {src}")

    def read(name):
        with open(os.path.join(src, name), encoding='utf-8') as f:
            return dict(line.split(None, 1) for line in f.read().splitlines() if line)

    paths = {k: v.strip() for k, v in read('images.txt').items()}
    split = {k: v.strip() for k, v in read('train_test_split.txt').items()}

    out = os.path.join(a.root, 'cub')
    n = {'train': 0, 'test': 0}
    for img_id, rel in paths.items():
        # is_training = 1 -> train. Ten thu muc lop giu nguyen cua ban goc
        # (001.Black_footed_Albatross), nen ImageFolder sap xep lop theo dung
        # thu tu so hieu.
        sub = 'train' if split[img_id] == '1' else 'test'
        cls = rel.split('/')[0]
        dst_dir = os.path.join(out, sub, cls)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, os.path.basename(rel))
        if not os.path.exists(dst):
            s = os.path.join(src, 'images', rel.replace('/', os.sep))
            (os.link if a.link else shutil.copyfile)(s, dst)
        n[sub] += 1

    ncls = len(os.listdir(os.path.join(out, 'train')))
    print(f"[cub] {n['train']} train / {n['test']} test | {ncls} lop -> {out}")
    assert (n['train'], n['test'], ncls) == (5994, 5794, 200), \
        'so anh khong khop ban chinh thuc - kiem tra lai file tai ve'


if __name__ == '__main__':
    main()
