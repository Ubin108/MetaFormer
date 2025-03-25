import cv2
cv2.setNumThreads(1)
from os import path as osp
from basicsr.utils import scandir


def paths_from_folder(folder, key):
    gt_paths = list(scandir(folder))
    paths = []
    for idx in range(len(gt_paths)):
        gt_path = gt_paths[idx]
        gt_path = osp.join(folder, gt_path)
        paths.append(
            dict([(f'{key}_path', gt_path)]))
    return paths