from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paths_from_folder
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding
from natsort import natsorted
import random
import numpy as np
import torch
import cv2
import os
import random


class Dataset_PaddedImage(data.Dataset):
    """Padded image dataset for image restoration.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            io_backend (dict): IO backend type and other kwarg.
            gt_size (int): Cropped patched size for gt patches.
            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_PaddedImage, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        
        self.gt_folder = opt['dataroot_gt']
        self.paths = paths_from_folder(self.gt_folder, 'gt')
            
        self.sensor_size = opt['sensor_size']
        self.psf_size = opt['psf_size']
        self.padded_size = self.sensor_size + 2 * self.psf_size

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt = padding(img_gt, gt_size)   # h,w,c
            orig_h, orig_w, _ = img_gt.shape

            # Fit one axis to sensor height (width)
            longer = max(orig_h, orig_w)
            scale = float(longer / self.sensor_size)
            resolution = (int(orig_w / scale), int(orig_h / scale))
            img_gt = cv2.resize(img_gt, resolution, interpolation=cv2.INTER_LINEAR) # sensor_size,x,3   or y,sensor_size,3 where x,y <= sensor_size
        
        resized_h, resized_w, _ = img_gt.shape
        # add padding
        pad_h = self.padded_size - resized_h
        pad_w = self.padded_size - resized_w
        pad_l = pad_r = pad_w // 2
        if pad_w % 2:
            pad_r += 1
        pad_t = pad_b = pad_h // 2
        if pad_h % 2:
            pad_b += 1
        img_gt = np.pad(img_gt, ((pad_t, pad_b), (pad_l, pad_r), (0,0))) # padded_size,padded_size,3
            
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor(img_gt, bgr2rgb=True,
                                    float32=True)

        return {
            'gt': img_gt,
            'gt_path': gt_path,
            'padding': (pad_t-self.psf_size, pad_b-self.psf_size, pad_l-self.psf_size, pad_r-self.psf_size) 
        }

    def __len__(self):
        return len(self.paths)

class Dataset_PaddedImage_npy(data.Dataset):
    # validation only
    def __init__(self, opt):
        super(Dataset_PaddedImage_npy, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        self.lq_paths = natsorted(os.listdir(self.lq_folder))
        self.gt_paths = natsorted(os.listdir(self.gt_folder))

        self.sensor_size = opt['sensor_size']
        self.psf_size = opt['psf_size']
        self.padded_size = self.sensor_size + 2 * self.psf_size


    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.gt_paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = f"{self.gt_folder}/{self.gt_paths[index]}"
        lq_path = f"{self.lq_folder}/{self.lq_paths[index]}"
        assert os.path.basename(gt_path).split(".")[0] == os.path.basename(lq_path).split(".")[0]
        
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        img_lq = torch.tensor(np.load(lq_path))   # 1,1,81,3,405,405
        
        resized_h, resized_w, _ = img_gt.shape
        pad_h = self.padded_size - resized_h
        pad_w = self.padded_size - resized_w
        pad_l = pad_r = pad_w // 2
        if pad_w % 2:
            pad_r += 1
        pad_t = pad_b = pad_h // 2
        if pad_h % 2:
            pad_b += 1

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor(img_gt, bgr2rgb=True,
                                    float32=True)

        return {
            'gt': img_gt,
            'lq': img_lq,
            'lq_path': lq_path,
            'gt_path': gt_path,
            'padding': (pad_t-self.psf_size, pad_b-self.psf_size, pad_l-self.psf_size, pad_r-self.psf_size) 
        }

    def __len__(self):
        return len(self.gt_paths)
