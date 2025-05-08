import importlib
import torch
import os
import gc
import random
import torch.nn.functional as F

from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
from functools import partial

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.nano import apply_conv_n_deconv
from basicsr.metrics.other_metrics import compute_img_metric

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')


class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1,1)).item()
    
        r_index = torch.randperm(target.size(0)).to(self.device)
    
        target = lam * target + (1-lam) * target[r_index, :]
        input_ = lam * input_ + (1-lam) * input_[r_index, :]
    
        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments)-1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_

class ImageCleanModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageCleanModel, self).__init__(opt)

        # define network

        self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
        if self.mixing_flag:
            mixup_beta       = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
            use_identity     = self.opt['train']['mixing_augs'].get('use_identity', False)
            self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = define_network(self.opt['network_g']).to(
                self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                  self.opt['path'].get('strict_load_g',
                                                       True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()


        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            raise ValueError('pixel loss are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()


    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)


    def feed_train_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def feed_data(self, data, psf=None, ks=None, val_conv=True):
        gt = data['gt'].to(self.device)
        padding = data['padding']
        padding = torch.stack(padding).T
        otf = psf
        M = ks.shape[1]
        if val_conv:    # Apply convolution on the fly (use gt img to create lr image)
            lq, gt = apply_conv_n_deconv(gt, otf, padding, M, 0, ks=ks, ph=135, num_psf=9, sensor_h=1215, crop=False, conv=True)
            self.lq = lq[None]
            self.gt = gt[None]  # TODO check dim. 이전에는 square에서 리턴해주는거 그대로 썼는데 지금은 원래 gt 바로 써서 shape 다를수도. 이후 아래랑 합치기
            # TODO 애초에 deconv(gt) 를 gt를 위에서 if else로 받아서 한 줄로 처리 가능

        else:   # loaded npy for validaiton
            lq = data['lq'].to(self.device)
            lq, gt = apply_conv_n_deconv(lq, otf, padding, M, 0, ks=ks, ph=135, num_psf=9, sensor_h=1215, crop=False, conv=False)
            self.lq = lq[None]
            self.gt = gt
                

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        preds = self.net_g(self.lq)
        if not isinstance(preds, list):
            preds = [preds]

        self.output = preds[-1]

        loss_dict = OrderedDict()
        # pixel loss
        l_pix = 0.
        for pred in preds:
            l_pix += self.cri_pix(pred, self.gt)

        loss_dict['l_pix'] = l_pix

        l_pix.backward()
        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def pad_test(self, window_size):        
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        h,w = self.lq.size()[-2:]
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq[0], (0, mod_pad_w, 0, mod_pad_h), 'reflect')[None]
        self.nonpad_test(img)
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def nonpad_test(self, img=None):
        if img is None:
            img = self.lq
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                pred = self.net_g_ema(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
        else:
            self.net_g.eval()
            with torch.no_grad():
                pred = self.net_g(img)

            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image, psf, ks, val_conv):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image, psf, ks, val_conv)
        else:
            return 0.
        

    def pre_process(self, padding_size):
        # pad to multiplication of window_size
        self.mod_pad_h, self.mod_pad_w = 0, 0
        h,w = self.lq.size()[-2:]  # BMCHW
        if h % padding_size != 0:
            self.mod_pad_h = padding_size - h % padding_size
        if w % padding_size != 0:
            self.mod_pad_w = padding_size - w % padding_size
        self.lq = F.pad(self.lq[0], (0, self.mod_pad_w, 0, self.mod_pad_h), 'reflect')[None]

    def post_process(self):
        _, _, h, w = self.output.size()
        self.output = self.output[...,0:h - self.mod_pad_h, 0:w - self.mod_pad_w]

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image, psf, ks, val_conv):
        dataset_name = dataloader.dataset.opt['name']
        base_path = self.opt['path']['visualization']

        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
            if save_img:
                cur_other_metrics = {'ssim': 0., 'lpips': 0.}
            else:
                cur_other_metrics = None

        window_size = self.opt['val'].get('window_size', 0)

        if window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test

        cnt = 0

        for idx, val_data in enumerate(tqdm(dataloader)):
            img_name = osp.splitext(osp.basename(val_data['gt_path'][0]))[0]
            self.feed_data(val_data, psf, ks, val_conv)
            pad_for_OCB = self.opt['val'].get('padding')
            if pad_for_OCB is not None:
                self.pre_process(pad_for_OCB)
            
            torch.cuda.empty_cache()
            gc.collect()

            test()

            if pad_for_OCB is not None:
                self.post_process()

            if save_img and with_metrics and use_image:
                visuals = self.get_current_visuals(to_cpu=False)
                cur_other_metrics['ssim'] += compute_img_metric(visuals['result'][0], visuals['gt'][0], 'ssim')
                cur_other_metrics['lpips'] += compute_img_metric(visuals['result'][0], visuals['gt'][0], 'lpips').item()

            visuals = self.get_current_visuals()

            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()
            gc.collect()

            if save_img:
                if self.opt['is_train']:
                    if 'eval_only' in self.opt['train']:
                        save_img_path = osp.join(base_path + self.opt['train']['eval_name'],
                                                f'{img_name}_{current_iter}.png')
                    else:
                        save_img_path = osp.join(base_path,
                                                f'{img_name}_{current_iter}.png')
                else:
                    save_img_path = osp.join(
                        base_path,
                        f'{img_name}.png')
                    save_gt_img_path = osp.join(
                        base_path, dataset_name,
                        f'{img_name}_gt.png')
                    
                imwrite(sr_img, save_img_path)
                    
            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1

        
        # tentative for out of GPU memory
        torch.cuda.empty_cache()
        gc.collect()
        
        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]
            if save_img:
                cur_other_metrics['ssim'] /= cnt 
                cur_other_metrics['lpips'] /= cnt 

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)
        return current_metric, cur_other_metrics


    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self, to_cpu=True):
        if to_cpu:
            out_dict = OrderedDict()
            out_dict['lq'] = self.lq.detach().cpu()
            out_dict['result'] = self.output.detach().cpu()
            if hasattr(self, 'gt'):
                out_dict['gt'] = self.gt.detach().cpu()
        else:
            out_dict = OrderedDict()
            out_dict['lq'] = self.lq.detach()
            out_dict['result'] = self.output.detach()
            if hasattr(self, 'gt'):
                out_dict['gt'] = self.gt.detach()
        return out_dict

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema],
                              'net_g',
                              current_iter,
                              param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
