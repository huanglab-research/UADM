import torch
import torch.amp as amp
import torch.nn as nn
import functools
import math
import lpips
import os
import os.path as osp
import pyiqa
import numpy as np
import random
import tqdm
import torchvision
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
import matplotlib.pyplot as plt


# from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.utils import get_obj_from_str, get_root_logger, ImageSpliterTh, imwrite, tensor2img
from basicsr.utils.ocean_util import RMSEMetric, BiasMetric, R2Metric, PSNROceanMetric, SSIMOceanMetric, DeltaRMSEMetric, DeltaBiasMetric, DeltaR2Metric, DeltaPSNROceanMetric, DeltaSSIMOceanMetric, compute_geostrophic, save_patch_to_nc
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from .base_model import BaseModel
from .sr_model import SRModel
from .gan_model import GANModel
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from contextlib import nullcontext
from copy import deepcopy
from torch.nn.parallel import DataParallel
from collections import OrderedDict
from torchvision import transforms
from PIL import Image


class OceanModel(GANModel):
    """Diffusion SR model for single image super-resolution."""

    def __init__(self, opt):
        super(OceanModel, self).__init__(opt)
        self.opt = opt
        logger = get_root_logger()
        self.sf = self.opt['scale']

        # define base_diffusion
        diff_opt = self.opt['diffusion']
        self.base_diffusion = build_network(diff_opt)

        if self.opt['rank'] == 0:
            self.metrics_fr = {}
            self.metrics_nr = {}
            self.metrics_fr['psnr'] = PSNROceanMetric()
            self.metrics_fr['ssim'] = SSIMOceanMetric()
            self.metrics_fr['rmse'] = RMSEMetric()
            self.metrics_fr['r2'] = R2Metric()
            self.metrics_fr['bias'] = BiasMetric()
            self.metrics_fr['delta_psnr'] = DeltaPSNROceanMetric()
            self.metrics_fr['delta_ssim'] = DeltaSSIMOceanMetric()
            self.metrics_fr['delta_rmse'] = DeltaRMSEMetric()
            self.metrics_fr['delta_r2'] = DeltaR2Metric()
            self.metrics_fr['delta_bias'] = DeltaBiasMetric()

    def load_network(self, net, load_path, strict=True, param_key='params'):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        logger = get_root_logger()
        net = self.get_bare_model(net)
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            if param_key not in load_net and 'params' in load_net:
                param_key = 'params'
                logger.info('Loading: params_ema does not exist, use params.')
            if param_key in load_net:
                load_net = load_net[param_key]
        logger.info(f'Loading {net.__class__.__name__} model from {load_path}, with param key: [{param_key}].')
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        self._print_different_keys_loading(net, load_net, strict)
        net.load_state_dict(load_net, strict=strict)

    def init_training_settings(self):

        self.net_g.train()
        if self.net_d:
            self.net_d.train()
        train_opt = self.opt['train']

        if train_opt.get('adversarial_opt'):
            self.cri_gan = build_loss(train_opt['adversarial_opt']).to(self.device)
        else:
            self.cri_gan = None

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

        self.amp_scaler = amp.GradScaler() if self.opt['train'].get('use_fp16', False) else None


    def backward_step(self, dif_loss_wrapper, micro_lq, micro_gt, micro_strain, num_grad_accumulate, tt, current_iter):
        loss_dict = OrderedDict()

        if self.opt['train'].get('use_fp16', False):
            context = lambda: amp.autocast(device_type="cuda")
        else:
            context = nullcontext

        # -----------------
        # Generator backward
        # -----------------
        with context():
            # diffusion loss
            losses, x_t, x0_pred = dif_loss_wrapper()
            loss_dict['l_pix'] = losses['mse'].mean() / num_grad_accumulate
            l_total = loss_dict['l_pix']

        # backward G
        if self.amp_scaler is None:
            l_total.backward()
        else:
            self.amp_scaler.scale(l_total).backward()


        return loss_dict, x_t, x0_pred

    def optimize_parameters(self, current_iter):
        current_batchsize = self.lq.shape[0]
        micro_batchsize = self.opt['datasets']['train']['micro_batchsize']
        num_grad_accumulate = math.ceil(current_batchsize / micro_batchsize)

        self.optimizer_g.zero_grad()
        if self.cri_gan:
            self.optimizer_d.zero_grad()

        for jj in range(0, current_batchsize, micro_batchsize):
            micro_lq = self.lq[jj:jj+micro_batchsize,]
            micro_gt = self.gt[jj:jj+micro_batchsize,]

            last_batch = (jj+micro_batchsize >= current_batchsize)
            if self.opt['diffusion'].get('one_step', False):
                tt = torch.ones(
                    size=(micro_gt.shape[0],),
                    device=self.lq.device,
                    dtype=torch.int32,
                    ) * (self.base_diffusion.num_timesteps - 1)
            else:
                tt = torch.randint(
                        0, self.base_diffusion.num_timesteps,
                        size=(micro_gt.shape[0],),
                        device=self.lq.device,
                        )

            with torch.no_grad():
                micro_lq_bicubic = torch.nn.functional.interpolate(
                        micro_lq, scale_factor=self.sf, mode='bicubic', align_corners=False,
                        )
                if self.opt['diffusion']['un'] > 0:
                    raw = compute_geostrophic(micro_lq_bicubic, self.lat, self.lon)
                    raw = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
                    raw = raw.abs()
                    p = raw / (raw.amax(dim=(-1, -2, -3), keepdim=True) + 1e-8)
                    b_un = self.opt['diffusion']['min_noise']
                    micro_uncertainty = torch.clamp(b_un + (1 - b_un) * p, 1e-6, 1 - 1e-6)
                else:
                    micro_uncertainty = torch.full_like(micro_lq_bicubic, 0.5)

            # n
            noise = torch.randn_like(micro_lq_bicubic)

            lq_cond = nn.PixelUnshuffle(self.sf)(micro_lq_bicubic)

            model_kwargs={'lq':lq_cond,} if self.opt['network_g']['params']['cond_lq'] else None
            compute_losses = functools.partial(
                self.base_diffusion.training_losses,
                self.net_g,
                micro_gt,
                micro_lq_bicubic,
                micro_uncertainty,
                tt,
                model_kwargs=model_kwargs,
                noise=noise,
            )

            if last_batch or self.opt['num_gpu'] <= 1:
                losses, x_t, x0_pred = self.backward_step(compute_losses, micro_lq, micro_gt, micro_uncertainty, num_grad_accumulate, tt, current_iter)
            else:
                with self.net_g.no_sync():
                    losses, x_t, x0_pred = self.backward_step(compute_losses, micro_lq, micro_gt, micro_uncertainty, num_grad_accumulate, tt, current_iter)

            
        if self.opt['train'].get('use_fp16', False):
            self.amp_scaler.step(self.optimizer_g)
            if self.cri_gan:
                self.amp_scaler.step(self.optimizer_d)
            self.amp_scaler.update()
        else:
            self.optimizer_g.step()
            if self.cri_gan:
                self.optimizer_d.step()

        self.net_g.zero_grad()
        if self.cri_gan:
            self.net_d.zero_grad()

        self.log_dict = self.reduce_loss_dict(losses)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def sample_func(self, y0, noise_repeat=False):
        desired_min_size = self.opt['val']['desired_min_size']
        ori_h, ori_w = y0.shape[2:]
        if not (ori_h % desired_min_size == 0 and ori_w % desired_min_size == 0):
            flag_pad = True
            pad_h = (math.ceil(ori_h / desired_min_size)) * desired_min_size - ori_h
            pad_w = (math.ceil(ori_w / desired_min_size)) * desired_min_size - ori_w
            y0 = F.pad(y0, pad=(0, pad_w, 0, pad_h), mode='reflect')
        else:
            flag_pad = False

        y_bicubic = torch.nn.functional.interpolate(
            y0, scale_factor=self.sf, mode='bicubic', align_corners=False,
            )

        if self.opt['diffusion']['un'] > 0:
            micro_strain = compute_geostrophic(y_bicubic, self.lat, self.lon)
            un_max = self.opt['diffusion']['un']
            b_un = self.opt['diffusion']['min_noise']
            un = torch.abs(micro_strain).clamp_(0., un_max) / un_max
            un = b_un + (1 - b_un) * un
        else:
            un = torch.ones_like(y_bicubic)

        lq_cond = nn.PixelUnshuffle(self.sf)(y_bicubic)

        model_kwargs = {'lq': lq_cond,} if self.opt['network_g']['params']['cond_lq'] else None
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            net = self.net_g_ema
        else:
            self.net_g.eval()
            net = self.net_g
        results = self.base_diffusion.ddim_sample_loop(
                y=y_bicubic,
                un=un,
                model=net,
                first_stage_model=None,
                noise=None,
                noise_repeat=noise_repeat,
                # clip_denoised=(self.autoencoder is None),
                clip_denoised=False,
                denoised_fn=None,
                model_kwargs=model_kwargs,
                progress=False,
                one_step=self.opt['diffusion'].get('one_step', False),
                )    

        if flag_pad:
            results = results[:, :, :ori_h*self.sf, :ori_w*self.sf]

        return results.clamp_(-1.0, 1.0)

    def test(self):

        def _process_per_image(im_lq_tensor):
            if im_lq_tensor.shape[2] > self.opt['val']['chop_size'] or im_lq_tensor.shape[3] > self.opt['val']['chop_size']:
                im_spliter = ImageSpliterTh(
                        im_lq_tensor,
                        self.opt['val']['chop_size'],
                        stride=self.opt['val']['chop_stride'],
                        sf=self.opt['scale'],
                        extra_bs=self.opt['val']['chop_bs'],
                        )
                for im_lq_pch, index_infos in im_spliter:
                    im_sr_pch = self.sample_func(
                            (im_lq_pch - 0.5) / 0.5,
                            noise_repeat=self.opt['val']['noise_repeat'],
                            )     # 1 x c x h x w, [-1, 1]
                    im_spliter.update(im_sr_pch, index_infos)
                im_sr_tensor = im_spliter.gather()
            else:
                im_sr_tensor = self.sample_func(
                        (im_lq_tensor - 0.5) / 0.5,
                        noise_repeat=self.opt['val']['noise_repeat'],
                        )     # 1 x c x h x w, [-1, 1]

            im_sr_tensor = im_sr_tensor * 0.5 + 0.5
            return im_sr_tensor
        
        self.output = _process_per_image(self.lq)


    @torch.no_grad()
    def feed_data(self, data, training=True):
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images.
        """
        # for paired training or validation
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        else:
            self.gt = None
        if 'mean' in data:
            self.mean = data['mean'].to(self.device)
        if 'std' in data:
            self.std = data['std'].to(self.device)
        if 'lat' in data:
            self.lat = data['lat'].to(self.device)
        if 'lon' in data:
            self.lon = data['lon'].to(self.device)

        if training:
            self.lq = (self.lq - 0.5) / 0.5
            self.gt = (self.gt - 0.5) / 0.5

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)
        logger = get_root_logger()
        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: [] for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)

        if use_pbar:
            pbar = tqdm.tqdm(total=len(dataloader), unit='image')

        num_img = 0

        for idx, val_data in enumerate(dataloader):
            self.feed_data(val_data, training=False)
            self.test()
            visuals = self.get_current_visuals()

            if with_metrics:
                # calculate metrics
                if visuals['gt'] is not None:
                    for name, metric in self.metrics_fr.items():
                        self.metric_results[name].extend(metric(visuals['sr'], visuals['gt']))
                for name, metric in self.metrics_nr.items():
                    self.metric_results[name].extend(metric(visuals['sr']))
            if 'gt' in visuals:
                del self.gt

            for ii in range(len(val_data['gt_path'])):
                image_name_ext = os.path.basename(val_data['gt_path'][ii])
                logger_parts = [image_name_ext]
                for metric in self.metric_results.keys():
                    metric_value = self.metric_results[metric][num_img]
                    logger_parts.append(f"{metric}: {metric_value:<4.4f}")
                logger_parts = " | ".join(logger_parts)
                num_img += 1


                if save_img:
                    logger.info(logger_parts)
                    save_patch_to_nc(
                        model= 'UDAM',
                        filename=str(image_name_ext),
                        patch=visuals['sr'][ii],
                        lat=val_data['lat'][ii],
                        lon=val_data['lon'][ii]
                    )

            # tentative for out of GPU memory
            del self.lq
            del self.output
            del visuals
            torch.cuda.empty_cache()

            if use_pbar:
                pbar.update(1)

        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] = np.stack(self.metric_results[metric]).mean()
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
            del self.metric_results
    def get_current_visuals(self):
        lq_max = 35.0
        lq_min = -2.0
        self.lq = (self.lq) * (lq_max - lq_min) + lq_min
        self.gt = (self.gt) * (lq_max - lq_min) + lq_min
        self.output = (self.output) * (lq_max - lq_min) + lq_min
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.squeeze(1).detach().cpu().numpy()
        out_dict['sr'] = self.output.squeeze(1).detach().cpu().numpy()
        if hasattr(self, 'gt') and self.gt is not None:
            out_dict['gt'] = self.gt.squeeze(1).detach().cpu().numpy()
        out_dict['lat'] = self.lat.cpu().numpy()
        out_dict['lon'] = self.lon.cpu().numpy()
        return out_dict
