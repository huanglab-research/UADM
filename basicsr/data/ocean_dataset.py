import random
import math
import numpy as np
import torch.utils.data as data
import os
import csv
import torch
import netCDF4 as nc
import cv2
from basicsr.utils.registry import DATASET_REGISTRY
@DATASET_REGISTRY.register()
class OceanDataset(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/H for SISR.
    # If only "paths_H" is provided, sythesize bicubicly downsampled L on-the-fly.
    # -----------------------------------------
    # e.g., SRResNet
    # -----------------------------------------
    '''

    def __init__(self, opt):
        super(OceanDataset, self).__init__()
        self.opt = opt
        self.data_info = []
        self.lq_dir = opt['dataroot_lq']
        self.gt_dir = opt['dataroot_gt']
        with open(opt['meta_info_file'], 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                self.data_info.append(row)

    def __getitem__(self, index):
        row = self.data_info[index]

        lq_file = os.path.join(self.lq_dir, row['lq_patch_filename'])
        gt_file = os.path.join(self.gt_dir, row['gt_patch_filename'])
        lq_data, lq_lat, lq_lon = self.read_nc_file(lq_file, variable='sst', time=0)
        gt_data, gt_lat, gt_lon = self.read_nc_file(gt_file, variable='sst', time=0)
        lq_max = 35.0
        lq_min = -5.0
        lq_data = (lq_data - lq_min) / (lq_max - lq_min)
        gt_data = (gt_data - lq_min) / (lq_max - lq_min)
        lq_data = torch.tensor(lq_data[None], dtype=torch.float32)
        gt_data = torch.tensor(gt_data[None], dtype=torch.float32)
        gt_lat = torch.tensor(gt_lat, dtype=torch.float32).unsqueeze(0).unsqueeze(2)
        gt_lon = torch.tensor(gt_lon, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
        return {'lq': lq_data, 'gt': gt_data, 'max': lq_max, 'min': lq_min, 'lat': gt_lat, 'lon': gt_lon, 'gt_path': gt_file}

    def __len__(self):
        return len(self.data_info)

    def mod_pad(self, data, window):
        data = data.copy()
        h, w = data.shape[0], data.shape[1]
        pad_h, pad_w = window-h%window, window-w%window
        padded_matrix = np.pad(data, pad_width=((0, pad_h), (0, pad_w)), mode='constant',
                                   constant_values=0)
        return padded_matrix

    def read_patch(self, data, y_range, x_range, lat, lon):
        patch_data = data[0, y_range[0]:y_range[1]+1, x_range[0]:x_range[1]+1]
        patch_lat = (lat[y_range[0]], lat[y_range[1]])
        patch_lon = (lon[x_range[0]], lon[x_range[1]])

        return patch_data.filled(0), patch_lat, patch_lon

    def min_max_scaling(self, data):
        data_min = data.min()
        data_max = data.max()
        normalized_data = (data - data_min) / (data_max - data_min)
        data = torch.tensor(normalized_data[None], dtype=torch.float32)
        return data, data_min, data_max

    def z_score_normalization(self, data):
        data_mean = data.mean()
        data_std = data.std()
        normalized_data = (data - data_mean) / data_std
        data = torch.tensor(normalized_data[None], dtype=torch.float32)
        return data, data_mean, data_std
    def read_nc_file(self, filepath, variable, time):
        with nc.Dataset(filepath, 'r') as dataset:
            data = dataset.variables[variable][time].astype(np.float32)
            lat = dataset.variables['lat'][:]
            lon = dataset.variables['lon'][:]
        return data, lat, lon

    def convert_lon(self, lon):
        return np.where(lon > 180, lon - 360, lon)

    def reorder_lat_lon(self, lr_data, lr_lat, lr_lon):
        # Reorder latitude from [-90, 90] to [90, -90]
        lr_lat_sorted_indices = np.argsort(lr_lat)[::-1]
        lr_lat = lr_lat[lr_lat_sorted_indices]
        lr_data = lr_data[:, lr_lat_sorted_indices, :]

        # Convert longitude from [0, 360] to [-180, 180]
        lr_lon = self.convert_lon(lr_lon)
        lr_lon_sorted_indices = np.argsort(lr_lon)
        lr_lon = lr_lon[lr_lon_sorted_indices]
        lr_data = lr_data[:, :, lr_lon_sorted_indices]
        return lr_data, lr_lat, lr_lon