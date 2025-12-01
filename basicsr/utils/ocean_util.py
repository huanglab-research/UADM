import numpy as np
import os
import re
import torch
import netCDF4 as nc
import torch.nn as nn
from datetime import datetime
import torch.nn.functional as F
from sklearn.metrics import r2_score
def calculate_delta(img):
    """
    计算 ΔSST (SST梯度)，返回梯度的模值
    img: 2D numpy array, SST 场
    """
    # numpy.gradient 默认 axis=(0->y方向, 1->x方向)
    dy, dx = np.gradient(img)
    grad_mag = np.sqrt(dx**2 + dy**2)
    return grad_mag
def save_patch_to_nc(model, filename, patch, lat, lon):
    match = re.search(r"(\d{8})", filename)
    if not match:
        raise ValueError("Not match (YYYYMMDD format)")
    date_str = match.group(1)
    time_value = datetime.strptime(date_str, "%Y%m%d")

    results_name = filename.replace("OSTIA", model).replace("_gt_", "_")
    out_path = os.path.join('results', model+'_results')
    os.makedirs(out_path, exist_ok=True)
    out_path = os.path.join(out_path, results_name)
    lat_1d = lat.squeeze()
    lon_1d = lon.squeeze()
    if lat_1d.ndim > 1:
        lat_1d = lat_1d[:, 0]
    if lon_1d.ndim > 1:
        lon_1d = lon_1d[0, :]

    with nc.Dataset(out_path, 'w', format='NETCDF4') as ds:
        ds.createDimension('time', 1)
        ds.createDimension('lat', len(lat_1d))
        ds.createDimension('lon', len(lon_1d))

        times = ds.createVariable('time', 'f8', ('time',))
        lats = ds.createVariable('lat', 'f4', ('lat',))
        lons = ds.createVariable('lon', 'f4', ('lon',))
        sst = ds.createVariable('sst', 'f4', ('time', 'lat', 'lon'), zlib=True)
        # 时间单位定义
        time_unit = "seconds since 2020-01-01 00:00:00"
        calendar = "standard"

        # 转换时间为浮点数（秒数）
        time_value = nc.date2num(time_value, units=time_unit, calendar=calendar)
        times.units = time_unit
        times.calendar = calendar

        times[0] = time_value
        lats[:] = lat_1d
        lons[:] = lon_1d
        sst[0, :, :] = patch.astype(np.float32)

        ds.description = 'Deep learning based SST daily sparse grid'
        ds.source = 'None'
        sst.units = 'degree_Celsius'
        lats.units = 'degrees_north'
        lons.units = 'degrees_east'
class RMSEMetric(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        batch_size = pred.shape[0]
        rmse_values = []
        for i in range(batch_size):
            rmse = np.sqrt(np.mean((pred[i] - target[i]) ** 2))
            rmse_values.append(rmse)
        return rmse_values

class BiasMetric(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        batch_size = pred.shape[0]
        bias_values = []
        for i in range(batch_size):
            bias = np.abs(np.mean(pred[i] - target[i]))
            bias_values.append(bias)
        return bias_values

class R2Metric(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        batch_size = pred.shape[0]
        r2_values = []
        for i in range(batch_size):
            r2 = r2_score(pred[i], target[i])
            r2_values.append(r2)
        return r2_values

class PSNROceanMetric(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        batch_size = pred.shape[0]
        psnr_values = []
        for i in range(batch_size):
            mse = np.mean((pred[i] - target[i]) ** 2)
            max_val = max(pred[i].max(), target[i].max())
            psnr = 10 * np.log10((max_val ** 2) / (mse + self.eps))
            psnr_values.append(psnr)
        return psnr_values

class SSIMOceanMetric(nn.Module):
    def __init__(self, C1=0.001, C2=0.001, eps=1e-8):
        super().__init__()
        self.C1 = C1
        self.C2 = C2
        self.eps = eps

    def forward(self, pred, target):
        batch_size = pred.shape[0]
        ssim_values = []
        for i in range(batch_size):
            sr_mean = pred[i].mean()
            gt_mean = target[i].mean()

            sr_var = ((pred[i] - sr_mean) ** 2).mean()
            gt_var = ((target[i] - gt_mean) ** 2).mean()
            covariance = ((pred[i] - sr_mean) * (target[i] - gt_mean)).mean()

            numerator = (2 * sr_mean * gt_mean + self.C1) * (2 * covariance + self.C2)
            denominator = (sr_mean ** 2 + gt_mean ** 2 + self.C1) * (sr_var + gt_var + self.C2)

            ssim = numerator / (denominator + self.eps)
            ssim_values.append(ssim)
        return ssim_values
class DeltaRMSEMetric(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        batch_size = pred.shape[0]
        rmse_values = []
        for i in range(batch_size):
            pred[i] = calculate_delta(pred[i])
            target[i] = calculate_delta(target[i])
            rmse = np.sqrt(np.mean((pred[i] - target[i]) ** 2))
            rmse_values.append(rmse)
        return rmse_values

class DeltaBiasMetric(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        batch_size = pred.shape[0]
        bias_values = []
        for i in range(batch_size):
            pred[i] = calculate_delta(pred[i])
            target[i] = calculate_delta(target[i])
            bias = np.abs(np.mean(pred[i] - target[i]))
            bias_values.append(bias)
        return bias_values

class DeltaR2Metric(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        batch_size = pred.shape[0]
        r2_values = []
        for i in range(batch_size):
            pred[i] = calculate_delta(pred[i])
            target[i] = calculate_delta(target[i])
            r2 = r2_score(pred[i], target[i])
            r2_values.append(r2)
        return r2_values

class DeltaPSNROceanMetric(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        batch_size = pred.shape[0]
        psnr_values = []
        for i in range(batch_size):
            pred[i] = calculate_delta(pred[i])
            target[i] = calculate_delta(target[i])
            mse = np.mean((pred[i] - target[i]) ** 2)
            max_val = max(pred[i].max(), target[i].max())
            psnr = 10 * np.log10((max_val ** 2) / (mse + self.eps))
            psnr_values.append(psnr)
        return psnr_values

class DeltaSSIMOceanMetric(nn.Module):
    def __init__(self, C1=0.001, C2=0.001, eps=1e-8):
        super().__init__()
        self.C1 = C1
        self.C2 = C2
        self.eps = eps

    def forward(self, pred, target):
        batch_size = pred.shape[0]
        ssim_values = []
        for i in range(batch_size):
            pred[i] = calculate_delta(pred[i])
            target[i] = calculate_delta(target[i])

            sr_mean = pred[i].mean()
            gt_mean = target[i].mean()

            sr_var = ((pred[i] - sr_mean) ** 2).mean()
            gt_var = ((target[i] - gt_mean) ** 2).mean()
            covariance = ((pred[i] - sr_mean) * (target[i] - gt_mean)).mean()

            numerator = (2 * sr_mean * gt_mean + self.C1) * (2 * covariance + self.C2)
            denominator = (sr_mean ** 2 + gt_mean ** 2 + self.C1) * (sr_var + gt_var + self.C2)

            ssim = numerator / (denominator + self.eps)
            ssim_values.append(ssim)
        return ssim_values

def compute_strain(u: torch.Tensor, v: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor):
    """
    使用 torch 根据 u, v, dx, dy 计算 total strain magnitude（总形变率）

    输入:
        u:  (B, 1, H, W) — zonal 风速 (单位: m/s)
        v:  (B, 1, H, W) — meridional 风速 (单位: m/s)
        dx: (B, 1, H, W) — 经向网格间距 (单位: m)
        dy: (B, 1, H, W) — 纬向网格间距 (单位: m)

    输出:
        strain: (B, 1, H, W) — 总形变率 (单位: 1/s)
    """

    # === 中心差分函数 ===
    def central_diff(tensor, dim, spacing):
        pad = [0, 0, 0, 0]
        pad[2 * (3 - dim)] = 1
        pad[2 * (3 - dim) + 1] = 1
        # tensor_pad = F.pad(tensor, pad=pad, mode='replicate')
        # spacing_pad = F.pad(spacing, pad=pad, mode='replicate')
        return (tensor.roll(-1, dims=dim) - tensor.roll(1, dims=dim)) / (2 * spacing)

    # === 分别计算偏导 ===
    B, _, H, W = u.shape
    du_dx = central_diff(u, dim=3, spacing=dx)  # ∂u/∂x
    dv_dy = central_diff(v, dim=2, spacing=dy)  # ∂v/∂y
    dv_dx = central_diff(v, dim=3, spacing=dx)  # ∂v/∂x
    du_dy = central_diff(u, dim=2, spacing=dy)  # ∂u/∂y

    # === 应变量公式 ===
    strain = torch.sqrt((du_dx - dv_dy) ** 2 + (dv_dx + du_dy) ** 2)

    # === Norm ===
    strain_min = strain.view(B, -1).min(dim=1)[0].view(B, 1, 1, 1)
    strain_max = strain.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1)
    strain_norm = (strain - strain_min) / (strain_max - strain_min + 1e-8)  # [B, C, H, W]

    return strain_norm  # 单位: 1/s
def compute_dx_dy(lat: torch.Tensor, lon: torch.Tensor):
    """
    lat: (B, 1, H, 1)
    lon: (B, 1, 1, W)
    return: dx, dy (meters), shape: (B, 1, H, W)
    """
    R = 6371000  # Earth radius in meters

    B, _, H, W = lat.shape[0], 1, lat.shape[2], lon.shape[3]

    # 纬度弧度，用于 cos(φ)
    lat_rad = lat * torch.pi / 180.0  # (B, 1, H, 1)
    cos_phi = torch.cos(lat_rad)     # (B, 1, H, 1)

    # 计算间距：假设规则网格，每个样本纬度间距都一致（行方向）
    delta_lat = torch.abs(lat[:, :, 1:, :] - lat[:, :, :-1, :]).mean(dim=2, keepdim=True) * torch.pi / 180  # (B, 1, 1, 1)
    delta_lon = torch.abs(lon[:, :, :, 1:] - lon[:, :, :, :-1]).mean(dim=3, keepdim=True) * torch.pi / 180  # (B, 1, 1, 1)

    dy = R * delta_lat  # (B, 1, 1, 1)
    dx = R * delta_lon * cos_phi  # (B, 1, H, 1)

    dx = dx.expand(-1, -1, -1, W)  # (B,1,H,W)
    dy = dy.expand(-1, -1, H, W)   # (B,1,H,W)

    return dx, dy
def compute_geostrophic(ssh: torch.Tensor, lat: torch.Tensor, lon: torch.Tensor):
    """
    计算地转动能 Kinetic Energy

    输入:
        ssh: (B, 1, H, W) — 海面高度 η（单位 m）
        lat: (B, 1, H, 1) — 纬度（单位 °）
        lon: (B, 1, 1, W) — 经度（单位 °）

    返回:
        ke:  (B, 1, H, W) — 动能场（单位 m²/s²）
    """
    lat = lat.to('cuda')
    lon = lon.to('cuda')

    g = 9.81  # 重力加速度
    Omega = 7.2921e-5  # 地球自转角速度
    B, _, H, W = ssh.shape

    # === Step 1: Coriolis 参数 f0 = 2Ωsin(φ) ===
    phi = lat * torch.pi / 180.0  # 弧度 (B, 1, H, 1)
    f0 = 2 * Omega * torch.sin(phi)  # (B, 1, H, 1)
    f0 = f0.expand(-1, -1, -1, W)  # (B, 1, H, W)

    # 避免 f0 为 0，防止除法爆炸
    f0 = torch.where(torch.abs(f0) < 1e-10, torch.full_like(f0, 1e-10), f0)

    # === Step 2: Streamfunction ψ = g / f0 * ssh ===
    psi = (g / f0) * ssh  # (B, 1, H, W)

    # === Step 3: 网格间距 dx, dy（单位：米） ===
    dx, dy = compute_dx_dy(lat, lon)  # 均为 (B, 1, H, W)

    # === Step 4: 中心差分近似 u = -∂ψ/∂y, v = ∂ψ/∂x ===
    def central_diff_x(tensor, dx):
        return (tensor[:, :, :, 2:] - tensor[:, :, :, :-2]) / (2 * dx[:, :, :, 1:-1])  # (B,1,H,W-2)

    def central_diff_y(tensor, dy):
        return (tensor[:, :, 2:, :] - tensor[:, :, :-2, :]) / (2 * dy[:, :, 1:-1, :])  # (B,1,H-2,W)

    # 裁剪中间区域做差分
    psi_crop = psi  # (B,1,H,W)
    dpsi_dx = central_diff_x(psi_crop, dx)  # (B,1,H,W-2)
    dpsi_dy = central_diff_y(psi_crop, dy)  # (B,1,H-2,W)

    # 为了形状一致，裁剪 ψ
    psi_inner = psi[:, :, 1:-1, 1:-1]  # (B,1,H-2,W-2)

    v = dpsi_dx[:, :, 1:-1, :]  # (B,1,H-2,W-2)
    u = -dpsi_dy[:, :, :, 1:-1]  # (B,1,H-2,W-2)
    v = F.pad(v, (1, 1, 1, 1), mode='replicate')
    u = F.pad(u, (1, 1, 1, 1), mode='replicate')

    # # === Step 5: Kinetic Energy = 0.5 * (u² + v²) ===
    # ke = 0.5 * (u ** 2 + v ** 2)  # (B,1,H-2,W-2)
    # # === Step 6: Norm ===
    # ke_min = ke.view(B, -1).min(dim=1)[0].view(B, 1, 1, 1)
    # ke_max = ke.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1)
    # R_norm = (ke - ke_min) / (ke_max - ke_min + 1e-8)  # [B, C, H, W]

    strain = compute_strain(u, v, dx, dy)

    return strain