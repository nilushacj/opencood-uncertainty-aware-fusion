# -*- coding: utf-8 -*-
# Author: Hao Xiang <haxiang@g.ucla.edu>, Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


from numpy import record
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.where2comm_mutihead import Where2comm
from opencood.models.sub_modules.psm_mask import Communication
from opencood.models.sub_modules.positioning_error_correction_our import get_transform_distribution
import torch
import os 
import csv
import math

def extract_frame_identifier_from_data_dict(data_dict, batch_index=0):
    # Preferred field injected from inference.py
    if 'frame_id' in data_dict:
        value = data_dict['frame_id']
        if isinstance(value, (list, tuple)) and len(value) > batch_index:
            return str(value[batch_index])
        return str(value)

    # Numeric fallback
    if 'dataset_idx' in data_dict:
        value = data_dict['dataset_idx']
        if isinstance(value, (list, tuple)) and len(value) > batch_index:
            return str(value[batch_index])
        return str(value)

    return f"unknown_batch_{batch_index}"

def compute_mask_iou(mask_a, mask_b, threshold=0.5):
    a = mask_a > threshold
    b = mask_b > threshold
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def append_dict_row_to_csv(csv_path, row_dict, header_order=None):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    file_exists = os.path.exists(csv_path)

    if header_order is None:
        header_order = list(row_dict.keys())

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header_order)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)

def warp_affine_simple(src, M, dsize,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False):

    B, C, H, W = src.size()
    grid = F.affine_grid(M,
                         [B, C, dsize[0], dsize[1]],
                         align_corners=align_corners).to(src)
    return F.grid_sample(src, grid, align_corners=align_corners)

def trans_tx(t_matrix,H,W):
    a00 = t_matrix[0][0]
    a01 = t_matrix[0][1]
    a02 = t_matrix[0][2]
    a12 = t_matrix[1][2]
    tx = (a02 - ((1 - a00) * W * 0.5 - a01 * H * 0.5))/(W * 0.5)
    ty = (a12 - (a01 * W * 0.5 + (1 - a00) * H * 0.5))/(H * 0.5)
    
    # DONE: checking this because 2nd row should be [a10, a11, ty]
    # --> the paper explicitly notes that FeaCo still has limitations under high heading errors, suggesting the rectification is sensitive to rotation handling. Thus this is essentially some rotation convention compensation 
    T = np.float32([[a00,a01,tx],
                    [a01,a00,ty],
                    [0,0,1]])
                    
    hang = np.linalg.det(T)
    if hang == 0:
        return  np.float32([[1,0,0],
                    [0,1,0]])
    a = np.linalg.inv(T)
    return a[0:2,:]

class PointPillarWhere2commOur(nn.Module):
    def __init__(self, args):
        super(PointPillarWhere2commOur, self).__init__()

        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        # if 'resnet' in args['base_bev_backbone'] and args['base_bev_backbone']['resnet']:
        #     self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64)
        
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        # self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64)

        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        self.compression = False

        if args['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])

        self.dcn = False

        # self.fusion_net = TransformerFusion(args['fusion_args'])
        self.fusion_net = Where2comm(args['fusion_args'])
        self.multi_scale = args['fusion_args']['multi_scale']

        self.cls_head = nn.Conv2d(128 * 2, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 2, 7 * args['anchor_number'],
                                  kernel_size=1)
        
        self.naive_communication = Communication(args['communication'])

        if args['backbone_fix']:
            self.backbone_fix()
        
        self.enable_cov_logging = args.get('enable_cov_logging', False)
        self.cov_log_dir = args.get('cov_log_dir', 'debug_prm/cov_logs')
        self.cov_log_filename = args.get('cov_log_filename', 'cov_stats.csv')
        self.cov_log_every_n_pairs = args.get('cov_log_every_n_pairs', 1)

        self._cov_log_pair_counter = 0
        self._cov_log_sample_counter = 0

        self._cov_log_header = [
            'sample_idx',
            'frame_id',
            'dataset_idx',
            'record_len',
            'scene_batch_idx',
            'collaborator_idx',
            'theta',
            'tx',
            'ty',
            'covariance_method',
            'cov_theta_var',
            'cov_tx_var',
            'cov_ty_var',
            'cov_trace',
            'cov_trace_clipped',
            'uncertainty_score',
            'uncertainty_weight',
            'gate_mode',
            'gate_threshold',
            'gate_decision',
            'gate_applied_weight',
            'gate_skipped',
            'fused_into_feature_list',
            'status_gate_enabled',
            'status_gate_decision',
            'ok_tail_skip_enabled',
            'ok_tail_skip_threshold',
            'blur_enabled',
            'blur_applied',
            'blur_sigma',
            'blur_kernel_size',
            'ok_tail_vertical_blur_enabled',
            'ok_blur_threshold',
            'vertical_blur_decision',
            'vertical_blur_used_ty_only',
            'mc_warp_enabled',
            'mc_warp_applied',
            'mc_trace_threshold',
            'mc_num_samples',
            'mc_use_translation_only',
            'mc_mean_sampled_theta_abs_dev',
            'mc_mean_sampled_tx_abs_dev',
            'mc_mean_sampled_ty_abs_dev',
            'cov_condition_number',
            'sigma2_hat',
            'num_centroids0_raw',
            'num_centroids1_raw',
            'num_centroids0_filtered',
            'num_centroids1_filtered',
            'num_matches_before_mad',
            'num_matches_after_mad',
            'mean_match_cost_before_mad',
            'mean_match_cost_after_mad',
            'icp_residual_mean',
            'icp_residual_std',
            'mask_iou_after_warp',
            'status'
        ]

        # -- IGNORE (older version): uncertainty weight settings --
        self.enable_uncertainty_weighting = args.get('enable_uncertainty_weighting', False)
        self.uncertainty_weight_alpha = args.get('uncertainty_weight_alpha', 0.5)
        self.uncertainty_weight_trace_clip = args.get('uncertainty_weight_trace_clip', 50.0)
        self.uncertainty_weight_min = args.get('uncertainty_weight_min', 0.2)
        self.enable_uncertainty_gating = args.get('enable_uncertainty_gating', False)
        self.uncertainty_gate_mode = args.get('uncertainty_gate_mode', 'hard_skip')
        self.uncertainty_gate_threshold = args.get('uncertainty_gate_threshold', 10.0)
        self.uncertainty_gate_low_weight = args.get('uncertainty_gate_low_weight', 0.5)
        # -------------------

        # -- status-aware gating settings --
        self.enable_status_based_skip = args.get('enable_status_based_skip', False)

        # -- skip very high uncertainty rows even when status is 'ok' (set to false) --
        self.enable_ok_tail_skip = args.get('enable_ok_tail_skip', False)
        self.ok_tail_skip_threshold = args.get('ok_tail_skip_threshold', 100.0)

        # -- uncertainty-aware blur settings (set to false) --
        self.enable_uncertainty_blur = args.get('enable_uncertainty_blur', False)

        # -- blur is applied only when status == 'ok' --
        self.blur_use_only_ok = args.get('blur_use_only_ok', True)

        # -- scale factor converting covariance-derived sigma to blur sigma in pixels --
        self.blur_sigma_scale = args.get('blur_sigma_scale', 1.0)

        # -- lower / upper clamp for blur sigma --
        self.blur_sigma_min = args.get('blur_sigma_min', 0.0)
        self.blur_sigma_max = args.get('blur_sigma_max', 2.5)

        # -- if blur sigma is below this, skip blurring --
        self.blur_apply_threshold = args.get('blur_apply_threshold', 0.15)

        # -- tail-only directional blur settings --
        self.enable_ok_tail_vertical_blur = args.get('enable_ok_tail_vertical_blur', False)

        # -- only apply blur to ok rows with cov_trace above this threshold --
        self.ok_blur_threshold = args.get('ok_blur_threshold', 9.61)

        # -- use ty variance only for first directional version --
        self.vertical_blur_use_ty_only = args.get('vertical_blur_use_ty_only', True)

        # -- tail-only Monte Carlo warp settings --
        self.enable_tail_mc_warp = args.get('enable_tail_mc_warp', False)

        # -- apply MC warp only to ok rows with cov_trace above this threshold --
        self.mc_trace_threshold = args.get('mc_trace_threshold', 9.800544452667232)

        # -- number of transform samples, including the mean sample only implicitly via sampling --
        self.mc_num_samples = args.get('mc_num_samples', 3)

        # -- for reproducibility if desired; set None to use default randomness --
        self.mc_random_seed = args.get('mc_random_seed', None)

        # -- if True, only use translation covariance (tx, ty) and keep theta fixed at mean --
        self.mc_use_translation_only = args.get('mc_use_translation_only', False)

        # -- optional safety clamp on sampled theta deviation (radians) --
        self.mc_theta_clip = args.get('mc_theta_clip', 0.15)

        # -- optional safety clamp on sampled tx/ty deviation (feature-map pixels) --
        self.mc_translation_clip = args.get('mc_translation_clip', 3.0)

        # -- for monte carlo --
        if self.enable_tail_mc_warp and self.mc_random_seed is not None:
            np.random.seed(int(self.mc_random_seed))
        ###
    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False
    
    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def covariance_to_weight(self, Sigma_params):
        """
        Convert 3x3 covariance over [theta, tx, ty] into a scalar confidence weight.
        Larger covariance -> smaller weight.
        """
        cov_trace = float(np.trace(Sigma_params))
        cov_trace = max(cov_trace, 0.0)
        cov_trace_clipped = min(cov_trace, self.uncertainty_weight_trace_clip)

        uncertainty_score = np.log1p(cov_trace_clipped)
        weight = np.exp(-self.uncertainty_weight_alpha * uncertainty_score)

        weight = float(np.clip(weight, self.uncertainty_weight_min, 1.0))
        return weight, cov_trace, uncertainty_score
    
    def covariance_to_gate_decision(self, Sigma_params):
        """
        Convert covariance into a gate decision for collaborator handling.

        Returns:
            applied_weight: float
            skip_collaborator: bool
            gate_decision: str
            cov_trace: float
        """
        cov_trace = float(np.trace(Sigma_params))
        cov_trace = max(cov_trace, 0.0)

        threshold = float(self.uncertainty_gate_threshold)

        if self.uncertainty_gate_mode == 'hard_skip':
            if cov_trace > threshold:
                return 0.0, True, 'skip', cov_trace
            else:
                return 1.0, False, 'keep', cov_trace

        elif self.uncertainty_gate_mode == 'binary_weight':
            if cov_trace > threshold:
                low_w = float(self.uncertainty_gate_low_weight)
                low_w = float(np.clip(low_w, 0.0, 1.0))
                return low_w, False, 'downweight', cov_trace
            else:
                return 1.0, False, 'keep', cov_trace

        else:
            # fallback: no gate
            return 1.0, False, 'none', cov_trace

    def status_to_gate_decision(self, stats, Sigma_params):
        """
        Status-aware collaborator gating.

        Returns:
            applied_weight: float
            skip_collaborator: bool
            status_gate_decision: str
        """
        status = str(stats.get('status', 'unknown'))

        # Base rule: skip all non-ok cases
        if status != 'ok':
            return 0.0, True, f'skip_status_{status}'

        # Optional stronger rule: skip extreme ok-tail cases too
        if self.enable_ok_tail_skip:
            cov_trace = float(np.trace(Sigma_params))
            cov_trace = max(cov_trace, 0.0)
            if cov_trace > float(self.ok_tail_skip_threshold):
                return 0.0, True, 'skip_ok_tail'

        return 1.0, False, 'keep_status_ok'

    def covariance_to_vertical_blur_sigma(self, Sigma_params):
        """
        Convert covariance over [theta, tx, ty] into a vertical-only blur sigma.
        First directional version: use ty uncertainty only.
        """
        if self.vertical_blur_use_ty_only:
            ty_var = float(Sigma_params[2, 2])
            ty_var = max(ty_var, 0.0)
            sigma = math.sqrt(ty_var)
        else:
            tx_var = float(Sigma_params[1, 1])
            ty_var = float(Sigma_params[2, 2])
            tx_var = max(tx_var, 0.0)
            ty_var = max(ty_var, 0.0)
            sigma = math.sqrt(0.5 * (tx_var + ty_var))

        sigma = sigma * self.blur_sigma_scale
        sigma = float(np.clip(sigma, self.blur_sigma_min, self.blur_sigma_max))
        return sigma
    
    def covariance_to_blur_sigma(self, Sigma_params):
        """
        Convert covariance over [theta, tx, ty] into a scalar blur sigma.
        First version: use translation uncertainty only.
        """
        tx_var = float(Sigma_params[1, 1])
        ty_var = float(Sigma_params[2, 2])

        tx_var = max(tx_var, 0.0)
        ty_var = max(ty_var, 0.0)

        sigma = math.sqrt(0.5 * (tx_var + ty_var))
        sigma = sigma * self.blur_sigma_scale
        sigma = float(np.clip(sigma, self.blur_sigma_min, self.blur_sigma_max))
        return sigma


    def gaussian_kernel_2d(self, sigma, device, dtype):
        """
        Create a 2D Gaussian kernel for depthwise convolution.
        """
        if sigma <= 0:
            kernel = torch.tensor([[1.0]], device=device, dtype=dtype)
            return kernel, 1

        radius = max(1, int(math.ceil(3.0 * sigma)))
        ksize = 2 * radius + 1

        coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(coords, coords, indexing='ij')
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()

        return kernel, ksize

    def gaussian_kernel_vertical_1d(self, sigma, device, dtype):
        """
        Create a vertical 1D Gaussian kernel of shape [ksize, 1].
        """
        if sigma <= 0:
            kernel = torch.tensor([[1.0]], device=device, dtype=dtype)
            return kernel, 1

        radius = max(1, int(math.ceil(3.0 * sigma)))
        ksize = 2 * radius + 1

        coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(ksize, 1)

        return kernel, ksize

    def apply_gaussian_blur_depthwise(self, x, sigma):
        """
        x: [B, C, H, W]
        applies same Gaussian kernel to each channel independently
        """
        if sigma < self.blur_apply_threshold:
            return x, False, 1

        kernel_2d, ksize = self.gaussian_kernel_2d(sigma, x.device, x.dtype)
        kernel_2d = kernel_2d.view(1, 1, ksize, ksize)

        C = x.shape[1]
        weight = kernel_2d.repeat(C, 1, 1, 1)

        x_blur = F.conv2d(
            x,
            weight,
            bias=None,
            stride=1,
            padding=ksize // 2,
            groups=C
        )
        return x_blur, True, ksize

    def apply_vertical_gaussian_blur_depthwise(self, x, sigma):
        """
        x: [B, C, H, W]
        Applies a vertical-only Gaussian blur to each channel independently.
        """
        if sigma < self.blur_apply_threshold:
            return x, False, 1

        kernel_1d, ksize = self.gaussian_kernel_vertical_1d(sigma, x.device, x.dtype)
        kernel_2d = kernel_1d.view(1, 1, ksize, 1)

        C = x.shape[1]
        weight = kernel_2d.repeat(C, 1, 1, 1)

        x_blur = F.conv2d(
            x,
            weight,
            bias=None,
            stride=1,
            padding=(ksize // 2, 0),
            groups=C
        )
        return x_blur, True, ksize

    def params_to_affine(self, theta, tx, ty):
        """
        Convert [theta, tx, ty] to 2x3 affine matrix in pixel coordinates.
        """
        c = math.cos(theta)
        s = math.sin(theta)
        return np.float32([
            [c, -s, tx],
            [s,  c, ty]
        ])
    
    def sample_transform_params(self, mu_params, Sigma_params):
        mu = np.asarray(mu_params, dtype=np.float64).reshape(3)
        Sigma = np.asarray(Sigma_params, dtype=np.float64).reshape(3, 3)

        Sigma = 0.5 * (Sigma + Sigma.T)
        Sigma = Sigma + 1e-8 * np.eye(3, dtype=np.float64)

        if self.mc_use_translation_only:
            Sigma_mod = np.zeros((3, 3), dtype=np.float64)
            Sigma_mod[1:, 1:] = Sigma[1:, 1:]
            Sigma_mod = 0.5 * (Sigma_mod + Sigma_mod.T)
            Sigma_mod = Sigma_mod + 1e-8 * np.eye(3, dtype=np.float64)
            sample = np.random.multivariate_normal(mu, Sigma_mod)
            sample[0] = mu[0]
        else:
            sample = np.random.multivariate_normal(mu, Sigma)

        # Safety clipping relative to the mean
        dtheta = sample[0] - mu[0]
        dtx = sample[1] - mu[1]
        dty = sample[2] - mu[2]

        dtheta = float(np.clip(dtheta, -self.mc_theta_clip, self.mc_theta_clip))
        dtx = float(np.clip(dtx, -self.mc_translation_clip, self.mc_translation_clip))
        dty = float(np.clip(dty, -self.mc_translation_clip, self.mc_translation_clip))

        sample[0] = mu[0] + dtheta
        sample[1] = mu[1] + dtx
        sample[2] = mu[2] + dty

        return sample.astype(np.float32), abs(dtheta), abs(dtx), abs(dty)        
    
    def monte_carlo_warp_features(self, features_2d, mu_params, Sigma_params, mask_h, mask_w, H, W):
        """
        features_2d: [C, H, W]
        Returns:
            warped_mean: [1, C, H, W]
            mc_applied: bool
            stats_dict: dict
        """
        src_tensor = features_2d.unsqueeze(0)  # [1, C, H, W]
        return self.monte_carlo_warp_tensor(
            src_tensor=src_tensor,
            mu_params=mu_params,
            Sigma_params=Sigma_params,
            mask_h=mask_h,
            mask_w=mask_w,
            out_h=H,
            out_w=W,
        )    

    def monte_carlo_warp_tensor(self, src_tensor, mu_params, Sigma_params, mask_h, mask_w, out_h, out_w):
        """
        Generic Monte Carlo warp for any tensor of shape [1, C, H, W].

        Returns:
            warped_mean: [1, C, out_h, out_w]
            mc_applied: bool
            stats_dict: dict
        """
        K = int(self.mc_num_samples)

        if K <= 1:
            mu_affine = self.params_to_affine(
                float(mu_params[0]),
                float(mu_params[1]),
                float(mu_params[2])
            )
            t_matrix = trans_tx(mu_affine, mask_h, mask_w)
            t_matrix = torch.from_numpy(t_matrix).to(src_tensor.device).unsqueeze(0)
            warped = warp_affine_simple(src_tensor, t_matrix, (out_h, out_w))
            return warped, False, {
                'mc_mean_sampled_theta_abs_dev': 0.0,
                'mc_mean_sampled_tx_abs_dev': 0.0,
                'mc_mean_sampled_ty_abs_dev': 0.0,
            }

        warped_list = []
        theta_abs_devs = []
        tx_abs_devs = []
        ty_abs_devs = []

        for _ in range(K):
            sampled_params, dtheta_abs, dtx_abs, dty_abs = self.sample_transform_params(
                mu_params, Sigma_params
            )

            sampled_affine = self.params_to_affine(
                float(sampled_params[0]),
                float(sampled_params[1]),
                float(sampled_params[2]),
            )
            sampled_t_matrix = trans_tx(sampled_affine, mask_h, mask_w)
            sampled_t_matrix = torch.from_numpy(sampled_t_matrix).to(src_tensor.device).unsqueeze(0)

            warped_k = warp_affine_simple(src_tensor, sampled_t_matrix, (out_h, out_w))
            warped_list.append(warped_k)

            theta_abs_devs.append(dtheta_abs)
            tx_abs_devs.append(dtx_abs)
            ty_abs_devs.append(dty_abs)

        warped_mean = torch.mean(torch.stack(warped_list, dim=0), dim=0)

        return warped_mean, True, {
            'mc_mean_sampled_theta_abs_dev': float(np.mean(theta_abs_devs)) if theta_abs_devs else 0.0,
            'mc_mean_sampled_tx_abs_dev': float(np.mean(tx_abs_devs)) if tx_abs_devs else 0.0,
            'mc_mean_sampled_ty_abs_dev': float(np.mean(ty_abs_devs)) if ty_abs_devs else 0.0,
        }
    def forward(self, data_dict):

        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        
        record_len = data_dict['record_len']

        pairwise_t_matrix = data_dict['pairwise_t_matrix']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c 
        batch_dict = self.pillar_vfe(batch_dict)  
        # n, c -> N, C, H, W 
        batch_dict = self.scatter(batch_dict)
        # N, C, H', W'  
        batch_dict = self.backbone(batch_dict) 
        spatial_features_2d = batch_dict['spatial_features_2d']
        
        # downsample feature to reduce memory
        if self.shrink_flag: 
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)
        
        psm_single = self.cls_head(spatial_features_2d)

        split_psm_single = self.regroup(psm_single, record_len)

        _, communication_masks = self.naive_communication(split_psm_single, record_len)
        
        split_spatial_features_2d = self.regroup(batch_dict['spatial_features'], record_len) 
        feature_list = []

        for i in range(len(communication_masks)):
            mask = communication_masks[i].squeeze(1).to('cpu').numpy()
            frame_id = extract_frame_identifier_from_data_dict(data_dict, batch_index=i)
            if 'dataset_idx' in data_dict:
                dataset_idx_value = data_dict['dataset_idx']
                if isinstance(dataset_idx_value, (list, tuple)) and len(dataset_idx_value) > i:
                    dataset_idx = int(dataset_idx_value[i])
                else:
                    dataset_idx = int(dataset_idx_value)
            else:
                dataset_idx = -1

            record_len_i = int(record_len[i].item()) if hasattr(record_len[i], 'item') else int(record_len[i])
            cav_num,mask_h,mask_w = mask.shape
            ego_mask = mask[0]
            feature_list.append(split_spatial_features_2d[i][0].unsqueeze(0))

            for j in range(1,cav_num):
                features_2d = split_spatial_features_2d[i][j]
                C,H,W = features_2d.shape
                other_mask = mask[j]

                # ------ NOTE: ADDED BLOCK (LEVEL 2 - COVARIANCE VALIDATION) ------ 
                mu_affine, mu_params, Sigma_params, stats = get_transform_distribution(ego_mask, other_mask)
                t_matrix = trans_tx(mu_affine, mask_h, mask_w) # NOTE: only mean transform is still used  
                t_matrix = torch.from_numpy(t_matrix).to(features_2d.device).unsqueeze(0)
                # ----------------------------------

                # ------ NOTE: ADDED BLOCK (LEVEL 2 - UNCERTAINTY WEIGHTING) ------ 
                cov_trace_raw = float(np.trace(Sigma_params))
                cov_trace_raw = max(cov_trace_raw, 0.0)
                cov_trace_clipped = min(cov_trace_raw, self.uncertainty_weight_trace_clip)
                uncertainty_score = np.log1p(cov_trace_clipped)

                # ------ NOTE: ADDED MONTE CARLO SAMPLING VARIABLES ------
                mc_warp_applied = False
                mc_mean_sampled_theta_abs_dev = 0.0
                mc_mean_sampled_tx_abs_dev = 0.0
                mc_mean_sampled_ty_abs_dev = 0.0

                # default behavior no extra suppression
                uncertainty_weight = 1.0
                gate_applied_weight = 1.0
                gate_skipped = False
                gate_decision = 'none'
                status_gate_decision = 'disabled'

                # IGNORE: continuous weighting mode
                if self.enable_uncertainty_weighting:
                    uncertainty_weight, _, uncertainty_score = self.covariance_to_weight(Sigma_params)
                    gate_applied_weight = uncertainty_weight
                    gate_decision = 'continuous_weight'

                # IGNORE: threshold-based gating mode
                if self.enable_uncertainty_gating:
                    gate_applied_weight, gate_skipped, gate_decision, _ = self.covariance_to_gate_decision(Sigma_params)

                # ------ status-based gating mode ------
                if self.enable_status_based_skip:
                    status_weight, status_skip, status_gate_decision = self.status_to_gate_decision(stats, Sigma_params)

                    # status-based skip takes precedence over threshold-based gating
                    if status_skip:
                        gate_applied_weight = status_weight
                        gate_skipped = True
                        gate_decision = status_gate_decision
                    else:
                        # keep current gate settings if status says keep
                        # but mark the status-based decision in logs
                        pass
                # -------------------------------------------------------------

                # ------ Uncertainty blur ------
                blur_sigma = 0.0
                blur_applied = False
                blur_kernel_size = 1
                vertical_blur_decision = 'disabled'

                status_str = str(stats.get('status', 'unknown'))
                blur_is_allowed = False

                # ------ monte carlo use decision: apply MC warp only for valid, not-skipped, high-uncertainty ok rows ------
                mc_warp_is_allowed = False
                if self.enable_tail_mc_warp:
                    if (not gate_skipped) and (status_str == 'ok') and (cov_trace_raw > float(self.mc_trace_threshold)):
                        mc_warp_is_allowed = True       
                # -------------------------------------------------------------

                # only blur ok rows in the high-uncertainty tail
                if self.enable_ok_tail_vertical_blur:
                    if status_str == 'ok':
                        if cov_trace_raw > float(self.ok_blur_threshold):
                            blur_is_allowed = True
                            vertical_blur_decision = 'apply_ok_tail_vertical_blur'
                        else:
                            vertical_blur_decision = 'skip_below_ok_blur_threshold'
                    else:
                        vertical_blur_decision = f'skip_status_{status_str}'
                else:
                    vertical_blur_decision = 'disabled'

                if blur_is_allowed:
                    blur_sigma = self.covariance_to_vertical_blur_sigma(Sigma_params)

                # -------------------------------------------------------------
                
                # # -------------------------------------------------------------
                # # v1.0: Deterministic: warp collaborator feature map
                # features_2d = warp_affine_simple(features_2d.unsqueeze(0), t_matrix, (H, W))
                # # -------------------------------------------------------------

                # -------------------------------------------------------------
                # v2.0: Probabilistic: warp collaborator feature map
                if mc_warp_is_allowed:
                    features_2d, mc_warp_applied, mc_stats = self.monte_carlo_warp_features(
                        features_2d=features_2d,
                        mu_params=mu_params,
                        Sigma_params=Sigma_params,
                        mask_h=mask_h,
                        mask_w=mask_w,
                        H=H,
                        W=W,
                    )
                    mc_mean_sampled_theta_abs_dev = mc_stats['mc_mean_sampled_theta_abs_dev']
                    mc_mean_sampled_tx_abs_dev = mc_stats['mc_mean_sampled_tx_abs_dev']
                    mc_mean_sampled_ty_abs_dev = mc_stats['mc_mean_sampled_ty_abs_dev']
                else:
                    features_2d = warp_affine_simple(features_2d.unsqueeze(0), t_matrix, (H, W))                
                # -------------------------------------------------------------


                # IGNORE: old continuous weighting (if enabled)
                if self.enable_uncertainty_weighting:
                    features_2d = features_2d * uncertainty_weight

                # IGNORE: threshold-based weighting (only apply weight here; do NOT continue yet)
                if self.enable_uncertainty_gating and (not gate_skipped):
                    features_2d = features_2d * gate_applied_weight

                # uncertainty-aware blur for valid / kept collaborator features
                # Do not blur rows that are going to be skipped
                # if self.enable_uncertainty_blur and blur_is_allowed and (not gate_skipped):
                #     features_2d, blur_applied, blur_kernel_size = self.apply_gaussian_blur_depthwise(
                #         features_2d, blur_sigma
                #     )


                # -------------------------------------------------------------
                # # v1.0: vertical blur (MODIFIED TO VERSION BELOW FOR MC TEST)
                # if self.enable_ok_tail_vertical_blur and blur_is_allowed and (not gate_skipped):
                #     features_2d, blur_applied, blur_kernel_size = self.apply_vertical_gaussian_blur_depthwise(
                #         features_2d, blur_sigma
                #     )
                # -------------------------------------------------------------

                # -------------------------------------------------------------
                # v2.0: verticle blur disable if MC warp is applied
                if self.enable_ok_tail_vertical_blur and blur_is_allowed and (not gate_skipped) and (not mc_warp_applied):
                    features_2d, blur_applied, blur_kernel_size = self.apply_vertical_gaussian_blur_depthwise(
                        features_2d, blur_sigma
                    )

                # -------------------------------------------------------------
                # 2) Logging block: log ALL collaborators, including skipped ones
                if self.enable_cov_logging:
                    if self._cov_log_pair_counter % self.cov_log_every_n_pairs == 0:
                        other_mask_torch = torch.from_numpy(other_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(features_2d.device)

                        if mc_warp_applied:
                            warped_mask_tensor, _, _ = self.monte_carlo_warp_tensor(
                                src_tensor=other_mask_torch,
                                mu_params=mu_params,
                                Sigma_params=Sigma_params,
                                mask_h=mask_h,
                                mask_w=mask_w,
                                out_h=mask_h,
                                out_w=mask_w,
                            )
                        else:
                            warped_mask_tensor = warp_affine_simple(other_mask_torch, t_matrix, (mask_h, mask_w))

                        warped_mask = warped_mask_tensor.squeeze().detach().cpu().numpy()
                        mask_iou_after_warp = compute_mask_iou(
                            ego_mask,
                            warped_mask,
                            threshold=0.5
                        )

                        row = {
                            'sample_idx': int(self._cov_log_sample_counter),
                            'frame_id': str(frame_id),
                            'dataset_idx': int(dataset_idx),
                            'record_len': int(record_len_i),
                            'scene_batch_idx': int(i),
                            'collaborator_idx': int(j),
                            'theta': float(mu_params[0]),
                            'tx': float(mu_params[1]),
                            'ty': float(mu_params[2]),
                            'covariance_method': str(stats.get('covariance_method', 'unknown')),
                            'cov_theta_var': float(Sigma_params[0, 0]),
                            'cov_tx_var': float(Sigma_params[1, 1]),
                            'cov_ty_var': float(Sigma_params[2, 2]),
                            'cov_trace': float(np.trace(Sigma_params)),
                            'cov_trace_clipped': float(cov_trace_clipped),
                            'uncertainty_score': float(uncertainty_score),
                            'uncertainty_weight': float(uncertainty_weight),
                            'gate_mode': str(self.uncertainty_gate_mode) if self.enable_uncertainty_gating else 'disabled',
                            'gate_threshold': float(self.uncertainty_gate_threshold) if self.enable_uncertainty_gating else None,
                            'gate_decision': str(gate_decision),
                            'gate_applied_weight': float(gate_applied_weight),
                            'gate_skipped': int(gate_skipped),
                            'fused_into_feature_list': int(not gate_skipped),
                            'status_gate_enabled': int(self.enable_status_based_skip),
                            'status_gate_decision': str(status_gate_decision),
                            'ok_tail_skip_enabled': int(self.enable_ok_tail_skip),
                            'ok_tail_skip_threshold': float(self.ok_tail_skip_threshold) if self.enable_ok_tail_skip else None,
                            'blur_enabled': int(self.enable_uncertainty_blur or self.enable_ok_tail_vertical_blur),
                            'blur_applied': int(blur_applied),
                            'blur_sigma': float(blur_sigma),
                            'blur_kernel_size': int(blur_kernel_size),
                            'ok_tail_vertical_blur_enabled': int(self.enable_ok_tail_vertical_blur),
                            'ok_blur_threshold': float(self.ok_blur_threshold) if self.enable_ok_tail_vertical_blur else None,
                            'vertical_blur_decision': str(vertical_blur_decision),
                            'vertical_blur_used_ty_only': int(self.vertical_blur_use_ty_only),
                            'mc_warp_enabled': int(self.enable_tail_mc_warp),
                            'mc_warp_applied': int(mc_warp_applied),
                            'mc_trace_threshold': float(self.mc_trace_threshold) if self.enable_tail_mc_warp else None,
                            'mc_num_samples': int(self.mc_num_samples) if self.enable_tail_mc_warp else None,
                            'mc_use_translation_only': int(self.mc_use_translation_only),
                            'mc_mean_sampled_theta_abs_dev': float(mc_mean_sampled_theta_abs_dev),
                            'mc_mean_sampled_tx_abs_dev': float(mc_mean_sampled_tx_abs_dev),
                            'mc_mean_sampled_ty_abs_dev': float(mc_mean_sampled_ty_abs_dev),
                            'cov_condition_number': float(stats['cov_condition_number']) if stats.get('cov_condition_number') is not None else None,
                            'sigma2_hat': float(stats['sigma2_hat']) if stats.get('sigma2_hat') is not None else None,
                            'num_centroids0_raw': int(stats.get('num_centroids0_raw', 0)),
                            'num_centroids1_raw': int(stats.get('num_centroids1_raw', 0)),
                            'num_centroids0_filtered': int(stats.get('num_centroids0_filtered', 0)),
                            'num_centroids1_filtered': int(stats.get('num_centroids1_filtered', 0)),
                            'num_matches_before_mad': int(stats.get('num_matches_before_mad', 0)),
                            'num_matches_after_mad': int(stats.get('num_matches_after_mad', 0)),
                            'mean_match_cost_before_mad': float(stats['mean_match_cost_before_mad']) if stats.get('mean_match_cost_before_mad') is not None else None,
                            'mean_match_cost_after_mad': float(stats['mean_match_cost_after_mad']) if stats.get('mean_match_cost_after_mad') is not None else None,
                            'icp_residual_mean': float(stats['icp_residual_mean']) if stats.get('icp_residual_mean') is not None else None,
                            'icp_residual_std': float(stats['icp_residual_std']) if stats.get('icp_residual_std') is not None else None,
                            'mask_iou_after_warp': float(mask_iou_after_warp),
                            'status': str(stats.get('status', 'unknown'))
                        }

                        csv_path = os.path.join(self.cov_log_dir, self.cov_log_filename)
                        append_dict_row_to_csv(csv_path, row, header_order=self._cov_log_header)
                        self._cov_log_sample_counter += 1

                    self._cov_log_pair_counter += 1
                # -------------------------------------------------------------
  
                # -------------------------------------------------------------
                # 3) Now apply skip logic AFTER logging
                if gate_skipped:
                    continue
                # -------------------------------------------------------------

                feature_list.append(features_2d)


        batch_dict['spatial_features'] = torch.vstack(feature_list)
        
        if self.multi_scale:
            fused_feature= self.fusion_net(batch_dict['spatial_features'],
                                            record_len,
                                            self.backbone)
            # downsample feature to reduce memory
            if self.shrink_flag:
                fused_feature = self.shrink_conv(fused_feature)
            
        else:
            # batch_dict = self.backbone(batch_dict) 
            # spatial_features_2d = batch_dict['spatial_features_2d']
            # if self.shrink_flag:  
            #     spatial_features_2d = self.shrink_conv(spatial_features_2d)
            
            fused_feature = self.fusion_net(spatial_features_2d,
                                            record_len,)
            
        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)


        output_dict = {'psm': psm,
                       'rm': rm,
                       }
        

        return output_dict