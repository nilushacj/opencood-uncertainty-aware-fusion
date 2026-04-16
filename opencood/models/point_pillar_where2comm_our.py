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
            'status_gate_enabled',
            'status_gate_decision',
            'ok_tail_skip_enabled',
            'ok_tail_skip_threshold',
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

        # -- uncertainty weight settings --
        self.enable_uncertainty_weighting = args.get('enable_uncertainty_weighting', False)
        self.uncertainty_weight_alpha = args.get('uncertainty_weight_alpha', 0.5)
        self.uncertainty_weight_trace_clip = args.get('uncertainty_weight_trace_clip', 50.0)
        self.uncertainty_weight_min = args.get('uncertainty_weight_min', 0.2)

        # -- hard skip/binary weight settings --
        self.enable_uncertainty_gating = args.get('enable_uncertainty_gating', False)
        self.uncertainty_gate_mode = args.get('uncertainty_gate_mode', 'hard_skip')
        # supported:
        #   'hard_skip'       -> skip collaborator if trace > threshold
        #   'binary_weight'   -> use low weight if trace > threshold else 1.0

        self.uncertainty_gate_threshold = args.get('uncertainty_gate_threshold', 10.0)
        self.uncertainty_gate_low_weight = args.get('uncertainty_gate_low_weight', 0.5)

        # -- status-aware gating settings --
        self.enable_status_based_skip = args.get('enable_status_based_skip', False)

        # Optional stronger version:
        # if True, also skip very high-uncertainty rows even when status == 'ok'
        self.enable_ok_tail_skip = args.get('enable_ok_tail_skip', False)
        self.ok_tail_skip_threshold = args.get('ok_tail_skip_threshold', 100.0)

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

                # ------ NOTE: REMOVED BLOCK ------ 
                #t_matrix = get_t_matrix(ego_mask,other_mask) # get fine-grid transformation matrix 
                #t_matrix = trans_tx(t_matrix,mask_h,mask_w)
                # ----------------------------------

                # ------ NOTE: ADDED BLOCK (LEVEL 2 - COVARIANCE VALIDATION) ------ 
                mu_affine, mu_params, Sigma_params, stats = get_transform_distribution(ego_mask, other_mask)
                # if j == 1 and i == 0:
                #     print("mu_params:", mu_params)
                #     print("diag(Sigma):", np.diag(Sigma_params))
                #     print("stats:", stats)
                t_matrix = trans_tx(mu_affine, mask_h, mask_w) # NOTE: only mean transform is still used  
                t_matrix = torch.from_numpy(t_matrix).to(features_2d.device).unsqueeze(0)
                # ----------------------------------

                # ------ NOTE: ADDED BLOCK (LEVEL 2 - UNCERTAINTY WEIGHTING) ------ 
                cov_trace_raw = float(np.trace(Sigma_params))
                cov_trace_raw = max(cov_trace_raw, 0.0)
                cov_trace_clipped = min(cov_trace_raw, self.uncertainty_weight_trace_clip)
                uncertainty_score = np.log1p(cov_trace_clipped)

                # default behavior: no extra suppression
                uncertainty_weight = 1.0
                gate_applied_weight = 1.0
                gate_skipped = False
                gate_decision = 'none'
                status_gate_decision = 'disabled'

                # old continuous weighting mode
                if self.enable_uncertainty_weighting:
                    uncertainty_weight, _, uncertainty_score = self.covariance_to_weight(Sigma_params)
                    gate_applied_weight = uncertainty_weight
                    gate_decision = 'continuous_weight'

                # threshold-based gating mode
                if self.enable_uncertainty_gating:
                    gate_applied_weight, gate_skipped, gate_decision, _ = self.covariance_to_gate_decision(Sigma_params)

                # status-based gating mode
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
                # ----------------------------------

                # Logging block: does not change model behavior
                if self.enable_cov_logging:
                    if self._cov_log_pair_counter % self.cov_log_every_n_pairs == 0:
                        other_mask_torch = torch.from_numpy(other_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(features_2d.device)
                        warped_mask = warp_affine_simple(other_mask_torch, t_matrix, (mask_h, mask_w)).squeeze().detach().cpu().numpy()

                        mask_iou_after_warp = compute_mask_iou(ego_mask, warped_mask, threshold=0.5)

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
                            'status_gate_enabled': int(self.enable_status_based_skip),
                            'status_gate_decision': str(status_gate_decision),
                            'ok_tail_skip_enabled': int(self.enable_ok_tail_skip),
                            'ok_tail_skip_threshold': float(self.ok_tail_skip_threshold) if self.enable_ok_tail_skip else None,
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

                features_2d = warp_affine_simple(features_2d.unsqueeze(0), t_matrix, (H, W))
                
                # -- NOTE: ADDED FOR LEVEL 2 UNCERTAINTY WEIGHTING --
                 # continuous weighting (old experiment)
                if self.enable_uncertainty_weighting:
                    features_2d = features_2d * uncertainty_weight

                # threshold-based gating
                if self.enable_uncertainty_gating:
                    if gate_skipped:
                        continue
                    features_2d = features_2d * gate_applied_weight

                # status-based skip can also trigger even if threshold-gating is disabled
                if self.enable_status_based_skip and gate_skipped:
                    continue
                # --

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