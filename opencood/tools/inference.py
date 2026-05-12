# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>, Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import time
from tqdm import tqdm

import torch
import open3d as o3d
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils
import matplotlib.pyplot as plt
import numpy as np
import sys

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', required=True, type=str,
                        default='intermediate',
                        help='late, early or intermediate')
    parser.add_argument('--show_vis', action='store_true',
                        help='whether to show image visualization result')
    parser.add_argument('--show_sequence', action='store_true',
                        help='whether to show video visualization result.'
                             'it can note be set true with show_vis together ')
    parser.add_argument('--save_vis', action='store_true',
                        help='whether to save visualization result')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy_test file')
    parser.add_argument('--global_sort_detections', action='store_false', # NOTE: this was changed to false
                        help='whether to globally sort detections by confidence score.'
                             'If set to True, it is the mainstream AP computing method,'
                             'but would increase the tolerance for FP (False Positives).')
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate']
    assert not (opt.show_vis and opt.show_sequence), 'you can only visualize ' \
                                                    'the results in single ' \
                                                    'image mode or video mode'

    hypes = yaml_utils.load_yaml(None, opt) #TODO: change if you dont want it to always load "config.yaml" (can verify from 'opencood/hypes_yaml/yaml_utils.py')

    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    print(f"{len(opencood_dataset)} samples found.")
    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=16,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model(saved_path, model)
    model.eval()

    # Create the dictionary for evaluation.
    # also store the confidence score for each prediction
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                   0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                   0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}

    if opt.show_sequence:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        vis.get_render_option().point_size = 1.0
        vis.get_render_option().show_coordinate_frame = True

        # used to visualize lidar points
        vis_pcd = o3d.geometry.PointCloud()
        # used to visualize object bounding box, maximum 50
        vis_aabbs_gt = []
        vis_aabbs_pred = []
        for _ in range(50):
            vis_aabbs_gt.append(o3d.geometry.LineSet())
            vis_aabbs_pred.append(o3d.geometry.LineSet())

    for i, batch_data in tqdm(enumerate(data_loader)):
        # print(i)
        # Add a stable frame identifier for downstream logging
        if 'ego' in batch_data and isinstance(batch_data['ego'], dict):
            batch_data['ego']['frame_id'] = [f"{i:04d}"]
            batch_data['ego']['dataset_idx'] = [i]
        else:
            print('!!!!!!NO EGO KEY IN DICT!!!!')
            sys.exit()
        with torch.no_grad():            
            batch_data = train_utils.to_device(batch_data, device)
            if opt.fusion_method == 'late':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_late_fusion(batch_data,
                                                          model,
                                                          opencood_dataset)
            elif opt.fusion_method == 'early':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_early_fusion(batch_data,
                                                           model,
                                                           opencood_dataset)
            elif opt.fusion_method == 'intermediate':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_intermediate_fusion(batch_data,
                                                                  model,
                                                                  opencood_dataset)
            else:
                raise NotImplementedError('Only early, late and intermediate'
                                          'fusion is supported.')

            # # ----------------------- shape test -----------------------------
            # if i == 0:
            #     # record_len location depends on dataset; try both
            #     if 'record_len' in batch_data:
            #         print("record_len:", batch_data['record_len'])
            #     elif 'ego' in batch_data and 'record_len' in batch_data['ego']:
            #         print("record_len:", batch_data['ego']['record_len'])
            #     lidar = batch_data['ego']['origin_lidar'][0]  # torch tensor, shape [N, 4] usually
            #     print("\n[DEBUG shapes]")
            #     print("pred_box_tensor:", type(pred_box_tensor), getattr(pred_box_tensor, "shape", None))
            #     print("pred_score     :", type(pred_score), getattr(pred_score, "shape", None))
            #     print("gt_box_tensor  :", type(gt_box_tensor), getattr(gt_box_tensor, "shape", None))
            #     print("origin_lidar   :", type(lidar), getattr(lidar, "shape", None))

            #     # If they are torch tensors, also print min/max sanity checks
            #     if torch.is_tensor(pred_score) and pred_score.numel() > 0:
            #         print("pred_score min/max:", pred_score.min().item(), pred_score.max().item())
            #     if torch.is_tensor(lidar) and lidar.numel() > 0:
            #         print("lidar xyz min:", lidar[:, :3].min(dim=0).values.tolist())
            #         print("lidar xyz max:", lidar[:, :3].max(dim=0).values.tolist())
            #     print("[/DEBUG shapes]\n")
            # # ---------------------------------

            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.3)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.5)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.7)
            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy') # NOTE: hardcoded to avoid creating many numpy outputs
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                inference_utils.save_prediction_gt(pred_box_tensor,
                                                   gt_box_tensor,
                                                   batch_data['ego'][
                                                       'origin_lidar'][0],
                                                   i,
                                                   npy_save_path)

                # NEW: save confidence scores
                score_path = os.path.join(npy_save_path, f"{i:04d}_score.npy")
                np.save(score_path, pred_score.detach().cpu().numpy())

                # NEW: save number of agents participating (record_len)
                if 'record_len' in batch_data:
                    rl = batch_data['record_len'].detach().cpu().numpy()  # shape (B,)
                elif 'ego' in batch_data and 'record_len' in batch_data['ego']:
                    rl = batch_data['ego']['record_len'].detach().cpu().numpy()
                else:
                    rl = None

                if rl is not None:
                    # batch_size=1, so rl[0] is #agents (incl ego)
                    agent_path = os.path.join(npy_save_path, f"{i:04d}_agents.npy")
                    np.save(agent_path, rl)

            if opt.show_vis or opt.save_vis:
                vis_save_path = ''
                if opt.save_vis:
                    vis_save_path = os.path.join(opt.model_dir, 'vis')
                    if not os.path.exists(vis_save_path):
                        os.makedirs(vis_save_path)
                    vis_save_path = os.path.join(vis_save_path, '%05d.png' % i)

                opencood_dataset.visualize_result(pred_box_tensor,
                                                  gt_box_tensor,
                                                  batch_data['ego'][
                                                      'origin_lidar'],
                                                  opt.show_vis,
                                                  vis_save_path,
                                                  dataset=opencood_dataset)

            if opt.show_sequence:
                pcd, pred_o3d_box, gt_o3d_box = \
                    vis_utils.visualize_inference_sample_dataloader(
                        pred_box_tensor,
                        gt_box_tensor,
                        batch_data['ego']['origin_lidar'],
                        vis_pcd,
                        mode='constant'
                        )
                if i == 0:
                    vis.add_geometry(pcd)
                    vis_utils.linset_assign_list(vis,
                                                 vis_aabbs_pred,
                                                 pred_o3d_box,
                                                 update_mode='add')

                    vis_utils.linset_assign_list(vis,
                                                 vis_aabbs_gt,
                                                 gt_o3d_box,
                                                 update_mode='add')

                vis_utils.linset_assign_list(vis,
                                             vis_aabbs_pred,
                                             pred_o3d_box)
                vis_utils.linset_assign_list(vis,
                                             vis_aabbs_gt,
                                             gt_o3d_box)
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.001)
        #break
    yaml_name=opt.model_dir[len("opencood/pretrained/feaco/generated_"):]
    eval_utils.eval_final_results(result_stat,
                                  opt.model_dir,
                                  opt.global_sort_detections,
                                  yaml_name)
    if opt.show_sequence:
        vis.destroy_window()


if __name__ == '__main__':
    main()

"""
Usage:
    python opencood/tools/inference.py --model_dir opencood/pretrained/v2x-vit  --fusion_method intermediate --save_npy
    python opencood/tools/inference.py --model_dir opencood/pretrained/cobevt_lidar  --fusion_method intermediate --save_npy
    python opencood/tools/inference.py --model_dir opencood/pretrained/feaco  --fusion_method intermediate --save_npy


    python opencood/tools/inference.py \
    --model_dir opencood/pretrained/feaco/generated_feaco_baseline__clean__xyz0__ryp0__seed20 \
    --fusion_method intermediate

Note: does not work in hopper architectures
"""


