#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

def ordered_rect_xy(corners_8_3):
    pts = corners_8_3[:4, :2]
    if not np.isfinite(pts).all():
        return None
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    return pts[np.argsort(ang)]

def bev_iou(corners1, corners2):
    c1 = ordered_rect_xy(corners1)
    c2 = ordered_rect_xy(corners2)
    if c1 is None or c2 is None:
        return 0.0

    p1 = Polygon(c1)
    p2 = Polygon(c2)

    if not p1.is_valid:
        p1 = p1.buffer(0)
    if not p2.is_valid:
        p2 = p2.buffer(0)

    if p1.is_empty or p2.is_empty:
        return 0.0

    try:
        # suppress the specific runtime warning only in this block
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in intersection")
            inter = p1.intersection(p2).area
            union = p1.union(p2).area

        if not np.isfinite(inter) or not np.isfinite(union) or union <= 0:
            return 0.0
        return float(inter / union)

    except Exception:  # includes GEOSException + numeric issues
        return 0.0

def nms_bev(boxes_8_3, scores, iou_thresh=0.1):
    # sort descending by score
    order = scores.argsort()[::-1]
    keep = []
    suppressed = np.zeros(len(order), dtype=bool)

    for _i, idx_i in enumerate(order):
        if suppressed[_i]:
            continue
        keep.append(idx_i)
        for _j in range(_i + 1, len(order)):
            if suppressed[_j]:
                continue
            idx_j = order[_j]
            if bev_iou(boxes_8_3[idx_i], boxes_8_3[idx_j]) > iou_thresh:
                suppressed[_j] = True
    return np.array(keep, dtype=int)

def corners_to_poly_xy(corners_8_3: np.ndarray):
    pts = ordered_rect_xy(corners_8_3)
    if pts is None:
        return None
    return np.vstack([pts, pts[0:1]])


def plot_bev(pcd_xyi: np.ndarray,
             gt_boxes: np.ndarray,
             pred_boxes: np.ndarray,
             out_path: str,
             out_path_pdf: str,
             xlim=None,
             ylim=None,
             title: str = "",
             point_size: float = 0.05,
             max_points: int = 120000):
    """
    pcd_xyi: (N,4) or (N,3/2) but we use first two columns as x,y
    gt_boxes: (M,8,3)
    pred_boxes: (K,8,3)
    """
    # Downsample points for speed/clarity
    if pcd_xyi.shape[0] > max_points:
        idx = np.random.choice(pcd_xyi.shape[0], size=max_points, replace=False)
        pcd = pcd_xyi[idx]
    else:
        pcd = pcd_xyi

    x = pcd[:, 0]
    y = pcd[:, 1]

    fig = plt.figure(figsize=(12, 8), dpi=600)
    ax = plt.gca()

    ax.scatter(x, y, s=point_size, rasterized=True)  # default color, user didn't request specific colors

    # Plot GT boxes
    if gt_boxes is not None and gt_boxes.size > 0:
        for b in gt_boxes:
            poly = corners_to_poly_xy(b)
            if poly is not None:
                ax.plot(poly[:, 0], poly[:, 1], linewidth=0.5, color="green", label="_gt")

    # Plot predicted boxes
    if pred_boxes is not None and pred_boxes.size > 0:
        for b in pred_boxes:
            poly = corners_to_poly_xy(b)
            if poly is not None:
                ax.plot(poly[:, 0], poly[:, 1], linewidth=0.5, color="red", linestyle="--", label="_pred")

    # Add a simple legend proxy
    ax.plot([], [], linewidth=0.5, color="green", label="GT boxes")
    ax.plot([], [], linewidth=0.5, color="red", linestyle="--", label="Pred boxes")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.legend(loc="upper right")
    #ax.grid(True, linewidth=0.3)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=600)
    plt.savefig(out_path_pdf, bbox_inches="tight")  # vector
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy_dir", required=True, help="Directory containing *_pcd.npy, *_pred.npy, *_gt.npy_test.npy")
    ap.add_argument("--out_dir", required=True, help="Output directory for PNGs")
    ap.add_argument("--max_images", type=int, default=700, help="How many frames to render")
    ap.add_argument("--xlim", type=float, nargs=2, default=None, help="Optional x limits: xmin xmax")
    ap.add_argument("--ylim", type=float, nargs=2, default=None, help="Optional y limits: ymin ymax")
    ap.add_argument("--max_points", type=int, default=120000, help="Downsample point cloud to this many points")
    args = ap.parse_args()

    # Find frames by pcd files
    pcd_files = sorted(glob.glob(os.path.join(args.npy_dir, "*_pcd.npy")))
    if not pcd_files:
        raise RuntimeError(f"No *_pcd.npy found in {args.npy_dir}")

    args.max_images = len(pcd_files)
    #for idx, pcd_path in enumerate(pcd_files[:args.max_images]):
    for idx, pcd_path in enumerate(pcd_files):
        base = os.path.basename(pcd_path).replace("_pcd.npy", "")
        pred_path = os.path.join(args.npy_dir, f"{base}_pred.npy")
        gt_path = os.path.join(args.npy_dir, f"{base}_gt.npy_test.npy")
        score_path = os.path.join(args.npy_dir, f"{base}_score.npy")
        agent_path = os.path.join(args.npy_dir, f"{base}_agents.npy")
        pcd = np.load(pcd_path)  # (N,4)
        pred = np.load(pred_path) if os.path.exists(pred_path) else None
        gt = np.load(gt_path) if os.path.exists(gt_path) else None
        # Load scores if available
        if os.path.exists(score_path):
            score = np.load(score_path)
        else:
            score = None

        # Score thresholding (do this BEFORE NMS)
        print(f'Before len pred: {len(pred)}')
        score_thresh = 0.5  # <-- tune: 0.3, 0.5, 0.7
        if pred is not None and pred.size > 0 and score is not None:
            keep = score >= score_thresh
            pred = pred[keep]
            score = score[keep]
        print(f'After len pred: {len(pred)}')

        # BEV NMS (only if we still have boxes)
        if pred is not None and pred.size > 0 and score is not None:
            keep_idx = nms_bev(pred, score, iou_thresh=0.05)
            pred = pred[keep_idx]
            score = score[keep_idx]

        out_path = os.path.join(args.out_dir, f"{base}_bev.png")
        out_path_pdf = os.path.join(args.out_dir, f"{base}_bev.pdf")
        # Title (after we know base; independent of filtering)
        if os.path.exists(agent_path):
            n_agents = int(np.load(agent_path)[0])   # assumes batch_size=1
            title = f"Frame {base} | agents={n_agents}"
        else:
            title = f"Frame {base}"
        plot_bev(
            pcd_xyi=pcd,
            gt_boxes=gt,
            pred_boxes=pred,
            out_path=out_path,
            out_path_pdf=out_path_pdf,
            xlim=args.xlim,
            ylim=args.ylim,
            title=title,
            max_points=args.max_points
        )

        print(f"[{idx+1}/{min(len(pcd_files), args.max_images)}] saved {out_path}")


if __name__ == "__main__":
    main()

"""
Usage:
    python opencood/tools/vis_bev_from_npy.py \
        --npy_dir opencood/pretrained/v2x-vit/npy \
        --out_dir opencood/pretrained/v2x-vit/vis_bev

    python opencood/tools/vis_bev_from_npy.py \
        --npy_dir opencood/pretrained/feaco/npy \
        --out_dir opencood/pretrained/feaco/vis_bev

    python opencood/tools/vis_bev_from_npy.py \
        --npy_dir opencood/pretrained/cobevt_lidar/npy \
        --out_dir opencood/pretrained/cobevt_lidar/vis_bev
"""