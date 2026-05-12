#!/usr/bin/env python3
import os
import glob
import argparse
import warnings
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
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in intersection")
            inter = p1.intersection(p2).area
            union = p1.union(p2).area

        if not np.isfinite(inter) or not np.isfinite(union) or union <= 0:
            return 0.0
        return float(inter / union)

    except Exception:
        return 0.0


def nms_bev(boxes_8_3, scores, iou_thresh=0.1):
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


def box_center_xy(box_8_3: np.ndarray):
    pts = box_8_3[:, :2]
    if pts.size == 0 or not np.isfinite(pts).all():
        return None
    return pts.mean(axis=0)


def filter_points_in_range(pcd_xyi: np.ndarray, x_range, y_range):
    if pcd_xyi is None or pcd_xyi.size == 0:
        return pcd_xyi
    mask = (
        (pcd_xyi[:, 0] >= x_range[0]) &
        (pcd_xyi[:, 0] <= x_range[1]) &
        (pcd_xyi[:, 1] >= y_range[0]) &
        (pcd_xyi[:, 1] <= y_range[1])
    )
    return pcd_xyi[mask]


def filter_boxes_in_range(boxes: np.ndarray, x_range, y_range):
    """
    Keep boxes whose center lies inside the requested x/y range.
    boxes: (M, 8, 3)
    """
    if boxes is None or boxes.size == 0:
        return boxes

    keep = []
    for b in boxes:
        c = box_center_xy(b)
        if c is None:
            continue
        if (x_range[0] <= c[0] <= x_range[1]) and (y_range[0] <= c[1] <= y_range[1]):
            keep.append(b)

    if len(keep) == 0:
        return np.empty((0, 8, 3), dtype=boxes.dtype)

    return np.stack(keep, axis=0)


def downsample_points(pcd_xyi: np.ndarray, max_points: int):
    if pcd_xyi is None or pcd_xyi.size == 0:
        return pcd_xyi
    if pcd_xyi.shape[0] > max_points:
        idx = np.random.choice(pcd_xyi.shape[0], size=max_points, replace=False)
        return pcd_xyi[idx]
    return pcd_xyi


def draw_bev_panel(ax,
                   pcd_xyi: np.ndarray,
                   gt_boxes: np.ndarray,
                   pred_boxes: np.ndarray,
                   point_size: float = 0.05,
                   xlim=None,
                   ylim=None,
                   show_axes=True,
                   show_labels=True,
                   show_title=False,
                   title="",
                   show_legend=False):
    """
    Draw a single BEV panel on a provided axis.
    """
    if pcd_xyi is not None and pcd_xyi.size > 0:
        x = pcd_xyi[:, 0]
        y = pcd_xyi[:, 1]
        ax.scatter(x, y, s=point_size, rasterized=True)

    if gt_boxes is not None and gt_boxes.size > 0:
        for b in gt_boxes:
            poly = corners_to_poly_xy(b)
            if poly is not None:
                ax.plot(poly[:, 0], poly[:, 1], linewidth=0.5, color="green", label="_gt")

    if pred_boxes is not None and pred_boxes.size > 0:
        for b in pred_boxes:
            poly = corners_to_poly_xy(b)
            if poly is not None:
                ax.plot(poly[:, 0], poly[:, 1], linewidth=0.5, color="red", linestyle="--", label="_pred")

    if show_legend:
        ax.plot([], [], linewidth=0.5, color="green", label="GT boxes")
        ax.plot([], [], linewidth=0.5, color="red", linestyle="--", label="Pred boxes")
        ax.legend(loc="upper right")

    ax.set_aspect("equal", adjustable="box")

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if show_title:
        ax.set_title(title)

    if show_labels:
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

    if not show_axes:
        #ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(True)

def plot_bev_side_by_side(pcd_xyi: np.ndarray,
                          gt_boxes: np.ndarray,
                          pred_boxes: np.ndarray,
                          out_path: str,
                          title: str = "",
                          point_size: float = 0.05,
                          max_points_full: int = 120000,
                          max_points_zoom: int = 120000,
                          full_xlim=None,
                          full_ylim=None,
                          zoom_range: float = 20.0):
    """
    Create a single PNG with:
      - Left: full BEV
      - Right: zoomed BEV around ego center (0, 0), restricted to +/- zoom_range
    """
    zoom_x = (-zoom_range, zoom_range)
    zoom_y = (-zoom_range, zoom_range)

    # Full-view point cloud downsampling
    pcd_full = downsample_points(pcd_xyi, max_points_full)

    # Zoom-view filtering + downsampling
    pcd_zoom = filter_points_in_range(pcd_xyi, zoom_x, zoom_y)
    pcd_zoom = downsample_points(pcd_zoom, max_points_zoom)

    gt_zoom = filter_boxes_in_range(gt_boxes, zoom_x, zoom_y)
    pred_zoom = filter_boxes_in_range(pred_boxes, zoom_x, zoom_y)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=400)

    # Left: normal/full view
    draw_bev_panel(
        ax=axes[0],
        pcd_xyi=pcd_full,
        gt_boxes=gt_boxes,
        pred_boxes=pred_boxes,
        point_size=point_size,
        xlim=full_xlim,
        ylim=full_ylim,
        show_axes=True,
        show_labels=True,
        show_title=True,
        title=title,
        show_legend=True
    )

    # Right: zoomed view, no axes / no legend / no title
    draw_bev_panel(
        ax=axes[1],
        pcd_xyi=pcd_zoom,
        gt_boxes=gt_zoom,
        pred_boxes=pred_zoom,
        point_size=point_size,
        xlim=zoom_x,
        ylim=zoom_y,
        show_axes=False,
        show_labels=False,
        show_title=True,
        title="Zoomed in (40 meter vicinity)",
        show_legend=False
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=400)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy_dir", required=True,
                    help="Directory containing *_pcd.npy, *_pred.npy, *_gt.npy_test.npy")
    ap.add_argument("--out_dir", required=True,
                    help="Output directory for PNGs")
    ap.add_argument("--max_images", type=int, default=700,
                    help="How many frames to render")
    ap.add_argument("--xlim", type=float, nargs=2, default=None,
                    help="Optional x limits for the full BEV panel: xmin xmax")
    ap.add_argument("--ylim", type=float, nargs=2, default=None,
                    help="Optional y limits for the full BEV panel: ymin ymax")
    ap.add_argument("--zoom_range", type=float, default=20.0,
                    help="Zoomed panel shows x,y in [-zoom_range, +zoom_range]")
    ap.add_argument("--max_points", type=int, default=120000,
                    help="Downsample point cloud to this many points per panel")
    ap.add_argument("--score_thresh", type=float, default=0.5,
                    help="Prediction score threshold before NMS")
    ap.add_argument("--nms_iou", type=float, default=0.05,
                    help="BEV NMS IoU threshold")
    args = ap.parse_args()

    pcd_files = sorted(glob.glob(os.path.join(args.npy_dir, "*_pcd.npy")))
    if not pcd_files:
        raise RuntimeError(f"No *_pcd.npy found in {args.npy_dir}")

    args.max_images = min(args.max_images, len(pcd_files))

    for idx, pcd_path in enumerate(pcd_files[:args.max_images]):
        base = os.path.basename(pcd_path).replace("_pcd.npy", "")
        pred_path = os.path.join(args.npy_dir, f"{base}_pred.npy")
        gt_path = os.path.join(args.npy_dir, f"{base}_gt.npy_test.npy")
        score_path = os.path.join(args.npy_dir, f"{base}_score.npy")
        agent_path = os.path.join(args.npy_dir, f"{base}_agents.npy")

        pcd = np.load(pcd_path)
        pred = np.load(pred_path) if os.path.exists(pred_path) else None
        gt = np.load(gt_path) if os.path.exists(gt_path) else None
        score = np.load(score_path) if os.path.exists(score_path) else None

        print(f"Before len pred: {0 if pred is None else len(pred)}")

        # Score thresholding before NMS
        if pred is not None and pred.size > 0 and score is not None:
            keep = score >= args.score_thresh
            pred = pred[keep]
            score = score[keep]

        print(f"After len pred: {0 if pred is None else len(pred)}")

        # NMS
        if pred is not None and pred.size > 0 and score is not None:
            keep_idx = nms_bev(pred, score, iou_thresh=args.nms_iou)
            pred = pred[keep_idx]
            score = score[keep_idx]

        out_path = os.path.join(args.out_dir, f"{base}_bev_side_by_side.png")

        if os.path.exists(agent_path):
            n_agents = int(np.load(agent_path)[0])  # assumes batch_size=1
            #title = f"Frame {base} | agents={n_agents}"
            title = f"Vehicle agents={n_agents}"
        else:
            #title = f"Frame {base}"
            title = f""

        plot_bev_side_by_side(
            pcd_xyi=pcd,
            gt_boxes=gt,
            pred_boxes=pred,
            out_path=out_path,
            title=title,
            point_size=0.05,
            max_points_full=args.max_points,
            max_points_zoom=args.max_points,
            full_xlim=args.xlim,
            full_ylim=args.ylim,
            zoom_range=args.zoom_range
        )

        print(f"[{idx + 1}/{args.max_images}] saved {out_path}")


if __name__ == "__main__":
    main()

"""
python opencood/tools/vis_bev_from_npy_zoomed_view.py \
    --npy_dir opencood/pretrained/feaco/npy \
    --out_dir opencood/pretrained/feaco/vis_bev_circus \
    --zoom_range 40
"""