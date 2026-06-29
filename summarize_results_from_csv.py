from pathlib import Path
import pandas as pd
import numpy as np
import sys

# -----------------------------
# Configuration
# -----------------------------
input_dir = Path("debug_prm/cov_logs")
generated_parent_dir = Path("opencood/pretrained/feaco")  # parent directory containing generated_* folders
output_csv = "summary_metrics_with_ap_uncertainty_fusion_29_06_2026.csv"


# -----------------------------
# Helper functions
# -----------------------------
def numeric_binary_ratio_if_column_exists(df, column_name):
    if column_name not in df.columns:
        return "N/A"

    values = pd.to_numeric(df[column_name], errors="coerce").dropna()

    if values.empty:
        return "N/A"

    return values.mean()


def numeric_stat_filtered(df, column_name, stat, filter_column=None, filter_value=None):
    if column_name not in df.columns:
        return "N/A"

    temp = df

    if filter_column is not None:
        if filter_column not in df.columns:
            return "N/A"
        temp = temp[temp[filter_column] == filter_value]

    values = pd.to_numeric(temp[column_name], errors="coerce").dropna()

    if values.empty:
        return "N/A"

    if stat == "mean":
        return values.mean()
    elif stat == "median":
        return values.median()
    elif stat == "p95":
        return values.quantile(0.95)
    else:
        raise ValueError(f"Unknown stat: {stat}")

def ratio_if_column_exists(df, column_name, target_value):
    """
    Returns ratio of rows where df[column_name] == target_value.
    Returns 'N/A' if the column does not exist.
    """
    if column_name not in df.columns:
        return "N/A"

    total_rows = len(df)
    if total_rows == 0:
        return "N/A"

    return (df[column_name] == target_value).sum() / total_rows


def numeric_stat_if_column_exists(df, column_name, stat):
    """
    Returns mean, median, or p95 for a numeric column.
    Returns 'N/A' if the column does not exist or has no valid numeric values.
    """
    if column_name not in df.columns:
        return "N/A"

    values = pd.to_numeric(df[column_name], errors="coerce").dropna()

    if values.empty:
        return "N/A"

    if stat == "mean":
        return values.mean()
    elif stat == "median":
        return values.median()
    elif stat == "p95":
        return np.percentile(values, 95)
    else:
        raise ValueError(f"Unknown stat: {stat}")


def read_ap_values(run_name, generated_parent_dir):
    """
    Given a run name like:
        feaco_baseline__clean__xyz0__ryp0__seed20

    Reads:
        generated_feaco_baseline__clean__xyz0__ryp0__seed20/
            eval_global_sort_feaco_baseline__clean__xyz0__ryp0__seed20.yaml

    Returns AP@0.3, AP@0.5, AP@0.7.
    If the directory, YAML file, or key is missing, returns 'N/A' for that value.
    """
    generated_dir = generated_parent_dir / f"generated_{run_name}"
    eval_yaml = generated_dir / f"eval_global_sort_{run_name}.yaml"

    ap_values = {
        "AP@0.3": "N/A",
        "AP@0.5": "N/A",
        "AP@0.7": "N/A",
    }

    if not eval_yaml.exists():
        print(f"Warning: eval YAML not found: {eval_yaml}")
        return ap_values

    with open(eval_yaml, "r") as f:
        for line in f:
            line = line.strip()

            if not line or ":" not in line:
                continue

            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()

            try:
                value = float(value)
            except ValueError:
                continue

            if key == "ap30":
                ap_values["AP@0.3"] = value
            elif key == "ap_50":
                ap_values["AP@0.5"] = value
            elif key == "ap_70":
                ap_values["AP@0.7"] = value

    return ap_values


# -----------------------------
# Main processing
# -----------------------------
summary_rows = []

csv_inputs = sorted(input_dir.glob("*.csv"))
print(f"Found {len(csv_inputs)} CSV files")

for csv_path in csv_inputs:
    df = pd.read_csv(csv_path)
    total_rows = len(df)

    run_name = csv_path.stem
    ap_values = read_ap_values(run_name, generated_parent_dir)

    row = {
        "Run name": run_name,
        "Collab pairs": total_rows,

        "Status ok ratio": ratio_if_column_exists(
            df, "status", "ok"
        ),

        "Non-ok ratio": (
            1.0 - ratio_if_column_exists(df, "status", "ok")
            if ratio_if_column_exists(df, "status", "ok") != "N/A"
            else "N/A"
        ),

        "Fused ratio": numeric_binary_ratio_if_column_exists(
            df, "fused_into_feature_list"
        ),

        "Skipped ratio": numeric_binary_ratio_if_column_exists(
            df, "gate_skipped"
        ),

        "Selected for blur ratio": ratio_if_column_exists(
            df, "vertical_blur_decision", "apply_ok_tail_vertical_blur"
        ),

        "Blurred ratio": numeric_binary_ratio_if_column_exists(
            df, "blur_applied"
        ),

        "Mean cov trace": numeric_stat_if_column_exists(
            df, "cov_trace", "mean"
        ),

        "Median cov trace": numeric_stat_if_column_exists(
            df, "cov_trace", "median"
        ),

        "p95 cov trace all": numeric_stat_if_column_exists(
            df, "cov_trace", "p95"
        ),

        "p95 cov trace ok-only": numeric_stat_filtered(
            df, "cov_trace", "p95", filter_column="status", filter_value="ok"
        ),

        "Mean mask IoU after warp": numeric_stat_if_column_exists(
            df, "mask_iou_after_warp", "mean"
        ),

        "Median mask IoU after warp": numeric_stat_if_column_exists(
            df, "mask_iou_after_warp", "median"
        ),

        "Mean mask IoU after warp ok-only": numeric_stat_filtered(
            df, "mask_iou_after_warp", "mean", filter_column="status", filter_value="ok"
        ),

        "Median mask IoU after warp ok-only": numeric_stat_filtered(
            df, "mask_iou_after_warp", "median", filter_column="status", filter_value="ok"
        ),

        "AP@0.3": ap_values["AP@0.3"],
        "AP@0.5": ap_values["AP@0.5"],
        "AP@0.7": ap_values["AP@0.7"],
    }

    summary_rows.append(row)


summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(output_csv, index=False)

print(f"Saved summary CSV to: {output_csv}")