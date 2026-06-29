#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import pandas as pd


def parse_run_name(run_name: str):
    """
    Expected pattern, for example:
      feaco_baseline__high_combined__xyz1__ryp4__seed20
      level2_status_skip_vertical_blur_bss0p05__clean__xyz0__ryp0__seed28
    """
    parts = run_name.split("__")
    if len(parts) < 5:
        raise ValueError(f"Unexpected run name format: {run_name}")

    method_name = parts[0]
    noise_name = parts[1]

    xyz_part = parts[2]
    ryp_part = parts[3]
    seed_part = parts[4]

    def parse_num(part: str, prefix: str):
        assert part.startswith(prefix), f"Expected prefix {prefix} in {part}"
        val = part[len(prefix):].replace("p", ".").replace("m", "-")
        return float(val)

    xyz_std = parse_num(xyz_part, "xyz")
    ryp_std = parse_num(ryp_part, "ryp")
    seed = int(seed_part.replace("seed", ""))

    return {
        "method_name": method_name,
        "noise_name": noise_name,
        "xyz_std": xyz_std,
        "ryp_std": ryp_std,
        "seed": seed,
    }


def pct(x: float) -> float:
    return 100.0 * x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_csv",
        help="Path to summary_metrics_with_ap.csv",
    )
    parser.add_argument(
        "--output_csv",
        default="noise_level_summary_with_robustness_gains.csv",
        help="Path to output CSV",
    )
    parser.add_argument(
        "--baseline_method",
        default="feaco_baseline",
        help="Baseline method_name in 'Run name'",
    )
    parser.add_argument(
        "--ours_method",
        default="level2_status_skip_vertical_blur_bss0p05",
        help="Updated/proposed method_name in 'Run name'",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    # Parse run metadata from "Run name"
    parsed = df["Run name"].apply(parse_run_name).apply(pd.Series)
    df = pd.concat([df, parsed], axis=1)

    # Keep only the two methods we want to compare
    df = df[df["method_name"].isin([args.baseline_method, args.ours_method])].copy()

    # Split into baseline and ours
    base = df[df["method_name"] == args.baseline_method].copy()
    ours = df[df["method_name"] == args.ours_method].copy()

    # Merge baseline and ours on noise + seed
    merged = base.merge(
        ours,
        on=["noise_name", "xyz_std", "ryp_std", "seed"],
        suffixes=("_base", "_ours"),
        how="inner",
    )

    if merged.empty:
        raise RuntimeError("No matching baseline/proposed run pairs found.")

    # Get clean AP for each seed, separately for baseline and ours
    clean_base = (
        base[base["noise_name"] == "clean"][["seed", "AP@0.3", "AP@0.5", "AP@0.7"]]
        .rename(columns={
            "AP@0.3": "AP@0.3_clean_base",
            "AP@0.5": "AP@0.5_clean_base",
            "AP@0.7": "AP@0.7_clean_base",
        })
    )

    clean_ours = (
        ours[ours["noise_name"] == "clean"][["seed", "AP@0.3", "AP@0.5", "AP@0.7"]]
        .rename(columns={
            "AP@0.3": "AP@0.3_clean_ours",
            "AP@0.5": "AP@0.5_clean_ours",
            "AP@0.7": "AP@0.7_clean_ours",
        })
    )

    merged = merged.merge(clean_base, on="seed", how="left")
    merged = merged.merge(clean_ours, on="seed", how="left")

    # Delta AP = ours - baseline
    merged["dAP_03"] = merged["AP@0.3_ours"] - merged["AP@0.3_base"]
    merged["dAP_05"] = merged["AP@0.5_ours"] - merged["AP@0.5_base"]
    merged["dAP_07"] = merged["AP@0.7_ours"] - merged["AP@0.7_base"]

    # AP drop from clean
    merged["drop_03_base"] = merged["AP@0.3_clean_base"] - merged["AP@0.3_base"]
    merged["drop_05_base"] = merged["AP@0.5_clean_base"] - merged["AP@0.5_base"]
    merged["drop_07_base"] = merged["AP@0.7_clean_base"] - merged["AP@0.7_base"]

    merged["drop_03_ours"] = merged["AP@0.3_clean_ours"] - merged["AP@0.3_ours"]
    merged["drop_05_ours"] = merged["AP@0.5_clean_ours"] - merged["AP@0.5_ours"]
    merged["drop_07_ours"] = merged["AP@0.7_clean_ours"] - merged["AP@0.7_ours"]

    # Robustness gain = baseline drop - ours drop
    merged["dRG_03"] = merged["drop_03_base"] - merged["drop_03_ours"]
    merged["dRG_05"] = merged["drop_05_base"] - merged["drop_05_ours"]
    merged["dRG_07"] = merged["drop_07_base"] - merged["drop_07_ours"]

    # Average across seeds per noise setting
    summary = (
        merged.groupby("noise_name", as_index=False)
        .agg({
            "dAP_03": "mean",
            "dRG_03": "mean",
            "dAP_05": "mean",
            "dRG_05": "mean",
            "dAP_07": "mean",
            "dRG_07": "mean",
            "xyz_std": "first",
            "ryp_std": "first",
            "seed": "count",
        })
        .rename(columns={"seed": "num_seeds"})
    )

    # Optional ordering
    preferred_order = [
        "clean",
        "rot_only_1p0",
        "rot_only_2p0",
        "rot_only_4p0",
        "pos_only_0p2",
        "low_combined",
        "pos_only_0p5",
        "medium_combined",
        "pos_only_1p0",
        "high_combined",
        "pos_only_3p0",
        "pos_only_10p0",
    ]
    order_map = {name: i for i, name in enumerate(preferred_order)}
    summary["sort_key"] = summary["noise_name"].map(lambda x: order_map.get(x, 999))
    summary = summary.sort_values(["sort_key", "xyz_std", "ryp_std"]).drop(columns=["sort_key"])

    # Convert to percentages
    out = pd.DataFrame({
        "Noise setting": summary["noise_name"],
        "ΔAP@0.3": summary["dAP_03"].map(pct),
        "ΔRG@0.3": summary["dRG_03"].map(pct),
        "ΔAP@0.5": summary["dAP_05"].map(pct),
        "ΔRG@0.5": summary["dRG_05"].map(pct),
        "ΔAP@0.7": summary["dAP_07"].map(pct),
        "ΔRG@0.7": summary["dRG_07"].map(pct),
    })

    out.to_csv(args.output_csv, index=False)

    print(f"Wrote: {args.output_csv}")
    print(out.to_string(index=False, float_format=lambda x: f"{x:+.2f}%"))


if __name__ == "__main__":
    main()

"""
Usage:
 python summarize_ap_summary.py ./summary_metrics_with_ap.csv --output_csv noise_level_summary_with_robustness_gains_28_04_2026.csv
 python summarize_ap_summary.py ./summary_metrics_with_ap_28_04_2026.csv --output_csv noise_level_summary_with_robustness_gains_28_04_2026.csv

 python summarize_ap_summary.py ./summary_metrics_with_ap_28_04_2026.csv --output_csv noise_level_summary_cov_only_to_skip_and_blur_robustness_gains_28_04_2026.csv --baseline_method level2_only --ours_method level2_status_skip_vertical_blur_bss0p05
 
 python summarize_ap_summary.py ./summary_metrics_with_ap_28_04_2026.csv --output_csv noise_level_summary_baseline_to_cov_only_robustness_gains_28_04_2026.csv --baseline_method feaco_baseline --ours_method level2_only

 ----------------
 python summarize_ap_summary.py ./summary_metrics_with_ap_08_06_2026.csv --output_csv noise_level_summary_baseline_to_uncertainty_fusion__08_06_2026.csv --baseline_method feaco_baseline --ours_method uncertainty_fusion_attbias_a0p10_b1p0
 python summarize_ap_summary.py ./summary_metrics_with_ap_08_06_2026.csv --output_csv noise_level_summary_level2_status_skip_vertical_blur_bss0p05_to_uncertainty_fusion__08_06_2026.csv --baseline_method level2_status_skip_vertical_blur_bss0p05 --ours_method uncertainty_fusion_attbias_a0p10_b1p0
 ----------------

 ----------------
 python summarize_ap_summary.py ./summary_metrics_with_ap_uncertainty_fusion_29_06_2026.csv --output_csv noise_level_summary_baseline_to_uncertainty_fusion__29_06_2026.csv --baseline_method feaco_baseline --ours_method uncertainty_fusion_attbias_a0p20_b3p5
 python summarize_ap_summary.py ./summary_metrics_with_ap_uncertainty_fusion_29_06_2026.csv --output_csv noise_level_summary_level2_status_skip_to_uncertainty_fusion__29_06_2026.csv --baseline_method level2_status_skip --ours_method uncertainty_fusion_attbias_a0p20_b3p5
 python summarize_ap_summary.py ./summary_metrics_with_ap_uncertainty_fusion_29_06_2026.csv --output_csv noise_level_summary_level2_status_skip_to_level2_status_skip_vertical_blur_bss0p05__29_06_2026.csv --baseline_method level2_status_skip --ours_method level2_status_skip_vertical_blur_bss0p05
 python summarize_ap_summary.py ./summary_metrics_with_ap_uncertainty_fusion_29_06_2026.csv --output_csv noise_level_summary_baseline_to_level2_status_skip__29_06_2026.csv --baseline_method feaco_baseline --ours_method level2_status_skip
 python summarize_ap_summary.py ./summary_metrics_with_ap_uncertainty_fusion_29_06_2026.csv --output_csv noise_level_summary_baseline_to_level2_status_skip_vertical_blur_bss0p05__29_06_2026.csv --baseline_method feaco_baseline --ours_method level2_status_skip_vertical_blur_bss0p05
 python summarize_ap_summary.py ./summary_metrics_with_ap_uncertainty_fusion_29_06_2026.csv --output_csv noise_level_summary_level2_status_skip_vertical_blur_bss0p05__29_06_2026_to_uncertainty_fusion.csv --baseline_method level2_status_skip_vertical_blur_bss0p05 --ours_method uncertainty_fusion_attbias_a0p20_b3p5


 ----------------






"""