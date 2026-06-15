#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


AP_COLS = ["AP@0.3", "AP@0.5", "AP@0.7"]


def parse_run_name(run_name: str):
    """
    Expected pattern:
      uncertainty_fusion_attbias_a0p05_b1p0__pos_1p0_rot_4p0__xyz1__ryp4__seed20

    Returns:
      method_name, noise_name, xyz_std, ryp_std, seed
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
        if not part.startswith(prefix):
            raise ValueError(f"Expected prefix {prefix} in {part}")
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


def parse_alpha_bias_from_method(method_name: str):
    """
    Expected method pattern:
      uncertainty_fusion_attbias_a0p05_b1p0
      uncertainty_fusion_attbias_a0p10_b2p0

    Returns:
      alpha, bias_strength
    """
    pattern = r"uncertainty_fusion_attbias_a(?P<alpha>[0-9p]+)_b(?P<bias>[0-9p]+)"
    match = re.search(pattern, method_name)

    if match is None:
        return np.nan, np.nan

    alpha = float(match.group("alpha").replace("p", "."))
    bias = float(match.group("bias").replace("p", "."))
    return alpha, bias


def to_numeric_ap(df: pd.DataFrame):
    for col in AP_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_clean_ap_and_drops(df: pd.DataFrame):
    """
    Adds clean AP values and AP drops from clean for each:
      method_name, alpha, bias_strength, seed

    Drop = clean_AP - noisy_AP.
    Smaller drop means better robustness.
    """
    clean = df[df["noise_name"] == "clean"][
        ["method_name", "alpha", "bias_strength", "seed"] + AP_COLS
    ].copy()

    clean = clean.rename(
        columns={
            "AP@0.3": "clean_AP@0.3",
            "AP@0.5": "clean_AP@0.5",
            "AP@0.7": "clean_AP@0.7",
        }
    )

    merged = df.merge(
        clean,
        on=["method_name", "alpha", "bias_strength", "seed"],
        how="left",
    )

    for thr in ["0.3", "0.5", "0.7"]:
        merged[f"drop_AP@{thr}"] = merged[f"clean_AP@{thr}"] - merged[f"AP@{thr}"]

    return merged


def classify_noise(row):
    """
    Optional helper category.
    Useful for seeing whether a hyperparameter is best for low/medium/high noise.
    You can change these thresholds later.
    """
    xyz = float(row["xyz_std"])
    ryp = float(row["ryp_std"])

    if row["noise_name"] == "clean":
        return "clean"

    if xyz >= 2.0 or ryp >= 8.0:
        return "high_noise"

    if xyz >= 1.0 or ryp >= 4.0:
        return "medium_noise"

    return "low_noise"


def pct(x):
    return 100.0 * x


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_csv",
        help="Input summary_metrics_with_ap_uncertainty_fusion_*.csv file",
    )

    parser.add_argument(
        "--output_prefix",
        default="uncertainty_fusion_hparam_summary",
        help="Prefix for output CSV files",
    )

    parser.add_argument(
        "--method_prefix",
        default="uncertainty_fusion_attbias",
        help="Only runs whose method_name starts with this will be analyzed",
    )

    parser.add_argument(
        "--score_ap07_weight",
        type=float,
        default=0.50,
        help="Weight of mean AP@0.7 in final ranking score",
    )

    parser.add_argument(
        "--score_ap05_weight",
        type=float,
        default=0.30,
        help="Weight of mean AP@0.5 in final ranking score",
    )

    parser.add_argument(
        "--score_ap03_weight",
        type=float,
        default=0.20,
        help="Weight of mean AP@0.3 in final ranking score",
    )

    parser.add_argument(
        "--drop_penalty_weight",
        type=float,
        default=0.50,
        help="Penalty weight for mean AP drop from clean. Higher means prioritize robustness more.",
    )

    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_prefix = Path(args.output_prefix)

    df = pd.read_csv(input_csv)
    df = to_numeric_ap(df)

    if "Run name" not in df.columns:
        raise RuntimeError("Input CSV must contain a 'Run name' column.")

    parsed = df["Run name"].apply(parse_run_name).apply(pd.Series)
    df = pd.concat([df, parsed], axis=1)

    # Keep only uncertainty-fusion runs
    df = df[df["method_name"].str.startswith(args.method_prefix)].copy()

    if df.empty:
        raise RuntimeError(
            f"No rows found with method_name starting with '{args.method_prefix}'."
        )

    # Prefer logged hyperparameter columns if they exist.
    # Fallback: parse from method name.
    if "uncertainty_conf_alpha" in df.columns:
        df["alpha"] = pd.to_numeric(df["uncertainty_conf_alpha"], errors="coerce")
    else:
        df["alpha"] = np.nan

    if "uncertainty_attention_bias_strength" in df.columns:
        df["bias_strength"] = pd.to_numeric(
            df["uncertainty_attention_bias_strength"],
            errors="coerce",
        )
    else:
        df["bias_strength"] = np.nan

    parsed_alpha_bias = df["method_name"].apply(parse_alpha_bias_from_method)
    df["alpha_from_name"] = parsed_alpha_bias.apply(lambda x: x[0])
    df["bias_from_name"] = parsed_alpha_bias.apply(lambda x: x[1])

    df["alpha"] = df["alpha"].fillna(df["alpha_from_name"])
    df["bias_strength"] = df["bias_strength"].fillna(df["bias_from_name"])

    if df["alpha"].isna().any() or df["bias_strength"].isna().any():
        bad = df[df["alpha"].isna() | df["bias_strength"].isna()][["Run name", "method_name"]]
        raise RuntimeError(
            "Could not parse alpha/bias for some rows:\n"
            + bad.to_string(index=False)
        )

    df["noise_group"] = df.apply(classify_noise, axis=1)

    # Add AP drops from clean for each combo and seed
    df = add_clean_ap_and_drops(df)

    # -----------------------------
    # 1. Summary by combo + noise
    # -----------------------------
    by_noise = (
        df.groupby(
            [
                "method_name",
                "alpha",
                "bias_strength",
                "noise_name",
                "xyz_std",
                "ryp_std",
                "noise_group",
            ],
            as_index=False,
        )
        .agg(
            num_seeds=("seed", "nunique"),
            mean_AP_03=("AP@0.3", "mean"),
            std_AP_03=("AP@0.3", "std"),
            mean_AP_05=("AP@0.5", "mean"),
            std_AP_05=("AP@0.5", "std"),
            mean_AP_07=("AP@0.7", "mean"),
            std_AP_07=("AP@0.7", "std"),
            mean_drop_03=("drop_AP@0.3", "mean"),
            mean_drop_05=("drop_AP@0.5", "mean"),
            mean_drop_07=("drop_AP@0.7", "mean"),
            mean_status_ok_ratio=("Status ok ratio", "mean")
                if "Status ok ratio" in df.columns else ("AP@0.3", "size"),
            mean_fusion_reliability=("fusion_reliability", "mean")
                if "fusion_reliability" in df.columns else ("AP@0.3", "size"),
        )
    )

    # Convert some values to percentages for readability
    for col in [
        "mean_AP_03", "std_AP_03", "mean_AP_05", "std_AP_05", "mean_AP_07", "std_AP_07",
        "mean_drop_03", "mean_drop_05", "mean_drop_07",
    ]:
        by_noise[col + "_pct"] = by_noise[col].map(pct)

    # -----------------------------
    # 2. Overall combo ranking
    # -----------------------------
    combo = (
        df.groupby(["method_name", "alpha", "bias_strength"], as_index=False)
        .agg(
            num_runs=("Run name", "count"),
            num_noise_settings=("noise_name", "nunique"),
            num_seeds=("seed", "nunique"),
            mean_AP_03=("AP@0.3", "mean"),
            mean_AP_05=("AP@0.5", "mean"),
            mean_AP_07=("AP@0.7", "mean"),
            median_AP_03=("AP@0.3", "median"),
            median_AP_05=("AP@0.5", "median"),
            median_AP_07=("AP@0.7", "median"),
            mean_drop_03=("drop_AP@0.3", "mean"),
            mean_drop_05=("drop_AP@0.5", "mean"),
            mean_drop_07=("drop_AP@0.7", "mean"),
            max_drop_03=("drop_AP@0.3", "max"),
            max_drop_05=("drop_AP@0.5", "max"),
            max_drop_07=("drop_AP@0.7", "max"),
            mean_status_ok_ratio=("Status ok ratio", "mean")
                if "Status ok ratio" in df.columns else ("AP@0.3", "size"),
            mean_fusion_reliability=("fusion_reliability", "mean")
                if "fusion_reliability" in df.columns else ("AP@0.3", "size"),
        )
    )

    # High-noise-only summary
    high_noise_df = df[df["noise_group"] == "high_noise"].copy()
    if not high_noise_df.empty:
        high_combo = (
            high_noise_df.groupby(["method_name", "alpha", "bias_strength"], as_index=False)
            .agg(
                high_noise_mean_AP_03=("AP@0.3", "mean"),
                high_noise_mean_AP_05=("AP@0.5", "mean"),
                high_noise_mean_AP_07=("AP@0.7", "mean"),
                high_noise_mean_drop_03=("drop_AP@0.3", "mean"),
                high_noise_mean_drop_05=("drop_AP@0.5", "mean"),
                high_noise_mean_drop_07=("drop_AP@0.7", "mean"),
            )
        )
        combo = combo.merge(high_combo, on=["method_name", "alpha", "bias_strength"], how="left")
    else:
        combo["high_noise_mean_AP_03"] = np.nan
        combo["high_noise_mean_AP_05"] = np.nan
        combo["high_noise_mean_AP_07"] = np.nan
        combo["high_noise_mean_drop_03"] = np.nan
        combo["high_noise_mean_drop_05"] = np.nan
        combo["high_noise_mean_drop_07"] = np.nan

    # Ranking score:
    # higher AP is good; lower AP drop is good.
    combo["weighted_mean_AP"] = (
        args.score_ap03_weight * combo["mean_AP_03"]
        + args.score_ap05_weight * combo["mean_AP_05"]
        + args.score_ap07_weight * combo["mean_AP_07"]
    )

    combo["weighted_mean_drop"] = (
        args.score_ap03_weight * combo["mean_drop_03"]
        + args.score_ap05_weight * combo["mean_drop_05"]
        + args.score_ap07_weight * combo["mean_drop_07"]
    )

    combo["ranking_score"] = (
        combo["weighted_mean_AP"]
        - args.drop_penalty_weight * combo["weighted_mean_drop"]
    )

    # Useful percentage columns
    percent_cols = [
        "mean_AP_03", "mean_AP_05", "mean_AP_07",
        "median_AP_03", "median_AP_05", "median_AP_07",
        "mean_drop_03", "mean_drop_05", "mean_drop_07",
        "max_drop_03", "max_drop_05", "max_drop_07",
        "high_noise_mean_AP_03", "high_noise_mean_AP_05", "high_noise_mean_AP_07",
        "high_noise_mean_drop_03", "high_noise_mean_drop_05", "high_noise_mean_drop_07",
        "weighted_mean_AP", "weighted_mean_drop", "ranking_score",
    ]

    for col in percent_cols:
        if col in combo.columns:
            combo[col + "_pct"] = combo[col].map(pct)

    combo = combo.sort_values(
        by=["ranking_score", "mean_AP_07", "high_noise_mean_AP_07"],
        ascending=[False, False, False],
    )

    # -----------------------------
    # 3. Best combo per noise setting
    # -----------------------------
    # For each noise setting, choose best by AP@0.7 first, then AP@0.5.
    best_by_noise = (
        by_noise.sort_values(
            by=["noise_name", "mean_AP_07", "mean_AP_05", "mean_AP_03"],
            ascending=[True, False, False, False],
        )
        .groupby("noise_name", as_index=False)
        .head(1)
        .copy()
    )

    # -----------------------------
    # Save outputs
    # -----------------------------
    combo_path = output_prefix.with_name(output_prefix.name + "_combo_ranking.csv")
    by_noise_path = output_prefix.with_name(output_prefix.name + "_by_noise.csv")
    best_by_noise_path = output_prefix.with_name(output_prefix.name + "_best_by_noise.csv")

    combo.to_csv(combo_path, index=False)
    by_noise.to_csv(by_noise_path, index=False)
    best_by_noise.to_csv(best_by_noise_path, index=False)

    print(f"Wrote: {combo_path}")
    print(f"Wrote: {by_noise_path}")
    print(f"Wrote: {best_by_noise_path}")

    print("\nTop hyperparameter combinations:")
    cols_to_print = [
        "method_name",
        "alpha",
        "bias_strength",
        "num_noise_settings",
        "num_seeds",
        "mean_AP_03_pct",
        "mean_AP_05_pct",
        "mean_AP_07_pct",
        "mean_drop_03_pct",
        "mean_drop_05_pct",
        "mean_drop_07_pct",
        "high_noise_mean_AP_07_pct",
        "ranking_score_pct",
    ]

    cols_to_print = [c for c in cols_to_print if c in combo.columns]
    print(
        combo[cols_to_print]
        .head(20)
        .to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )


if __name__ == "__main__":
    main()

"""
Usage:
python summarize_results_from_uncertainty_fusion_grid_csv.py \
  summary_metrics_with_ap_uncertainty_fusion_10_06_2026.csv \
  --output_prefix uncertainty_fusion_grid_10_06_2026
"""