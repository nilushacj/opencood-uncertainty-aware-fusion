#!/usr/bin/env python3
import copy
import csv
import re
from pathlib import Path
import os
import yaml


# ---------------------------------------------------------------------
# 1. Update these paths
# ---------------------------------------------------------------------
BASE_MODEL_DIR = Path("opencood/pretrained/feaco")
BASE_CONFIG_PATH = BASE_MODEL_DIR / "config_monte_carlo.yaml"
OUTPUT_ROOT_DIR = BASE_MODEL_DIR

CONFIG_FILENAME = "config.yaml"
GENERATED_PREFIX = "generated_"

# Symlink checkpoint files into each generated run directory
SYMLINK_CHECKPOINTS = True

# Usually OpenCOOD checkpoints are .pth files
CHECKPOINT_PATTERNS = ["*.pth"]


# ---------------------------------------------------------------------
# 2. OpenCOOD-like YAML loading/saving
# ---------------------------------------------------------------------


def make_opencood_yaml_loader():
    loader = yaml.Loader

    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))

    return loader


def load_yaml_raw(path: Path):
    """
    Load YAML without applying OpenCOOD's yaml_parser.

    This keeps the raw config structure so we can modify and save variants.
    """
    loader = make_opencood_yaml_loader()
    with open(path, "r") as f:
        return yaml.load(f, Loader=loader)


def save_yaml(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(obj, f, sort_keys=False)


# ---------------------------------------------------------------------
# 3. Strict update helpers
# ---------------------------------------------------------------------
def symlink_checkpoints_to_run_dir(base_model_dir: Path, run_dir: Path):
    """
    Symlink checkpoint files from the base pretrained directory into each
    generated run directory.

    This allows inference.py to use --model_dir generated_run_dir while still
    finding the same trained weights.
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_files = []
    for pattern in CHECKPOINT_PATTERNS:
        checkpoint_files.extend(base_model_dir.glob(pattern))

    if len(checkpoint_files) == 0:
        raise FileNotFoundError(
            f"No checkpoint files found in {base_model_dir} with patterns {CHECKPOINT_PATTERNS}"
        )

    for src in checkpoint_files:
        dst = run_dir / src.name

        if dst.exists() or dst.is_symlink():
            continue

        # Use relative symlink so folders are more portable
        relative_src = Path(os.path.relpath(src.resolve(), start=run_dir.resolve()))
        dst.symlink_to(relative_src)
        
def get_nested(config, key_path):
    cur = config
    for key in key_path:
        if key not in cur:
            raise KeyError(f"Missing key path in base YAML: {'.'.join(key_path)}")
        cur = cur[key]
    return cur


def set_existing(config, key_path, value):
    """
    Set a value only if the full key path already exists.

    This prevents accidentally adding new parameter items to the generated YAML.
    """
    cur = config
    for key in key_path[:-1]:
        if key not in cur:
            raise KeyError(f"Missing key path in base YAML: {'.'.join(key_path)}")
        cur = cur[key]

    last_key = key_path[-1]
    if last_key not in cur:
        raise KeyError(f"Missing key in base YAML: {'.'.join(key_path)}")

    cur[last_key] = value


def fmt_value(x):
    if isinstance(x, bool):
        return "true" if x else "false"

    if isinstance(x, float):
        s = f"{x:g}"
    else:
        s = str(x)

    return s.replace(".", "p").replace("-", "m")


def make_run_name(method_name, noise_name, xyz_std, ryp_std, seed):
    return (
        f"{method_name}"
        f"__{noise_name}"
        f"__xyz{fmt_value(xyz_std)}"
        f"__ryp{fmt_value(ryp_std)}"
        f"__seed{seed}"
    )


# ---------------------------------------------------------------------
# 4. Apply one run's changes
# ---------------------------------------------------------------------
def apply_run_config(base_cfg, run):
    cfg = copy.deepcopy(base_cfg)

    # Sanity checks
    get_nested(cfg, ["model"])
    get_nested(cfg, ["model", "args"])
    get_nested(cfg, ["wild_setting"])

    # --------------------------------------------------
    # Model selection
    # --------------------------------------------------
    set_existing(cfg, ["model", "core_method"], run["core_method"])

    # --------------------------------------------------
    # Logging
    # --------------------------------------------------
    set_existing(cfg, ["model", "args", "enable_cov_logging"], run["enable_cov_logging"])
    set_existing(cfg, ["model", "args", "cov_log_dir"], run["cov_log_dir"])
    set_existing(cfg, ["model", "args", "cov_log_filename"], run["cov_log_filename"])
    set_existing(cfg, ["model", "args", "cov_log_every_n_pairs"], run["cov_log_every_n_pairs"])

    # --------------------------------------------------
    # Continuous uncertainty weighting
    # --------------------------------------------------
    set_existing(cfg, ["model", "args", "enable_uncertainty_weighting"], run["enable_uncertainty_weighting"])
    set_existing(cfg, ["model", "args", "uncertainty_weight_alpha"], run["uncertainty_weight_alpha"])
    set_existing(cfg, ["model", "args", "uncertainty_weight_trace_clip"], run["uncertainty_weight_trace_clip"])
    set_existing(cfg, ["model", "args", "uncertainty_weight_min"], run["uncertainty_weight_min"])

    # --------------------------------------------------
    # Threshold-based uncertainty gating
    # --------------------------------------------------
    set_existing(cfg, ["model", "args", "enable_uncertainty_gating"], run["enable_uncertainty_gating"])
    set_existing(cfg, ["model", "args", "uncertainty_gate_mode"], run["uncertainty_gate_mode"])
    set_existing(cfg, ["model", "args", "uncertainty_gate_threshold"], run["uncertainty_gate_threshold"])
    set_existing(cfg, ["model", "args", "uncertainty_gate_low_weight"], run["uncertainty_gate_low_weight"])

    # --------------------------------------------------
    # Status-based skip
    # --------------------------------------------------
    set_existing(cfg, ["model", "args", "enable_status_based_skip"], run["enable_status_based_skip"])
    set_existing(cfg, ["model", "args", "enable_ok_tail_skip"], run["enable_ok_tail_skip"])
    set_existing(cfg, ["model", "args", "ok_tail_skip_threshold"], run["ok_tail_skip_threshold"])

    # --------------------------------------------------
    # Blur settings
    # --------------------------------------------------
    set_existing(cfg, ["model", "args", "enable_uncertainty_blur"], run["enable_uncertainty_blur"])
    set_existing(cfg, ["model", "args", "enable_ok_tail_vertical_blur"], run["enable_ok_tail_vertical_blur"])
    set_existing(cfg, ["model", "args", "ok_blur_threshold"], run["ok_blur_threshold"])
    set_existing(cfg, ["model", "args", "vertical_blur_use_ty_only"], run["vertical_blur_use_ty_only"])
    set_existing(cfg, ["model", "args", "blur_use_only_ok"], run["blur_use_only_ok"])
    set_existing(cfg, ["model", "args", "blur_sigma_scale"], run["blur_sigma_scale"])
    set_existing(cfg, ["model", "args", "blur_sigma_min"], run["blur_sigma_min"])
    set_existing(cfg, ["model", "args", "blur_sigma_max"], run["blur_sigma_max"])
    set_existing(cfg, ["model", "args", "blur_apply_threshold"], run["blur_apply_threshold"])

    # --------------------------------------------------
    # Tail-only Monte Carlo warp settings
    # --------------------------------------------------
    set_existing(cfg, ["model", "args", "enable_tail_mc_warp"], run["enable_tail_mc_warp"])
    set_existing(cfg, ["model", "args", "mc_trace_threshold"], run["mc_trace_threshold"])
    set_existing(cfg, ["model", "args", "mc_num_samples"], run["mc_num_samples"])
    set_existing(cfg, ["model", "args", "mc_random_seed"], run["mc_random_seed"])
    set_existing(cfg, ["model", "args", "mc_use_translation_only"], run["mc_use_translation_only"])
    set_existing(cfg, ["model", "args", "mc_theta_clip"], run["mc_theta_clip"])
    set_existing(cfg, ["model", "args", "mc_translation_clip"], run["mc_translation_clip"])

    # --------------------------------------------------
    # Wild/noise settings
    # --------------------------------------------------
    set_existing(cfg, ["wild_setting", "loc_err"], run["loc_err"])
    set_existing(cfg, ["wild_setting", "xyz_std"], run["xyz_std"])
    set_existing(cfg, ["wild_setting", "ryp_std"], run["ryp_std"])
    set_existing(cfg, ["wild_setting", "seed"], run["seed"])

    return cfg


# ---------------------------------------------------------------------
# 5. Define experiment matrix
# ---------------------------------------------------------------------
def build_runs():
    seeds = [20,28,42]

    noise_settings = [
        # noise_name, xyz_std, ryp_std, loc_err
        ("clean", 0.0, 0.0, False),

        # Combined position + yaw noise
        ("pos_0p2_rot_1p0", 0.2, 1.0, True),
        ("pos_0p5_rot_2p0", 0.5, 2.0, True),
        ("pos_1p0_rot_4p0", 1.0, 4.0, True),
        
        ("pos_2p0_rot_1p0", 2.0, 1.0, True),
        ("pos_2p0_rot_2p0", 2.0, 2.0, True),
        ("pos_2p0_rot_4p0", 2.0, 4.0, True),
        ("pos_2p0_rot_6p0", 2.0, 6.0, True),

        ("pos_3p0_rot_1p0", 3.0, 1.0, True),
        ("pos_3p0_rot_2p0", 3.0, 2.0, True),
        ("pos_3p0_rot_4p0", 3.0, 4.0, True),
        ("pos_3p0_rot_6p0", 3.0, 6.0, True),

        # ("pos_4p5_rot_1p0", 4.5, 1.0, True),
        # ("pos_4p5_rot_2p0", 4.5, 2.0, True),
        # ("pos_4p5_rot_4p0", 4.5, 4.0, True),
        # ("pos_4p5_rot_6p0", 4.5, 6.0, True),

        # ("pos_6p5_rot_1p0", 6.5, 1.0, True),
        # ("pos_6p5_rot_2p0", 6.5, 2.0, True),
        # ("pos_6p5_rot_4p0", 6.5, 4.0, True),
        # ("pos_6p5_rot_6p0", 6.5, 6.0, True),

        # ("pos_9p0_rot_1p0", 9.0, 1.0, True),
        # ("pos_9p0_rot_2p0", 9.0, 2.0, True),
        # ("pos_9p0_rot_4p0", 9.0, 4.0, True),
        # ("pos_9p0_rot_6p0", 9.0, 6.0, True),


        # Pure position noise
        ("pos_only_0p2", 0.2, 0.0, True),
        ("pos_only_0p5", 0.5, 0.0, True),
        ("pos_only_1p0", 1.0, 0.0, True),
        ("pos_only_3p0", 3.0, 0.0, True),
        # ("pos_only_4p5", 4.5, 0.0, True),
        # ("pos_only_6p5", 6.5, 0.0, True),
        # ("pos_only_9p0", 9.0, 0.0, True),

        # Pure rotation/yaw noise
        ("rot_only_1p0", 0.0, 1.0, True),
        ("rot_only_2p0", 0.0, 2.0, True),
        ("rot_only_4p0", 0.0, 4.0, True),
        ("rot_only_6p0", 0.0, 6.0, True),
        ("rot_only_8p0", 0.0, 8.0, True),
    ]

    common_defaults = {
        "enable_cov_logging": True,
        "cov_log_dir": "debug_prm/cov_logs",
        "cov_log_every_n_pairs": 1,

        "enable_uncertainty_weighting": False,
        "uncertainty_weight_alpha": 0.10,
        "uncertainty_weight_trace_clip": 50.0,
        "uncertainty_weight_min": 0.2,

        "enable_uncertainty_gating": False,
        "uncertainty_gate_mode": "binary_weight",
        "uncertainty_gate_threshold": 50.010002,
        "uncertainty_gate_low_weight": 0.5,

        "enable_status_based_skip": False,
        "enable_ok_tail_skip": False,
        "ok_tail_skip_threshold": 100.0,

        "enable_uncertainty_blur": False,
        "enable_ok_tail_vertical_blur": False,
        "ok_blur_threshold": 9.800544452667232,
        "vertical_blur_use_ty_only": True,
        "blur_use_only_ok": True,
        "blur_sigma_scale": 0.05,
        "blur_sigma_min": 0.0,
        "blur_sigma_max": 0.75,
        "blur_apply_threshold": 0.10,

        "enable_tail_mc_warp": False,
        "mc_trace_threshold": 9.800544452667232,
        "mc_num_samples": 3,
        "mc_random_seed": 123,
        "mc_use_translation_only": True,
        "mc_theta_clip": 0.15,
        "mc_translation_clip": 3.0,
    }

    methods = [
        # {
        #     "method_name": "feaco_baseline",
        #     "core_method": "point_pillar_where2comm_feaco",
        # },
        # {
        #     "method_name": "level2_only",
        #     "core_method": "point_pillar_where2comm_our",
        # },
        # {
        #     "method_name": "level2_status_skip",
        #     "core_method": "point_pillar_where2comm_our",
        #     "enable_status_based_skip": True,
        # },
        # {
        #     "method_name": "level2_status_skip_vertical_blur_bss0p05",
        #     "core_method": "point_pillar_where2comm_our",
        #     "enable_status_based_skip": True,
        #     "enable_ok_tail_vertical_blur": True,
        #     "vertical_blur_use_ty_only": True,
        #     "blur_use_only_ok": True,
        #     "blur_sigma_scale": 0.05,
        # },        
        {
            "method_name": "mc_status_skip_transonly_k3",
            "core_method": "point_pillar_where2comm_our",
            "enable_status_based_skip": True,
            "enable_ok_tail_vertical_blur": False,
            "enable_uncertainty_blur": False,

            "enable_tail_mc_warp": True,
            "mc_trace_threshold": 9.800544452667232,
            "mc_num_samples": 3,
            "mc_random_seed": 123,
            "mc_use_translation_only": True,
            "mc_theta_clip": 0.15,
            "mc_translation_clip": 3.0,
        },
        {
            "method_name": "mc_status_skip_full_k3",
            "core_method": "point_pillar_where2comm_our",
            "enable_status_based_skip": True,
            "enable_ok_tail_vertical_blur": False,
            "enable_uncertainty_blur": False,

            "enable_tail_mc_warp": True,
            "mc_trace_threshold": 9.800544452667232,
            "mc_num_samples": 3,
            "mc_random_seed": 123,
            "mc_use_translation_only": False,
            "mc_theta_clip": 0.15,
            "mc_translation_clip": 3.0,
        },
    ]

    runs = []

    for seed in seeds:
        for noise_name, xyz_std, ryp_std, loc_err in noise_settings:
            for method in methods:
                run_name = make_run_name(
                    method["method_name"],
                    noise_name,
                    xyz_std,
                    ryp_std,
                    seed,
                )

                run = copy.deepcopy(common_defaults)
                run.update(method)
                run.update(
                    {
                        "run_name": run_name,
                        "noise_name": noise_name,
                        "xyz_std": xyz_std,
                        "ryp_std": ryp_std,
                        "loc_err": loc_err,
                        "seed": seed,
                        "cov_log_filename": f"{run_name}.csv",
                    }
                )

                runs.append(run)

    return runs


# ---------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------
def main():
    OUTPUT_ROOT_DIR.mkdir(parents=True, exist_ok=True)

    base_cfg = load_yaml_raw(BASE_CONFIG_PATH)
    runs = build_runs()

    print(f"Loaded base config: {BASE_CONFIG_PATH}")
    print(f"Writing generated configs to: {OUTPUT_ROOT_DIR}")
    print(f"Number of configs to generate: {len(runs)}")

    manifest_rows = []

    for run in runs:
        run_name = run["run_name"]
        cfg = apply_run_config(base_cfg, run)

        run_dir = OUTPUT_ROOT_DIR / f"{GENERATED_PREFIX}{run_name}"
        config_path = run_dir / CONFIG_FILENAME

        if config_path.resolve() == BASE_CONFIG_PATH.resolve():
            raise RuntimeError("Refusing to overwrite the base config.")

        save_yaml(cfg, config_path)

        if SYMLINK_CHECKPOINTS:
            symlink_checkpoints_to_run_dir(BASE_MODEL_DIR, run_dir)

        manifest_rows.append(
            {
                "run_name": run_name,
                "run_dir": str(run_dir),
                "config_path": str(config_path),
                "method_name": run["method_name"],
                "core_method": run["core_method"],
                "noise_name": run["noise_name"],
                "loc_err": run["loc_err"],
                "xyz_std": run["xyz_std"],
                "ryp_std": run["ryp_std"],
                "seed": run["seed"],
                "cov_log_filename": run["cov_log_filename"],
                "enable_status_based_skip": run["enable_status_based_skip"],
                "enable_ok_tail_vertical_blur": run["enable_ok_tail_vertical_blur"],
                "vertical_blur_use_ty_only": run["vertical_blur_use_ty_only"],
                "blur_sigma_scale": run["blur_sigma_scale"],
                "enable_tail_mc_warp": run["enable_tail_mc_warp"],
                "mc_trace_threshold": run["mc_trace_threshold"],
                "mc_num_samples": run["mc_num_samples"],
                "mc_random_seed": run["mc_random_seed"],
                "mc_use_translation_only": run["mc_use_translation_only"],
                "mc_theta_clip": run["mc_theta_clip"],
                "mc_translation_clip": run["mc_translation_clip"],
            }
        )

        print(f"Wrote: {config_path}")

    manifest_path = OUTPUT_ROOT_DIR / f"{GENERATED_PREFIX}manifest.csv"

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("\nDone.")
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()

"""
Usage:
    python opencood/tools/config_creator.py 
"""