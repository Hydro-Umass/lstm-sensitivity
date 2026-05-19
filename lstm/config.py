import argparse
import tomllib
import tomli_w
import subprocess

FORCINGS = ["daymet", "nldas", "maurer"]

STATIC_VARS = [
    "p_mean",
    "pet_mean",
    "aridity",
    "frac_snow",
    "elev_mean",
    "slope_mean",
    "area_gages2",
    "sand_frac",
    "silt_frac",
    "clay_frac",
]

def parse_args_lstm():
    parser = argparse.ArgumentParser(
        description="Train an Entity-Aware LSTM (EA-LSTM) on CAMELS data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "forcing",
        type=str,
        choices=FORCINGS,
        help="Meteorological forcing dataset to use for training.",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing the HDF5 files (<forcing>.h5).",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Load the entire dataset into RAM before training. "
             "Recommended if you have >64 GB of RAM available.",
    )

    parser.add_argument(
        "--hidden-size",
        type=int,
        default=256,
        help="Number of hidden units in the LSTM.",
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.1,
        help="Dropout rate applied to the LSTM output.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Mini-batch size.",
    )
    parser.add_argument(
        "--clip-gradient-norm",
        type=float,
        default=1.0,
        help="Global gradient norm clipping threshold.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=5678,
        help="Random seed for reproducibility.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory where the trained model and config will be saved.",
    )

    # Perturbation args for sensitivity experiments
    parser.add_argument(
        "--bias",
        type=float,
        nargs="+",
        default=None,
        metavar="B",
        help="One or more precipitation bias multipliers to sweep over.",
    )

    parser.add_argument(
        "--stddev",
        type=float,
        nargs="+",
        default=None,
        metavar="S",
        help="One or more random error standard deviations to sweep over.",
    )

    return parser.parse_args()


def parse_args_vic():
    parser = argparse.ArgumentParser(
        description="Calibrate VIC hydrologic model on CAMELS data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "forcing",
        type=str,
        choices=FORCINGS,
        help="Meteorological forcing dataset to use for calibration.",
    )

    parser.add_argument(
        "--vic-exec",
        type=str,
        default="vicNl",
        help="Path to the VIC executable. ",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing the CAMELS data files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=5678,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--bias",
        type=float,
        nargs="+",
        default=None,
        metavar="B",
        help="One or more precipitation bias multipliers to sweep over.",
    )
    parser.add_argument(
        "--stddev",
        type=float,
        nargs="+",
        default=None,
        metavar="S",
        help="One or more random error standard deviations to sweep over.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory where calibration results will be saved.",
    )

    return parser.parse_args()


def build(args, h5path, model_path, config_path):
    return {
        "data": {
            "forcing": args.forcing,
            "data_dir": args.data_dir,
            "h5path": str(h5path),
            "static_vars": STATIC_VARS,
        },
        "model": {
            "hidden_size": args.hidden_size,
            "dropout_rate": args.dropout_rate,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "clip_gradient_norm": args.clip_gradient_norm,
            "seed": args.seed,
            "preload": args.preload,
        },
        "output": {
            "model_path": str(model_path),
            "config_path": str(config_path),
        },
    }

def get_git_commit():
    """Capture the current git state for reproducibility logging."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        short = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
        return {"commit": commit, "short_commit": short, "dirty": len(dirty) > 0}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": "unknown", "short_commit": "unknown", "dirty": False}

def save(config, config_path):
    with open(config_path, "wb") as f:
        tomli_w.dump(config, f)


def load(config_path):
    """Load a saved training config. Useful for inference or reproducing a run."""
    with open(config_path, "rb") as f:
        return tomllib.load(f)
