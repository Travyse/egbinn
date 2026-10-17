"""CLI SHAP analysis script: explain EGBINN predictions at hidden-layer level.

Usage:
    egbinn-shap --checkpoint resources/pretrained/egbinn/checkpoint.pth \\
        --config resources/pretrained/egbinn/config.yml \\
        --background_data /path/to/normal.memmap --test_data /path/to/tumor.memmap \\
        --mask_dir resources/masks/ --output_dir shap_results/
"""

import argparse
import os

from egbinn.data import load_betas_array
from egbinn.shap_analysis import run_shap_analysis
from egbinn.utils import attach_stream_logging, resolve_device, setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="EGBINN SHAP Feature Importance Analysis")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument(
        "--config",
        "--config_path",
        dest="config_path",
        type=str,
        required=True,
        help="Path to the YAML config used for training",
    )
    parser.add_argument(
        "--background_data", type=str, required=True, help="Path to background methylation data (.memmap or .csv)"
    )
    parser.add_argument("--test_data", type=str, required=True, help="Path to test methylation data (.memmap or .csv)")
    parser.add_argument("--background_imputation", type=str, default=None)
    parser.add_argument("--test_imputation", type=str, default=None)
    parser.add_argument(
        "--mask_dir", type=str, required=True, help="Base directory containing per-stage mask artifacts"
    )
    parser.add_argument("--edges_path", type=str, default=None, help="Path to edges.tsv for pathway display names")
    parser.add_argument("--output_dir", type=str, default="./shap_results")
    parser.add_argument(
        "--target_layers", type=str, nargs="+", default=None, help="Layers to explain (default: gene l2 l1 l0)"
    )
    parser.add_argument("--bg_cap", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    attach_stream_logging()

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(args.output_dir, "egbinn_shap", "shap.log")

    logger.info("Loading background data from %s", args.background_data)
    background_betas = load_betas_array(args.background_data, args.background_imputation)
    logger.info("Background: %s samples x %s features", *background_betas.shape)

    logger.info("Loading test data from %s", args.test_data)
    test_betas = load_betas_array(args.test_data, args.test_imputation)
    logger.info("Test: %s samples x %s features", *test_betas.shape)

    logger.info("Running SHAP analysis...")
    results = run_shap_analysis(
        checkpoint_path=args.checkpoint,
        config_path=args.config_path,
        background_betas=background_betas,
        test_betas=test_betas,
        mask_dir=args.mask_dir,
        output_dir=args.output_dir,
        target_layers=args.target_layers,
        edges_path=args.edges_path,
        bg_cap=args.bg_cap,
        batch_size=args.batch_size,
        device=device,
    )

    logger.info("SHAP analysis complete:")
    for layer, path in results.items():
        logger.info("  %s: %s", layer, path)


if __name__ == "__main__":
    main()
