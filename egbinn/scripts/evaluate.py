"""CLI evaluation script: run age prediction with a trained model.

Usage:
    egbinn-evaluate --checkpoint resources/pretrained/egbinn/checkpoint.pth \\
        --data /path/to/methylation.memmap --metadata /path/to/metadata.csv
    python -m egbinn.scripts.evaluate --checkpoint ... --data ... --metadata ...
    python scripts/evaluate.py --checkpoint ... --data ... --metadata ...
"""

import argparse
import os

import pandas as pd

from egbinn.evaluation import predict
from egbinn.inference import compute_metrics, per_dataset_metrics
from egbinn.utils import attach_stream_logging, resolve_device, setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="EGBINN Aging Clock Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument(
        "--config",
        "--config_path",
        dest="config_path",
        type=str,
        default=None,
        help="YAML training config (only needed for legacy plain state-dict checkpoints)",
    )
    parser.add_argument(
        "--data",
        "--data_path",
        dest="data_path",
        type=str,
        required=True,
        help="Path to methylation data (.memmap or .csv)",
    )
    parser.add_argument(
        "--metadata", "--metadata_path", dest="metadata_path", type=str, required=True, help="Path to metadata CSV"
    )
    parser.add_argument(
        "--imputation_path", type=str, default=None, help="Path to .npy with per-feature mean beta values"
    )
    parser.add_argument(
        "--data_format",
        type=str,
        default=None,
        choices=["memmap", "csv"],
        help="Data format (auto-detected if not specified)",
    )
    parser.add_argument(
        "--label_scaler_path",
        type=str,
        default=None,
        help="JSON with the training label scaler (legacy checkpoints only)",
    )
    parser.add_argument("--label_column", type=str, default="age")
    parser.add_argument(
        "--dataset_col",
        type=str,
        default="Accession_Code",
        help="Metadata column identifying each sample's dataset (per-dataset metrics are emitted when present)",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument(
        "--output_file", type=str, default=None, help="Output predictions CSV path (overrides output_dir)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    attach_stream_logging()

    output_file = args.output_file or os.path.join(args.output_dir, "predictions.csv")
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    logger = setup_logger(os.path.dirname(output_file) or ".", "egbinn_evaluate", "evaluate.log")

    predictions_df = predict(
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        metadata_path=args.metadata_path,
        config_path=args.config_path,
        imputation_path=args.imputation_path,
        data_format=args.data_format,
        label_scaler_path=args.label_scaler_path,
        label_column=args.label_column,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        output_file=output_file,
    )

    metrics = compute_metrics(predictions_df["chron_age"], predictions_df["pred_age"])
    logger.info(
        "Metrics: MAE=%.4f, MSE=%.4f, MedAE=%.4f, Pearson r=%.4f",
        metrics.get("mae", float("nan")),
        metrics.get("mse", float("nan")),
        metrics.get("medae", float("nan")),
        metrics.get("pearson_r", float("nan")),
    )

    metadata = pd.read_csv(args.metadata_path, index_col=0)
    if args.dataset_col in metadata.columns:
        joined = predictions_df.merge(metadata[[args.dataset_col]], left_on="sample", right_index=True, how="left")
        per_dataset = per_dataset_metrics(joined, dataset_col=args.dataset_col)
        if not per_dataset.empty:
            per_dataset_path = str(output_file).replace(".csv", "_per_dataset.csv")
            per_dataset.to_csv(per_dataset_path, index=False)
            logger.info("Saved per-dataset metrics to %s", per_dataset_path)


if __name__ == "__main__":
    main()
