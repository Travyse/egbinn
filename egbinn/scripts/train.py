"""CLI training script: train an EGBINN/BINN/MLP aging clock.

Usage:
    egbinn-train --config configs/egbinn.yml
    python -m egbinn.scripts.train --config configs/egbinn.yml
    python scripts/train.py --config configs/egbinn.yml
"""

import argparse

from egbinn.adapters import get_adapter
from egbinn.config import load_config, validate_config
from egbinn.data import infer_input_dim, load_data
from egbinn.training import build_model, train_model
from egbinn.utils import attach_stream_logging, resolve_device, seed_everything, setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="EGBINN Aging Clock Training")
    parser.add_argument(
        "--config",
        "--config_path",
        dest="config_path",
        type=str,
        required=True,
        help="Path to the YAML config file",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to train on (default: auto-detect GPU/MPS/CPU)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config_path)
    validate_config(config)
    seed_everything(args.seed)

    device = resolve_device(args.device)
    attach_stream_logging()

    general = config.general
    save_dir = config.output.get("save_dir", "./outputs")
    logger = setup_logger(save_dir, "egbinn_train", "train.log")
    logger.info("Config: %s", args.config_path)
    logger.info("Model: %s | Device: %s | Seed: %s", config.model, device, args.seed)

    data_cfg = config.data
    label_column = general.get("label_column", "age")
    batch_size = int(general.get("batch_size", 64))
    num_workers = int(general.get("num_workers", 0))
    imputation_path = data_cfg.get("imputation_path", None)
    data_format = data_cfg.get("data_format", None)
    train_path = data_cfg["train_methylation_path"]
    val_path = data_cfg.get("val_methylation_path", None)

    input_dim = infer_input_dim(train_path, imputation_path, data_format)
    logger.info("Input dim: %s", input_dim)

    mask_dir = data_cfg.get("mask_dir", None)
    model = build_model(config, input_dim, mask_dir)
    adapter = get_adapter(config.model)

    train_metadata_path = data_cfg.get("train_metadata_path", data_cfg.get("metadata_path"))
    val_metadata_path = data_cfg.get("val_metadata_path", data_cfg.get("metadata_path"))

    logger.info("Loading training data from %s", train_path)
    train_loader, label_scaler = load_data(
        methylation_path=train_path,
        metadata_path=train_metadata_path,
        label_column=label_column,
        label_scaler=None,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        imputation_path=imputation_path,
        is_validation=False,
        data_format=data_format,
    )

    val_loader = None
    if val_path:
        logger.info("Loading validation data from %s", val_path)
        val_loader, _ = load_data(
            methylation_path=val_path,
            metadata_path=val_metadata_path,
            label_column=label_column,
            label_scaler=label_scaler,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            imputation_path=imputation_path,
            is_validation=True,
            data_format=data_format,
        )

    logger.info("Starting training...")
    summary = train_model(
        model=model,
        adapter=adapter,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        logger=logger,
        label_scaler=label_scaler,
        input_dim=input_dim,
    )

    logger.info(
        "Training complete: best_epoch=%s best_val_mae=%s checkpoint=%s",
        summary["best_epoch"],
        summary["best_val_mae"],
        summary["checkpoint_path"],
    )


if __name__ == "__main__":
    main()
