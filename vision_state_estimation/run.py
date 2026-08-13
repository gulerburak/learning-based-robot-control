"""Train the two networks and compare the two angle representations.

Example:
    python -m vision_state_estimation.run --model both --epochs 50 --runs 3
"""

import argparse
from pathlib import Path

from vision_state_estimation.dataset import load_dataloaders
from vision_state_estimation.train import (
    plot_error_against_angle,
    plot_loss_curves,
    run_experiment,
)

DEFAULT_DATA_DIR = Path("data") / "pendulum_images"
OUTPUT_DIR = Path("outputs") / "vision"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["theta", "trig", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = load_dataloaders(
        args.data_dir, batch_size=args.batch_size, seed=0
    )
    print(f"Dataset: {len(train_loader.dataset)} train, {len(val_loader.dataset)} "
          f"validation, {len(test_loader.dataset)} test samples.")

    model_types = ["theta", "trig"] if args.model == "both" else [args.model]
    histories, models = {}, {}

    for model_type in model_types:
        _, history, model = run_experiment(
            model_type,
            train_loader,
            val_loader,
            test_loader,
            num_epochs=args.epochs,
            num_runs=args.runs,
            checkpoint_dir=args.output_dir / "checkpoints",
        )
        histories[model_type] = history
        models[model_type] = model

    plot_loss_curves(histories, args.output_dir / "loss_curves.pdf")
    plot_error_against_angle(models, test_loader, args.output_dir / "error_vs_angle.pdf")
    print(f"Figures are in {args.output_dir}.")


if __name__ == "__main__":
    main()
