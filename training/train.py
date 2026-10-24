"""Production training CLI for license plate detection.

Extracted from the exploratory notebook into a reusable, configurable CLI.
"""

import argparse
import sys
from pathlib import Path

from core.config import settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train YOLOv9 license plate detector")
    parser.add_argument(
        "--data",
        type=Path,
        default=settings.data_yaml,
        help="Path to dataset YAML",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=settings.model_weights,
        help="Initial weights (.pt)",
    )
    parser.add_argument(
        "--cfg",
        type=Path,
        default=settings.model_cfg,
        help="Model architecture YAML",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument(
        "--device", type=str, default="", help="CUDA device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("runs/train"),
        help="Save results to project/name",
    )
    parser.add_argument("--name", type=str, default="exp", help="Save results to project/name")
    parser.add_argument(
        "--hyp",
        type=Path,
        default=None,
        help="Hyperparameters YAML",
    )
    return parser


def train(args: argparse.Namespace) -> None:
    """Run training by delegating to the vendored YOLOv9 train.py."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "yolov9"))

    from train_dual import train as yolov9_train  # noqa: WPS433
    from utils.callbacks import Callbacks  # noqa: WPS433

    hyp = (
        args.hyp
        if args.hyp
        else Path(Path(__file__).resolve().parents[1] / "yolov9/data/hyps/hyp.scratch-high.yaml")
    )

    class Opt:
        pass

    opt = Opt()
    opt.weights = str(args.weights)
    opt.cfg = str(args.cfg)
    opt.data = str(args.data)
    opt.hyp = str(hyp)
    opt.epochs = args.epochs
    opt.batch_size = args.batch
    opt.imgsz = args.imgsz
    opt.workers = args.workers
    opt.device = args.device
    opt.project = str(args.project)
    opt.name = args.name
    opt.save_dir = args.project / args.name
    opt.single_cls = False
    opt.evolve = False
    opt.resume = False
    opt.noval = False
    opt.nosave = False
    opt.freeze = None
    opt.cache = False
    opt.image_weights = False
    opt.rect = False
    opt.cos_lr = False
    opt.label_smoothing = 0.0
    opt.patience = 100
    opt.multi_scale = False
    opt.close_mosaic = 10
    opt.save_period = -1
    opt.seed = 0
    opt.local_rank = -1

    callbacks = Callbacks()
    yolov9_train(hyp, opt, None, callbacks)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
