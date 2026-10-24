#!/usr/bin/env python3
"""Export a YOLOv9 PyTorch model to ONNX for production inference."""

import argparse
from pathlib import Path

import torch


def export(weights: Path, output: Path, img_size: int = 640, device: str = "cpu"):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "yolov9"))

    from models.common import DetectMultiBackend

    model = DetectMultiBackend(str(weights), device=device, dnn=False, data=None, fp16=False)
    model.eval()

    dummy = torch.randn(1, 3, img_size, img_size).to(device)
    torch.onnx.export(
        model,
        dummy,
        str(output),
        opset_version=12,
        input_names=["images"],
        output_names=["output"],
        dynamic_axes={"images": {0: "batch", 2: "height", 3: "width"}},
    )
    print(f"Exported ONNX model to {output}")


def main():
    parser = argparse.ArgumentParser(description="Export YOLOv9 to ONNX")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    export(args.weights, args.output, args.img_size, args.device)


if __name__ == "__main__":
    main()
