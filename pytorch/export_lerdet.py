"""Export a trained LERDet checkpoint for downstream RKNN conversion."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
from ultralytics import YOLO


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("weights", type=Path)
    parser.add_argument("--imgsz", nargs=2, type=int, default=(192, 320))
    parser.add_argument("--format", default="onnx")
    args = parser.parse_args()
    YOLO(str(args.weights)).export(format=args.format, imgsz=tuple(args.imgsz), simplify=False)


if __name__ == "__main__":
    main()
