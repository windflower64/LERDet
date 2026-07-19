"""Validate a trained LERDet checkpoint with the Ultralytics detector evaluator."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
from ultralytics import YOLO


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("weights", type=Path)
    parser.add_argument("--data", required=True)
    parser.add_argument("--imgsz", type=int, default=320)
    args = parser.parse_args()
    YOLO(str(args.weights)).val(data=args.data, imgsz=args.imgsz, workers=0, split="test")


if __name__ == "__main__":
    main()
