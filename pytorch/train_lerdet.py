"""Train LERDet with the final RGB-T model configuration.

The dataset YAML must point to the user's local Anti-UAV6K RGB-IR dataset.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
from ultralytics import YOLO


MODEL = ROOT / "ultralytics/cfg/models/11-RGBT/yolo11-RGBT-p2p5-refine-target-c2psa.yaml"


def main() -> None:
    model = YOLO(str(MODEL))
    model.train(
        data="/path/to/anti_uav6k-rgbt.yaml",
        imgsz=320,
        epochs=100,
        batch=16,
        optimizer="SGD",
        lr0=0.005,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        close_mosaic=10,
        workers=0,
        seed=0,
        deterministic=True,
        use_wiseiou=True,
    )


if __name__ == "__main__":
    main()
