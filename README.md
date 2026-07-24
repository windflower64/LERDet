# LERDet source bundle

This directory contains only the source needed to reproduce the final LERDet model and its RK3566 edge implementation. 
## Contents

- `pytorch/`: the retained Ultralytics training/validation core, the final RGB-T model configuration, DMR/TAG modules, WIoU loss support, and small command-line wrappers.
- `deployment/board/`: single-pair image inference, device benchmark, and RKNN performance-probe programs using the final raw-DFL detector.
- `deployment/video/`: original paired RGB-IR video end-to-end processing using the same detector implementation and OpenCV software decoding.
- `scripts/`: reproducibility checks and file-manifest utilities.

The source bundle does not include trained `.pt` or `.rknn` weights.  The board runtime header is included for compilation; the runtime library `librknnrt.so` must be installed on the RK3566 device.

## PyTorch path

```powershell
cd pytorch
python -m pip install -r requirements.txt
set YOLO_CONFIG_DIR=%CD%/.ultralytics
python train_lerdet.py
python validate_lerdet.py path/to/best.pt --data /path/to/anti_uav6k-rgbt.yaml
python export_lerdet.py path/to/best.pt --imgsz 192 320
```

Replace dataset and checkpoint paths with local paths. The training wrapper records the final paper settings: 320-pixel training input, SGD, 100 epochs, batch size 16, seed 0, deterministic execution, and the project dynamic IoU-quality modulation option.

## RK3566 path

On the board, install OpenCV 4.x development files, GCC/G++ with C++17 support, RKNN Runtime 2.3.2-compatible headers/library, and copy the final `.rknn` model to the path expected by the run script.

```bash
cd deployment/board
chmod +x *.sh
./verify_package.sh
./build_on_board.sh
./run_image.sh
./run_benchmark.sh
./run_perf_probe.sh
```

For original paired videos:

```bash
cd deployment/video
chmod +x *.sh
./build_on_board.sh
LERDET_VIDEO_DATA_ROOT=/userdata/datasets/stagdet_raw_videos \
LERDET_MODEL=/userdata/models/lerdet_320x192_raw_dfl.rknn \
./run_video_pair.sh
```

The video program uses the final `320x192` raw-DFL RKNN contract and performs CPU-side sparse score filtering, DFL decoding, and NMS. It does not draw boxes or encode result videos.

## Reproducibility boundary

The model name in the source configuration remains the historical `yolo11-RGBT-*` filename used by the training code; the paper-level method name is LERDet. The deployment source accepts one or two inputs and two or eight outputs so that the final merged raw-DFL model and the compatible per-scale diagnostic model can be inspected, while the documented deployment path uses the merged two-output model.
