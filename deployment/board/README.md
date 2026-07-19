# LERDet RK3566 paired-image package

This package runs the final LERDet RGB-T model on a Taishan Pi RK3566. It accepts one visible/infrared image pair, constructs channels in `B,G,R,IR` order, and prints detections and timing. It does not draw boxes or encode video.

## Board setup

Copy the assembled package to `/userdata`, then run from its directory:

```bash
cd /userdata/<stagdet-package>
chmod +x *.sh
./verify_package.sh
./build_on_board.sh
export LERDET_DATA_ROOT=/userdata/<prepared-test-data>
export LERDET_MODEL=/userdata/models/lerdet_320x192_raw_dfl.rknn
./run_image.sh
./run_benchmark.sh
./run_perf_probe.sh
```

`LERDET_DATA_ROOT` must contain `manifest.csv`, `visible/`, and `infrared/`. It may be omitted when the package itself contains `test_data/`. The run scripts use the first pair by default; pass a basename without `.jpg` as argument 1 to select another pair.

```bash
./run_image.sh 20190925_111757_1_10_0000
./run_benchmark.sh 20190925_111757_1_10_0000 10 100
```

Prepared image dimensions must match the fixed RKNN input dimensions. The infrared source must first be stretched to the visible source geometry before both modalities receive the same resize and padding.
