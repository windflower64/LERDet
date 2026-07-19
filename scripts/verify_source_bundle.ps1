$ErrorActionPreference = "Stop"
$bundle = Split-Path -Parent $PSScriptRoot
$required = @(
    "pytorch/ultralytics/nn/modules/rgbt_guidance.py",
    "pytorch/ultralytics/utils/loss.py",
    "pytorch/ultralytics/cfg/models/11-RGBT/yolo11-RGBT-p2p5-refine-target-c2psa.yaml",
    "deployment/board/src/stagdet_detector.cpp",
    "deployment/board/include/stagdet_detector.hpp",
    "deployment/video/src/video_pipeline_main.cpp"
)
foreach ($relative in $required) {
    $path = Join-Path $bundle $relative
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing required file: $relative" }
}
$forbidden = Get-ChildItem -LiteralPath $bundle -Recurse -File | Where-Object {
    $_.FullName -match "paper_figure|paper_table|figure_|debug\.json|\.docx$|\.pdf$|complete_metrics|results\.csv"
}
if ($forbidden) { throw "Forbidden paper/analysis artifact found: $($forbidden[0].FullName)" }
Write-Output "LERDet source bundle verification passed."
