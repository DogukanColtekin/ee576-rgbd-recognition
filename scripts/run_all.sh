#!/usr/bin/env bash
# Full pipeline, parts (a)-(d). Usage:
#   scripts/run_all.sh <rgbd-dataset dir> <kitchen_small_1 dir> [gt.csv] [FastSAM-s.onnx]
set -euo pipefail
DATA=${1:?object dataset root (contains bowl/ cap/ ...)}
SCENE=${2:?scene directory with kitchen_small_1_<k>.png and _depth.png}
GT=${3:-gt.csv}
ONNX=${4:-FastSAM-s.onnx}
B=./build

# (a)+(b) own segmentation feeds the descriptors (main result)
$B/train_eval --data "$DATA" --out out_train --seg plane --every 8 --aug 1 --bg 2
# (a)+(b) ablations: provided masks (upper bound) and depth-Otsu segmentation
$B/train_eval --data "$DATA" --out out_train_gt    --seg gt    --every 8 --aug 1 --bg 2
$B/train_eval --data "$DATA" --out out_train_depth --seg depth --every 8 --aug 1 --bg 2
# (c) FastSAM proposals -> recognition with the models of the main run
$B/scene_eval --scene "$SCENE" --gt "$GT" --model out_train --onnx "$ONNX" --out out_scene --every 2
# (d) tracking + timeline, for both classifiers
$B/tracking --dets out_scene/detections.csv --gt "$GT" --out out_track_svm --scorer svm --tau 0.5
$B/tracking --dets out_scene/detections.csv --gt "$GT" --out out_track_mlp --scorer mlp --tau 0.5
# figures for the report
$B/visualize --mode object --data "$DATA" --out viz_obj --every 16
$B/visualize --mode scene --scene "$SCENE" --dets out_scene/detections.csv --gt "$GT" --out viz_scene --tau 0.5
echo "done - see out_train*/, out_scene/, out_track_*/, viz_*/"
