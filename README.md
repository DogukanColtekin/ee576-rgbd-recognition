# RGB-D Object Recognition and Tracking

Improved version of a machine vision term project: recognition and tracking of
five object classes (bowl, cap, cereal box, coffee mug, soda can) on the Washington RGB-D
Object and Scenes datasets, in C++17 with OpenCV.

- **(a)** segmentation of the object images and a 719-D RGB-D descriptor
- **(b)** a linear SVM and an MLP per class, precision and recall at τ = 0.3, 0.5, 0.7
- **(c)** FastSAM proposals on the `kitchen_small_1` sequence and their recognition
- **(d)** objects that continue, appear and disappear between consecutive frames

## Results

| | |
|---|---|
| Segmentation IoU (own segmentation vs. provided masks) | 0.85 |
| Object test mAP, SVM / MLP | 0.95 / 0.94 |
| Proposal recall (IoU ≥ 0.5) | 0.93 |
| Scene mAP, SVM / MLP | 0.46 / 0.45 |
| Tracking IDs (six objects) | 28 |

## Build and run

Requires CMake ≥ 3.16 and OpenCV ≥ 4.10 (the `dnn` module of older versions cannot run
the FastSAM graph). `FastSAM-s.onnx` can be downloaded from the
[v1.0 release](https://github.com/DogukanColtekin/ee576-rgbd-recognition/releases/tag/v1.0),
or re-created with `scripts/export_fastsam.py` (`pip install ultralytics`).

```bash
mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j && cd ..
bash scripts/run_all.sh <object data> <kitchen_small_1 dir> gt.csv FastSAM-s.onnx
```

or step by step:

```bash
./build/train_eval --data <object data> --out out_train                        # (a), (b)
./build/scene_eval --scene <kitchen_small_1 dir> --gt gt.csv \
                   --model out_train --onnx FastSAM-s.onnx --out out_scene      # (c)
./build/tracking   --dets out_scene/detections.csv --gt gt.csv --out out_track  # (d)
```

Data: full-frame (640×480) Washington RGB-D Object dataset; instances 1–3 of each class
for training, the rest for testing; every 8th frame. `gt.csv` is the scene annotation
converted from `kitchen_small_1.mat` (`scripts/convert_gt.py`).

## Repository

```
src/        common library and the programs train_eval, scene_eval, tracking, visualize
scripts/    run_all.sh, convert_gt.py, export_fastsam.py
```

## References

- K. Lai, L. Bo, X. Ren, D. Fox, "A Large-Scale Hierarchical Multi-View RGB-D Object Dataset," ICRA 2011.
- X. Zhao et al., "Fast Segment Anything," arXiv:2306.12156, 2023.
- A. E. Johnson, M. Hebert, "Using Spin Images for Efficient Object Recognition in Cluttered 3D Scenes," IEEE TPAMI, 1999.
- N. Dalal, B. Triggs, "Histograms of Oriented Gradients for Human Detection," CVPR 2005.
- J. Platt, "Probabilistic Outputs for Support Vector Machines," 1999.
