#!/usr/bin/env python3
"""One-time export: FastSAM-s.pt -> FastSAM-s.onnx for cv::dnn.
Usage: python3 export_fastsam.py   (requires: pip install ultralytics)"""
from ultralytics import FastSAM
m = FastSAM("FastSAM-s.pt")          # auto-downloads weights
m.export(format="onnx", opset=12, imgsz=640)
print("exported FastSAM-s.onnx")
