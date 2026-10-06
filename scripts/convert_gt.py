#!/usr/bin/env python3
"""One-time converter: kitchen_small_1.mat -> gt.csv for the C++ pipeline.
Usage: python3 convert_gt.py kitchen_small_1.mat gt.csv
Output rows: frame,category,x1,y1,x2,y2 (0-based, inclusive), all categories
(the C++ side filters to the five project classes)."""
import sys
import numpy as np
from scipy.io import loadmat

mat, out = sys.argv[1], sys.argv[2]
bb = np.squeeze(loadmat(mat)["bboxes"])
with open(out, "w") as f:
    f.write("frame,category,x1,y1,x2,y2\n")
    for i in range(len(bb)):
        fr = np.squeeze(bb[i])
        if fr.size == 0:
            continue
        for o in np.atleast_1d(fr):
            cat = str(np.squeeze(o["category"]))
            top = int(np.squeeze(o["top"])); bot = int(np.squeeze(o["bottom"]))
            left = int(np.squeeze(o["left"])); right = int(np.squeeze(o["right"]))
            f.write(f"{i+1},{cat},{left-1},{top-1},{right-1},{bot-1}\n")
print("wrote", out)
