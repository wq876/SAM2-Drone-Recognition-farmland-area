# -*- coding: utf-8 -*-
"""
Step3: 将多块手动标注的农田 mask 合并为一张联合掩码
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np

MASK_DIR = Path(r"E:/SAM_data/farm_masks")
OUT_MASK_PATH = Path(r"E:/SAM_data/farm_masks/farm_mask_union.png")
OUT_LOCS_PATH = Path(r"E:/SAM_data/locs_union.json")
FIRST_FRAME_PATH = Path(r"E:/SAM_data/frames/frame_0001.jpg")


def main():
    # 1. 读取第一帧尺寸，确保所有 mask 尺寸一致
    if not FIRST_FRAME_PATH.exists():
        print("ERROR: first frame not found")
        return
    img = cv2.imread(str(FIRST_FRAME_PATH))
    if img is None:
        print("ERROR: cannot read first frame")
        return
    H, W = img.shape[:2]

    # 2. 查找所有 mask 文件
    mask_files = sorted(MASK_DIR.glob("farm_mask_*.png"))
    if not mask_files:
        print("ERROR: no farm_mask_*.png found")
        return

    print(f"Found {len(mask_files)} mask files, merging...")

    # 3. 读取并取并集
    union = np.zeros((H, W), dtype=bool)
    for mf in mask_files:
        mask = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"WARNING: skip {mf.name}")
            continue
        if mask.shape[:2] != (H, W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        union = np.logical_or(union, mask > 127)

    if not union.any():
        print("ERROR: union mask is empty")
        return

    # 4. 保存联合掩码
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    union_img = (union.astype(np.uint8)) * 255
    if not cv2.imwrite(str(OUT_MASK_PATH), union_img):
        print("ERROR: cannot write union_mask.png")
        return
    print(f"Union mask saved: {OUT_MASK_PATH}")

    # 5. 计算整体质心，生成 locs_union.json
    ys, xs = np.where(union)
    center_x = float(xs.mean())
    center_y = float(ys.mean())
    locs_data = {
        "farmlands": [
            {
                "id": 1,
                "center": [round(center_x, 2), round(center_y, 2)]
            }
        ]
    }
    with open(OUT_LOCS_PATH, "w", encoding="utf-8") as f:
        json.dump(locs_data, f, ensure_ascii=False, indent=2)
    print(f"locs_union.json saved: {OUT_LOCS_PATH}")
    print(f"Center: ({center_x:.1f}, {center_y:.1f})")


if __name__ == "__main__":
    main()