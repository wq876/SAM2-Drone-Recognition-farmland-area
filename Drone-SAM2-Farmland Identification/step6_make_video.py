# -*- coding: utf-8 -*-
"""
Step 6 (优化版): 生成左右分屏对比视频，解决标注不清晰的问题。
"""
import cv2
import numpy as np
from pathlib import Path
import re

# ================= 配置 =================
FRAMES_DIR = Path(r"E:\SAM_data\frames")
OVERLAYS_DIR = Path(r"E:\SAM_data\masks_frame")
OUTPUT_VIDEO = Path(r"E:\SAM_data\demo_comparison.mp4")

FPS = 24.0
FRAME_REPEAT = 12
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX

def natural_sort_key(p: str):
    m = re.search(r"(\d+)", Path(p).stem)
    return int(m.group(1)) if m else 0

def put_highlight_text(img, text, pos, font_scale=1.0):
    """在指定位置绘制带黑色背景的高亮文字，确保任何背景下都清晰可见"""
    x, y = pos
    # 获取文字尺寸
    (tw, th), baseline = cv2.getTextSize(text, TEXT_FONT, font_scale, 2)
    # 绘制黑色背景矩形（比文字略大一圈）
    cv2.rectangle(img, 
                  (x - 5, y - th - 10), 
                  (x + tw + 5, y + 5), 
                  (0, 0, 0), -1)
    # 绘制亮绿色文字
    cv2.putText(img, text, (x, y), TEXT_FONT, font_scale, (0, 255, 0), 2)

def main():
    frames = sorted(FRAMES_DIR.glob("frame_*.jpg"), key=lambda x: natural_sort_key(str(x)))
    overlays = sorted(OVERLAYS_DIR.glob("frame_*_overlay.jpg"), key=lambda x: natural_sort_key(str(x)))
    
    if not frames or not overlays:
        print("错误: 找不到图片文件"); return

    # 读取第一张图获取尺寸
    img_l = cv2.imread(str(frames[0]))
    img_r = cv2.imread(str(overlays[0]))
    if img_l is None or img_r is None:
        print("错误: 图片读取失败"); return

    h, w_l = img_l.shape[:2]
    _, w_r = img_r.shape[:2]
    total_w = w_l + w_r
    
    # 初始化视频写入器 (使用 H.264 编码，专为网页优化)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, FPS, (total_w, h))

    if not out.isOpened():
        print("错误: 无法创建视频文件"); return

    print(f"开始合成优化版视频，共 {len(frames)} 帧...")
    
    for i, (f_path, o_path) in enumerate(zip(frames, overlays)):
        img_l = cv2.imread(str(f_path))
        img_r = cv2.imread(str(o_path))
        
        # 统一尺寸
        if img_l.shape[0] != h or img_r.shape[0] != h:
            img_l = cv2.resize(img_l, (w_l, h))
            img_r = cv2.resize(img_r, (w_r, h))

        # 左右拼接
        combined = np.hstack((img_l, img_r))
        
        # 🎨 使用高亮文字标签，彻底解决看不清的问题
        put_highlight_text(combined, "ORIGINAL FRAME", (20, 40), 1.0)
        put_highlight_text(combined, "SAM2 + VLM TRACKING", (w_l + 20, 40), 1.0)
        
        # 帧复用
        for _ in range(FRAME_REPEAT):
            out.write(combined)
            
        if (i + 1) % 5 == 0:
            print(f"  进度: {i+1}/{len(frames)}")

    out.release()
    print(f"\n✅ 优化版视频已生成: {OUTPUT_VIDEO}")
    print("💡 若仍有兼容问题，请用 VLC 播放或拖入剪映转码。")

if __name__ == "__main__":
    main()