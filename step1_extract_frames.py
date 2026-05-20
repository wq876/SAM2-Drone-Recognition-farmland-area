"""
从 MP4 视频中按每秒 2 帧提取画面，保存到当前目录下的 frames/ 文件夹。
"""

import os
import cv2
from config import BASE_DIR
os.chdir(BASE_DIR)
from config import VIDEO_PATH, FRAMES_DIR, OUTPUT_FPS


def main() -> None:
    os.makedirs(FRAMES_DIR, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频: {VIDEO_PATH}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps is None or src_fps <= 1e-6:
        src_fps = 30.0

    # 源视频中每隔多少帧取 1 张，使输出约为每秒 OUTPUT_FPS 张
    step = max(1, int(round(float(src_fps) / OUTPUT_FPS)))

    save_index = 0
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % step == 0:
            save_index += 1
            out_path = os.path.join(FRAMES_DIR, f"frame_{save_index:04d}.jpg")
            cv2.imwrite(out_path, frame)
        frame_index += 1

    cap.release()
    print(f"已保存 {save_index} 帧到: {FRAMES_DIR}")


if __name__ == "__main__":
    main()
