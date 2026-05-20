"""
OpenCV + SAM2 交互式农田标注：点击加点、预览分割、回车确认、退出写 locs.json 与总叠加图。
"""

import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TORCH_HOME"] = r"E:\SAM_data\weights"

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

IMAGE_PATH = Path(r"E:\SAM_data\frames\frame_0001.jpg")
WEIGHT_PATH = Path(r"E:\SAM_data\weights\sam2.1_hiera_small.pt")
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"
FARM_MASK_DIR = Path(r"E:\SAM_data\farm_masks")
LOCS_PATH = Path(r"E:\SAM_data\locs.json")
OVERLAY_PATH = Path(r"E:\SAM_data\mask_overlay.jpg")

WIN_NAME = "SAM2 Interactive Farmland Labeling"
PREVIEW_ALPHA = 0.45
CONFIRMED_ALPHA = 0.22
PT_RADIUS = 6
BGR_POS = (0, 255, 0)
BGR_NEG = (0, 0, 255)


def die(msg: str, code: int = 1) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def mouse_callback(event: int, x: int, y: int, flags: int, param) -> None:
    app: "InteractiveApp" = param
    if event == cv2.EVENT_LBUTTONDOWN:
        app.points.append((float(x), float(y), 1))
        app.preview_mask = None
    elif event == cv2.EVENT_RBUTTONDOWN:
        app.points.append((float(x), float(y), 0))
        app.preview_mask = None


class InteractiveApp:
    def __init__(self) -> None:
        self.points: list[tuple[float, float, int]] = []
        self.preview_mask: np.ndarray | None = None
        self.confirmed: list[dict] = []
        self.bgr: np.ndarray | None = None
        self.image_rgb: np.ndarray | None = None
        self.predictor = None
        self.device = "cpu"
        self.autocast_ctx = nullcontext()

    def farm_index_labeling(self) -> int:
        return len(self.confirmed) + 1

    def count_labels(self) -> tuple[int, int]:
        n_pos = sum(1 for _, _, lb in self.points if lb == 1)
        n_neg = sum(1 for _, _, lb in self.points if lb == 0)
        return n_pos, n_neg

    def update_title(self) -> None:
        fi = self.farm_index_labeling()
        np_, nn_ = self.count_labels()
        title = f"正在标注第 {fi} 块农田 | 正样本 {np_} 负样本 {nn_}"
        cv2.setWindowTitle(WIN_NAME, title)

    def run_predict(self) -> np.ndarray | None:
        if self.predictor is None or self.image_rgb is None:
            return None
        pos = [p for p in self.points if p[2] == 1]
        if not pos:
            print("提示: 至少需要 1 个正样本点（左键）才能分割。")
            return None

        coords = np.array([[p[0], p[1]] for p in self.points], dtype=np.float32)
        labels = np.array([p[2] for p in self.points], dtype=np.int32)

        try:
            with torch.inference_mode(), self.autocast_ctx:
                masks, scores, _ = self.predictor.predict(
                    point_coords=coords,
                    point_labels=labels,
                    multimask_output=False,
                )
        except Exception as e:
            print(f"分割失败: {e}")
            return None

        m = np.asarray(masks)
        if m.ndim == 4:
            m = m.squeeze(0)
        if m.ndim == 3:
            m = m[0]
        return m.astype(bool)

    def redraw(self) -> np.ndarray:
        assert self.bgr is not None
        vis = self.bgr.copy().astype(np.float32)

        cmap = plt.get_cmap("tab10")
        for i, rec in enumerate(self.confirmed):
            mk = rec["mask"]
            col = np.array(cmap(i % 10)[:3][::-1], dtype=np.float32)
            for c in range(3):
                ch = vis[..., c]
                ch[mk] = ch[mk] * (1.0 - CONFIRMED_ALPHA) + col[c] * CONFIRMED_ALPHA

        if self.preview_mask is not None:
            col = np.array([0.0, 0.9, 0.15], dtype=np.float32)
            for c in range(3):
                ch = vis[..., c]
                ch[self.preview_mask] = (
                    ch[self.preview_mask] * (1.0 - PREVIEW_ALPHA)
                    + col[c] * PREVIEW_ALPHA
                )

        out = np.clip(vis, 0, 255).astype(np.uint8)
        for x, y, lb in self.points:
            color = BGR_POS if lb == 1 else BGR_NEG
            cv2.circle(out, (int(round(x)), int(round(y))), PT_RADIUS, color, -1)
            cv2.circle(out, (int(round(x)), int(round(y))), PT_RADIUS, (255, 255, 255), 1)
        return out

    def delete_last_point(self) -> None:
        if self.points:
            self.points.pop()
            self.preview_mask = None

    def preview_segment(self) -> None:
        m = self.run_predict()
        if m is not None:
            self.preview_mask = m

    def confirm_segment(self) -> None:
        pos = [p for p in self.points if p[2] == 1]
        if not pos:
            print("提示: 确认前请至少保留 1 个正样本点。")
            return

        m = self.run_predict()
        if m is None:
            return

        fid = len(self.confirmed) + 1
        cx = float(np.mean([p[0] for p in pos]))
        cy = float(np.mean([p[1] for p in pos]))

        FARM_MASK_DIR.mkdir(parents=True, exist_ok=True)
        bin_png = (m.astype(np.uint8) * 255)
        out_path = FARM_MASK_DIR / f"farm_mask_{fid:02d}.png"
        if not cv2.imwrite(str(out_path), bin_png):
            print(f"错误: 无法写入 {out_path}")
            return

        self.confirmed.append(
            {
                "id": fid,
                "center": [round(cx, 2), round(cy, 2)],
                "mask": m,
            }
        )
        self.points.clear()
        self.preview_mask = None
        print(f"已确认第 {fid} 块农田，mask -> {out_path}，center=({cx:.2f}, {cy:.2f})")

    def save_overlay(self) -> None:
        assert self.bgr is not None
        rgb = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        cmap = plt.get_cmap("tab10")
        for i, rec in enumerate(self.confirmed):
            mk = rec["mask"]
            col = np.array(cmap(i % 10)[:3], dtype=np.float32)
            for c in range(3):
                ch = rgb[..., c]
                ch[mk] = ch[mk] * (1.0 - PREVIEW_ALPHA) + col[c] * PREVIEW_ALPHA
        out_u8 = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        h, w = out_u8.shape[:2]
        fig = plt.figure(figsize=(w / 150.0, h / 150.0), dpi=150)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(out_u8)
        ax.axis("off")
        fig.savefig(str(OVERLAY_PATH), dpi=150, pad_inches=0)
        plt.close(fig)
        print(f"已保存总叠加图: {OVERLAY_PATH}")

    def save_locs(self) -> None:
        obj = {
            "farmlands": [
                {"id": rec["id"], "center": rec["center"]}
                for rec in self.confirmed
            ]
        }
        LOCS_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOCS_PATH.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已保存 locs.json（共 {len(self.confirmed)} 块）: {LOCS_PATH}")


def main() -> int:
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as e:
        die(f"错误: 无法导入 sam2: {e}")

    if not IMAGE_PATH.is_file():
        die(f"错误: 找不到图片: {IMAGE_PATH}")
    if not WEIGHT_PATH.is_file():
        die(f"错误: 找不到权重: {WEIGHT_PATH}")

    bgr = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_COLOR)
    if bgr is None:
        die(f"错误: 无法读取图片: {IMAGE_PATH}")

    image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        sam = build_sam2(MODEL_CFG, str(WEIGHT_PATH), device=device)
    except Exception as e:
        die(f"错误: 加载 SAM2 失败: {e}")

    predictor = SAM2ImagePredictor(sam)
    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()
    )

    app = InteractiveApp()
    app.bgr = bgr
    app.image_rgb = image_rgb
    app.predictor = predictor
    app.device = device
    app.autocast_ctx = autocast_ctx

    with torch.inference_mode(), autocast_ctx:
        predictor.set_image(image_rgb)

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN_NAME, mouse_callback, app)

    print(
        "操作说明: 左键=正样本(绿) 右键=负样本(红) | "
        "d=撤销上一点 | s=预览分割 | 回车=确认并保存 | q=退出并写 locs.json"
    )

    while True:
        app.update_title()
        frame = app.redraw()
        cv2.imshow(WIN_NAME, frame)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            break
        if key == ord("d"):
            app.delete_last_point()
        elif key == ord("s"):
            app.preview_segment()
        elif key in (13, 10):
            app.confirm_segment()

    cv2.destroyAllWindows()

    app.save_locs()
    if app.confirmed:
        app.save_overlay()
    else:
        rgb = cv2.cvtColor(app.bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        fig = plt.figure(figsize=(w / 150.0, h / 150.0), dpi=150)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(rgb)
        ax.axis("off")
        fig.savefig(str(OVERLAY_PATH), dpi=150, pad_inches=0)
        plt.close(fig)
        print(f"未确认农田，已将原图保存为: {OVERLAY_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())