"""
Step3&4 联合掩码版（修复版）
核心修复：
1. 正确使用 add_new_mask 注入联合掩码（处理 API 签名差异）
2. 联合掩码传播后，用连通域分析拆出每块独立农田
3. 时序平滑：同一连通域跨帧用 IoU 匹配，保持 ID 稳定
"""

import glob
import json
import os
import re
import shutil
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TORCH_HOME"] = r"E:\SAM_data\weights"

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ================= 配置 =================
FRAMES_GLOB     = r"E:\SAM_data\frames\frame_*.jpg"
FIRST_FRAME     = Path(r"E:\SAM_data\frames\frame_0001.jpg")
LOCS_PATH       = Path(r"E:\SAM_data\locs_union.json")
MASK_DIR        = Path(r"E:\SAM_data\farm_masks\farm_masks_union")
WEIGHT_PATH     = Path(r"E:\SAM_data\weights\sam2.1_hiera_small.pt")
MODEL_CFG       = "configs/sam2.1/sam2.1_hiera_s.yaml"
OUT_OVERLAY_DIR = Path(r"E:\SAM_data\masks_frame")
AREA_JSON       = Path(r"E:\SAM_data\area_list.json")

OVERLAY_ALPHA       = 0.45
MIN_COMPONENT_AREA  = 500    # 过滤小于500像素的碎斑（可调大）
IOU_MATCH_THRESH    = 0.3    # 跨帧连通域匹配阈值


# ================= 工具函数 =================

def natural_sort_frames(paths: list) -> list:
    def key(p):
        m = re.search(r"(\d+)", Path(p).stem)
        return int(m.group(1)) if m else 10**9
    return sorted(paths, key=key)


def collect_frame_paths() -> list:
    paths = glob.glob(FRAMES_GLOB)
    if not paths:
        paths = glob.glob(r"E:\SAM_data\frames\frame_*.jpeg")
    return natural_sort_frames(paths)


def load_farm_ids(locs_path: Path) -> list:
    if not locs_path.is_file():
        return []
    data = json.loads(locs_path.read_text(encoding="utf-8"))
    return [int(r["id"]) for r in data.get("farmlands", [])]


def read_bool_mask(path: Path, target_hw: tuple) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise ValueError(f"无法读取 mask: {path}")
    m = im > 127
    th, tw = target_hw
    if m.shape != (th, tw):
        m = cv2.resize(m.astype(np.uint8), (tw, th),
                       interpolation=cv2.INTER_NEAREST) > 0
    return m


def create_union_mask(mask_dir: Path, farm_ids: list, target_hw: tuple) -> np.ndarray:
    """把所有手动标注的 mask 取并集，作为第一帧的"农田"先验"""
    H, W = target_hw
    union = np.zeros((H, W), dtype=bool)
    missing = []
    for fid in farm_ids:
        p = mask_dir / f"farm_mask_{fid:02d}.png"
        if p.is_file():
            union = np.logical_or(union, read_bool_mask(p, (H, W)))
        else:
            missing.append(fid)
    if missing:
        print(f"  ⚠ 以下 mask 文件不存在，已跳过: {missing}")
    print(f"  ✅ 联合掩码像素数: {union.sum()}")
    return union


# ================= 修复：注入联合掩码 =================

def inject_union_mask(predictor, inference_state, union_mask: np.ndarray):
    """
    尝试多种方式把联合掩码注入 SAM2，兼容不同版本 API。
    优先用 add_new_mask（直接注入 mask），
    失败则退回到框+多点的方式。
    """
    mask_tensor = torch.from_numpy(union_mask.astype(np.uint8))   # bool→uint8

    # ---- 方式一：add_new_mask（SAM2 官方推荐，部分版本 obj_id 是关键字参数）----
    for kwargs in [
        dict(frame_idx=0, obj_id=1, mask=mask_tensor),
        dict(frame_idx=0, obj_id=1, mask=mask_tensor.bool()),
    ]:
        try:
            predictor.add_new_mask(inference_state, **kwargs)
            print("  ✅ add_new_mask 注入成功")
            return
        except Exception as e:
            last_err = e

    # ---- 方式二：add_new_points_or_box，用联合掩码的 bbox + 质心点 ----
    print(f"  ⚠ add_new_mask 失败({last_err})，回退到 bbox+point 模式")
    ys, xs = np.where(union_mask)
    if len(xs) == 0:
        sys.exit("❌ 联合掩码为空，无法继续")

    # 多点采样：在联合掩码内均匀采 20 个点，覆盖所有农田
    flat_idx = np.where(union_mask.ravel())[0]
    sample_idx = flat_idx[np.linspace(0, len(flat_idx)-1, min(20, len(flat_idx)), dtype=int)]
    H, W = union_mask.shape
    sample_pts = np.stack([sample_idx % W, sample_idx // W], axis=1).astype(np.float32)
    sample_labels = np.ones(len(sample_pts), dtype=np.int32)

    bbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)

    predictor.add_new_points_or_box(
        inference_state,
        frame_idx=0,
        obj_id=1,
        points=sample_pts,
        labels=sample_labels,
        box=bbox,
        clear_old_points=True,
        normalize_coords=False,   # 坐标已是像素坐标
    )
    print(f"  ✅ bbox+{len(sample_pts)}点 模式注入成功")


# ================= 连通域拆分 =================

def split_to_components(farm_mask: np.ndarray, min_area: int):
    """
    对二值农田掩码做连通域分析，返回各块独立 mask 列表。
    先做轻量形态学：腐蚀切断细脖子，再膨胀还原主体。
    """
    u8 = farm_mask.astype(np.uint8) * 255

    # # 腐蚀：切断细连接（3x3，迭代2次）
    # k_erode = np.ones((5, 5), np.uint8)
    # eroded = cv2.erode(u8, k_erode, iterations=2)

    # # 膨胀：还原主体（3x3，迭代2次）
    # k_dilate = np.ones((3, 3), np.uint8)
    # cleaned = cv2.dilate(eroded, k_dilate, iterations=2)

    # 1. 激进腐蚀：只负责切断细田埂 (保持你现有的 5x5)
    k_erode = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(u8, k_erode, iterations=2)

    # 2. 提取外部轮廓 (忽略内部的小黑洞)
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 3. 创建全黑画布，把轮廓画成实心 (彻底消灭内部空隙)
    filled_mask = np.zeros_like(eroded)
    cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)

    # 4. 轻微膨胀：把被 5x5 腐蚀削掉的农田外围边缘稍微补回来一点点
    k_dilate = np.ones((3, 3), np.uint8)
    cleaned = cv2.dilate(filled_mask, k_dilate, iterations=2)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8
    )

    components = []
    for lid in range(1, num_labels):   # 0 是背景
        area = int(stats[lid, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp_mask = (labels == lid)
        cx = float(centroids[lid, 0])
        cy = float(centroids[lid, 1])
        components.append({"label": lid, "mask": comp_mask,
                            "area": area, "cx": cx, "cy": cy})
    return components

def merge_nearby_fragments(components, max_dist=20):
    """将空间上很接近的碎片合并，避免同一农田被切成多块"""
    if len(components) <= 1:
        return components

    masks = [c["mask"] for c in components]
    n = len(masks)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    kernel = np.ones((max_dist, max_dist), np.uint8)
    for i in range(n):
        dilated_i = cv2.dilate(masks[i].astype(np.uint8), kernel, iterations=1)
        for j in range(i + 1, n):
            dilated_j = cv2.dilate(masks[j].astype(np.uint8), kernel, iterations=1)
            if np.logical_and(dilated_i, dilated_j).any():
                union(i, j)

    groups = {}
    for idx in range(n):
        root = find(idx)
        groups.setdefault(root, []).append(idx)

    merged = []
    for indices in groups.values():
        if len(indices) == 1:
            merged.append(components[indices[0]])
        else:
            combined_mask = np.logical_or.reduce([masks[i] for i in indices])
            area = int(combined_mask.sum())
            ys, xs = np.where(combined_mask)
            cx = float(xs.mean()) if len(xs) > 0 else 0.0
            cy = float(ys.mean()) if len(ys) > 0 else 0.0
            new_label = min(components[i]["label"] for i in indices)
            merged.append({"label": new_label, "mask": combined_mask,
                           "area": area, "cx": cx, "cy": cy})
    return merged

# ================= 跨帧 ID 匹配（IoU 贪心） =================

def iou(m1: np.ndarray, m2: np.ndarray) -> float:
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return float(inter) / float(union) if union > 0 else 0.0


class FrameTracker:
    """用 IoU 把当前帧的连通域与上一帧匹配，维持稳定 ID"""
    def __init__(self):
        self.prev_comps = []   # list of {"id", "mask"}
        self.next_id = 1

    def match(self, cur_comps: list) -> list:
        """返回 list of {"id", "mask", "area", "cx", "cy"}"""
        if not self.prev_comps:
            # 第一帧直接分配 ID
            result = []
            for c in cur_comps:
                c["id"] = self.next_id
                self.next_id += 1
                result.append(c)
            self.prev_comps = [{"id": r["id"], "mask": r["mask"]} for r in result]
            return result

        used_prev = set()
        matched = {}

        # 贪心匹配：IoU 最大的先匹配
        scores = []
        for ci, c in enumerate(cur_comps):
            for pi, p in enumerate(self.prev_comps):
                s = iou(c["mask"], p["mask"])
                if s >= IOU_MATCH_THRESH:
                    scores.append((s, ci, pi))
        scores.sort(reverse=True)

        for s, ci, pi in scores:
            if ci in matched or pi in used_prev:
                continue
            matched[ci] = self.prev_comps[pi]["id"]
            used_prev.add(pi)

        result = []
        for ci, c in enumerate(cur_comps):
            c["id"] = matched.get(ci, self.next_id)
            if ci not in matched:
                self.next_id += 1
            result.append(c)

        self.prev_comps = [{"id": r["id"], "mask": r["mask"]} for r in result]
        return result


# ================= 叠加图绘制 =================

def save_overlay(bgr: np.ndarray, components: list, out_path: Path):
    """
    绘制叠加图：半透明彩色mask + 醒目标注
    标注样式：半透明黑色圆角底 + 亮黄色文字，彻底解决黑色方块问题
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    cmap = plt.get_cmap("tab10")

    # === 第一步：绘制半透明彩色 mask ===
    for comp in components:
        fid = comp["id"]
        mb = comp["mask"]
        color_f = np.array(cmap((fid - 1) % 10)[:3], dtype=np.float32)
        for c in range(3):
            rgb[..., c][mb] = (rgb[..., c][mb] * (1 - OVERLAY_ALPHA)
                               + color_f[c] * OVERLAY_ALPHA)

    # 转为 uint8
    out_u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(out_u8, cv2.COLOR_RGB2BGR)

    # === 第二步：绘制标注文字 ===
    for comp in components:
        fid = comp["id"]
        area_px = comp["area"]
        cx, cy = int(comp["cx"]), int(comp["cy"])

        # 文字内容
        text = f"ID:{fid}  {area_px}px"

        # 根据图片大小自适应字体大小
        font_scale = max(out_bgr.shape[1] / 1800, 0.55)
        thickness = max(int(font_scale * 2.5), 1)

        # 获取文字尺寸
        (tw, th), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )

        # 背景矩形坐标（比文字略大一圈）
        pad_x, pad_y = 6, 6
        x1 = cx - tw // 2 - pad_x
        y1 = cy - th // 2 - pad_y
        x2 = cx + tw // 2 + pad_x
        y2 = cy + th // 2 + pad_y + baseline

        # 边界保护，防止超出画面
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(out_bgr.shape[1] - 1, x2)
        y2 = min(out_bgr.shape[0] - 1, y2)

        # === 关键修改：半透明黑色背景（而不是纯黑） ===
        # 先提取背景区域
        roi = out_bgr[y1:y2, x1:x2].copy()
        if roi.size == 0:
            continue

        # 创建半透明黑色遮罩 (alpha=0.5)
        overlay_bg = np.zeros_like(roi)
        alpha = 0.5
        blended = cv2.addWeighted(roi, 1 - alpha, overlay_bg, alpha, 0)

        # 把半透明背景贴回原图
        out_bgr[y1:y2, x1:x2] = blended

        # 绘制文字（亮黄色，在深浅背景下都清晰）
        text_x = x1 + pad_x
        text_y = y1 + th + pad_y
        cv2.putText(
            out_bgr, text, (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            (0, 255, 255),  # 亮黄色 (BGR: 0, 255, 255)
            thickness
        )
    # 保存
    cv2.imwrite(str(out_path), out_bgr)


# ================= 主流程 =================

def main():
    # 前置检查
    for p, name in [(WEIGHT_PATH, "模型权重"), (FIRST_FRAME, "第一帧图像"),
                    (LOCS_PATH, "locs.json"), (MASK_DIR, "farm_masks 目录")]:
        if not p.exists():
            sys.exit(f"❌ 找不到 {name}: {p}")

    frame_paths = collect_frame_paths()
    if not frame_paths:
        sys.exit("❌ 未找到帧图像")

    farm_ids = load_farm_ids(LOCS_PATH)
    if not farm_ids:
        sys.exit("❌ locs.json 为空")

    img0 = cv2.imread(str(FIRST_FRAME))
    H0, W0 = img0.shape[:2]

    print(f"📋 共 {len(frame_paths)} 帧，{len(farm_ids)} 块手动标注农田")
    print("🔀 合并联合掩码中...")
    union_mask = create_union_mask(MASK_DIR, farm_ids, (H0, W0))
    if not union_mask.any():
        sys.exit("❌ 联合掩码为空")

    # 导入 SAM2
    try:
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError as e:
        sys.exit(f"❌ 无法导入 sam2: {e}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥  使用设备: {device}")
    predictor = build_sam2_video_predictor(MODEL_CFG, str(WEIGHT_PATH), device=device)

    tracker = FrameTracker()
    area_records = []
    OUT_OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="sam2_union_") as tmpdir:
        # SAM2 要求纯数字文件名
        for i, src in enumerate(frame_paths):
            shutil.copy2(src, Path(tmpdir) / f"{i:06d}.jpg")

        ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
               if device == "cuda" else nullcontext())

        with torch.inference_mode(), ctx:
            inference_state = predictor.init_state(
                video_path=tmpdir,
                offload_video_to_cpu=(device == "cpu"),
                offload_state_to_cpu=(device == "cpu"),
                async_loading_frames=False,
            )
            vh = int(inference_state["video_height"])
            vw = int(inference_state["video_width"])

            # 如果视频分辨率与图像不同，缩放联合掩码
            if (vh, vw) != (H0, W0):
                union_mask = cv2.resize(
                    union_mask.astype(np.uint8), (vw, vh),
                    interpolation=cv2.INTER_NEAREST) > 0
                print(f"  ↔ 联合掩码已缩放至 {vw}x{vh}")

            print("💉 注入联合掩码...")
            inject_union_mask(predictor, inference_state, union_mask)

            print("🎬 开始视频传播追踪...")
            for frame_idx, obj_ids, video_res_masks in predictor.propagate_in_video(
                inference_state, start_frame_idx=0
            ):
                bgr = cv2.imread(frame_paths[frame_idx])
                if bgr is None:
                    print(f"  ⚠ 帧 {frame_idx} 读取失败，跳过")
                    continue
                H, W = bgr.shape[:2]

                # 提取 obj_id=1 的 mask
                vm = video_res_masks.detach().float().cpu().numpy()
                if vm.ndim == 4:
                    vm = vm[:, 0]          # (N, H, W)
                obj_id_list = list(obj_ids)
                if 1 not in obj_id_list:
                    print(f"  ⚠ 帧 {frame_idx} 未找到 obj_id=1，跳过")
                    continue
                idx1 = obj_id_list.index(1)
                raw_mask = vm[idx1]
                if raw_mask.shape != (H, W):
                    raw_mask = cv2.resize(raw_mask, (W, H))
                farm_mask = raw_mask > 0.5

                # 连通域拆分
                components = split_to_components(farm_mask, MIN_COMPONENT_AREA)
                components = merge_nearby_fragments(components, max_dist=10)

                # 跨帧 ID 匹配
                components = tracker.match(components)

                # 保存叠加图
                out_path = OUT_OVERLAY_DIR / f"{Path(frame_paths[frame_idx]).stem}_overlay.jpg"
                save_overlay(bgr, components, out_path)

                # 面积记录
                farmlands_stats = [
                    {"id": c["id"], "area_pixels": c["area"],
                     "cx": round(c["cx"], 1), "cy": round(c["cy"], 1)}
                    for c in components
                ]
                area_records.append({
                    "frame_index": frame_idx,
                    "frame_name": Path(frame_paths[frame_idx]).name,
                    "farmlands": farmlands_stats,
                })

                n = len(components)
                total_px = sum(c["area"] for c in components)
                print(f"  Frame {frame_idx+1}/{len(frame_paths)}: "
                      f"检测到 {n} 块农田，总面积 {total_px} px")

    # 保存 area_list.json
    def to_native(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: to_native(v) for k, v in obj.items()}
        if isinstance(obj, list): return [to_native(i) for i in obj]
        return obj

    with open(AREA_JSON, "w", encoding="utf-8") as f:
        json.dump(to_native(area_records), f, ensure_ascii=False, indent=2)

    print(f"\n✅ 完成，共 {len(area_records)} 帧，结果 → {AREA_JSON}")


if __name__ == "__main__":
    main()