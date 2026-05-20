"""
Step4 ：Mask并集 + 形态学搭桥断桥 + 实时物理面积换算
"""
import glob
import json
import math
import os
import re
import shutil
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import BASE_DIR
os.chdir(BASE_DIR)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TORCH_HOME"] = r"E:\SAM_data\weights"



try:
    import exifread
    HAS_EXIF = True
except ImportError:
    HAS_EXIF = False

# ================= 配置 =================
from config import (
    FRAMES_GLOB, FIRST_FRAME, UNION_LOCS_PATH, MASK_DIR,
    UNION_MASK_PATH, WEIGHT_PATH, MODEL_CFG,
    OVERLAY_DIR as OUT_OVERLAY_DIR, AREA_JSON,
    OVERLAY_ALPHA, MIN_COMPONENT_AREA_PX, MATCH_MAX_DIST,
    BRIDGE_CLOSE_KERNEL, CUT_OPEN_KERNEL,
    DEFAULT_GSD, MIN_AREA_SQM
)
LOCS_PATH = UNION_LOCS_PATH

# ================= 工具函数 =================
def get_dji_xmp_tags(image_path: Path) -> dict:
    if not HAS_EXIF or not image_path.is_file():
        return {}
    tags = {}
    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f, details=False)
    except Exception:
        return {}

    # 需要提取的三个物理量，按关键词模糊匹配
    result = {
        'relative_altitude': None,
        'sensor_width': None,
        'focal_length': None,
    }
    for key, val in tags.items():
        try:
            # exifread 的 value 可能是 IfdTag，取 values[0] 转为浮点数
            v = float(str(val)) if hasattr(val, 'printable') else float(val.values[0])
        except:
            continue
        key_lower = key.lower()
        if 'relativealtitude' in key_lower:
            result['relative_altitude'] = v
        elif 'sensorwidth' in key_lower:
            result['sensor_width'] = v
        elif 'calibratedfocallength' in key_lower or 'focallength' in key_lower:
            result['focal_length'] = v
    return result

def calculate_gsd(xmp_data: dict, img_width_px: int) -> float:
    try:
        rel_alt = xmp_data.get('relative_altitude')
        sensor_w = xmp_data.get('sensor_width')
        focal_len = xmp_data.get('focal_length')
        if rel_alt and sensor_w and focal_len and img_width_px > 0:
            return (rel_alt * sensor_w) / (focal_len * img_width_px)
    except:
        pass
    return DEFAULT_GSD

def natural_sort_frames(paths: list[str]) -> list[str]:
    def key(p: str) -> tuple[int, str]:
        m = re.search(r"(\d+)", Path(p).stem)
        return (int(m.group(1)), p) if m else (10**9, p)
    return sorted(paths, key=key)

def collect_frame_paths() -> list[str]:
    paths = glob.glob(FRAMES_GLOB)
    if not paths: paths = glob.glob(r"E:\SAM_data\frames\frame_*.jpeg")
    return natural_sort_frames(paths)

def load_locs(path: Path) -> list[dict[str, Any]]:
    if not path.is_file(): return []
    return json.loads(path.read_text(encoding="utf-8")).get("farmlands", [])

def load_union_mask(path: Path, target_hw: tuple[int, int]) -> np.ndarray:
    """读取 step3 生成的联合掩码"""
    if not path.is_file():
        sys.exit(f"❌ 联合掩码不存在: {path}\n请先运行 step3_merge_masks.py")
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        sys.exit(f"❌ 无法读取联合掩码: {path}")
    H, W = target_hw
    if mask.shape != (H, W):
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    return mask

def assign_global_ids(components_info, active_farms, next_id):
    results = []
    used_gids = set()
    for comp in components_info:
        cx, cy = comp['cx'], comp['cy']
        best_gid, min_d = None, float('inf')
        for gid, info in active_farms.items():
            if gid in used_gids: continue
            d = math.hypot(cx - info['cx'], cy - info['cy'])
            if d < min_d and d < MATCH_MAX_DIST: min_d, best_gid = d, gid
        if best_gid is not None:
            comp['global_id'] = best_gid
            active_farms[best_gid] = {'cx': cx, 'cy': cy}
            used_gids.add(best_gid)
        else:
            comp['global_id'] = next_id
            active_farms[next_id] = {'cx': cx, 'cy': cy}
            used_gids.add(next_id)
            next_id += 1
        results.append(comp)
    return results, active_farms, next_id

# 绘图函数直接接收并显示 "亩" 
def save_overlay(bgr, mask_list, area_mu_dict, out_path):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    cmap = plt.get_cmap("tab10")
    for idx, (label, mb) in enumerate(mask_list):
        if not mb.any(): continue
        color = np.array(cmap(idx % 10)[:3], dtype=np.float32)
        for c in range(3):
            rgb[..., c][mb] = rgb[..., c][mb] * (1 - OVERLAY_ALPHA) + color[c] * OVERLAY_ALPHA
    out_u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(out_u8, cv2.COLOR_BGR2RGB)
    
    for idx, (label, mb) in enumerate(mask_list):
        if not mb.any(): continue
        ys, xs = np.where(mb)
        cx, cy = int(xs.mean()), int(ys.mean())
        
        mu_val = area_mu_dict.get(label, 0)

        # text = f"ID:{label} {mu_val:.2f}mu"  # 如果需要用‘亩’标注，可以注释掉下面两行并用这行代码

        area_sqm = mu_val * 666.67
        text = f"ID:{label} {area_sqm:.1f}m2"
        
        font_scale = max(out_bgr.shape[1] / 1500, 0.6)
        thick = max(int(font_scale * 2), 1)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
        cv2.rectangle(out_bgr, (cx-2, cy-th-4), (cx+tw+2, cy+4), (0,0,0), -1)
        cv2.putText(out_bgr, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thick)
        
    cv2.imwrite(str(out_path), cv2.cvtColor(out_bgr, cv2.COLOR_RGB2BGR))

# ================= 主流程 =================
def main():
    if not WEIGHT_PATH.is_file(): sys.exit("错误: 找不到模型权重")
    frame_paths = collect_frame_paths()
    if not frame_paths: sys.exit("错误: 无帧图片")
    farms = load_locs(LOCS_PATH)
    if not farms: sys.exit("错误: locs.json 为空，请先运行 step2 标注")
    img0 = cv2.imread(str(FIRST_FRAME))
    if img0 is None: sys.exit("错误: 无法读取第一帧")
    H0, W0 = img0.shape[:2]

    # 在最开始计算真实物理换算系数 
    print("🔍 正在解析无人机底层物理参数...")
    xmp = get_dji_xmp_tags(FIRST_FRAME)
    gsd = calculate_gsd(xmp, W0)
    # 核心系数：1个像素等于多少亩？ (GSD的平方得到平方米，除以666.67得到亩)
    gsd_sq = gsd ** 2
    PX_TO_MU = gsd_sq / 666.67
    
    # 根据最小平方米阈值反算最小像素阈值
    gsd_sq = gsd ** 2  # 每像素 = 多少平方米
    MIN_COMPONENT_AREA = int(MIN_AREA_SQM / gsd_sq) if gsd_sq > 0 else 100
    print(f"   📏 最小面积过滤: {MIN_AREA_SQM} m² → {MIN_COMPONENT_AREA} 像素")
    
    if xmp:
        print(f"   ✅ 成功提取 XMP 数据，GSD = {gsd:.5f} 米/像素")
        print(f"   🎯 换算系数：1 像素 = {PX_TO_MU:.8f} 亩")
    else:
        print(f"   ⚠️ 未找到 XMP，使用估算 GSD = {gsd} 米/像素")
    print(f"   📏 最小面积过滤: {MIN_AREA_SQM} m² → {MIN_COMPONENT_AREA} 像素")

    print(f"🚀 读取联合掩码...")
    bridged_mask = load_union_mask(UNION_MASK_PATH, (H0, W0))
    debug_mask_path = Path(r"E:\SAM_data\debug_bridged_mask.png")
    cv2.imwrite(str(debug_mask_path), bridged_mask)

    from sam2.build_sam import build_sam2_video_predictor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = build_sam2_video_predictor(MODEL_CFG, str(WEIGHT_PATH), device=device)

    with tempfile.TemporaryDirectory(prefix="sam2_bridged_") as tmpdir:
        for i, src in enumerate(frame_paths):
            shutil.copy2(src, Path(tmpdir) / f"{i:06d}.jpg")
            
        inference_state = predictor.init_state(video_path=tmpdir, offload_video_to_cpu=True)
        vh, vw = int(inference_state["video_height"]), int(inference_state["video_width"])
        input_mask = bridged_mask
        if (vh, vw) != (H0, W0):
            input_mask = cv2.resize(bridged_mask, (vw, vh), interpolation=cv2.INTER_NEAREST)

        _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
            inference_state, frame_idx=0, obj_id=1, mask=torch.from_numpy(input_mask > 127)
        )

        area_records = []
        active_farms = {}  
        next_global_id = 1
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()
        
        print("📥 开始视频传播与'断桥'拆分...")
        with torch.inference_mode(), ctx:
            for frame_idx, obj_ids, video_res_masks in predictor.propagate_in_video(inference_state, start_frame_idx=0):
                bgr = cv2.imread(frame_paths[frame_idx])
                if bgr is None: continue
                H, W = bgr.shape[:2]

                vm = video_res_masks.detach().float().cpu().numpy()
                if vm.ndim != 4 or vm.shape[0] == 0: continue
                obj_idx = -1
                for i, oid in enumerate(obj_ids):
                    if oid == 1: obj_idx = i; break
                if obj_idx == -1: continue
                
                m = vm[obj_idx, 0] if vm.ndim == 4 else vm[obj_idx]
                if m.shape != (H, W): m = cv2.resize(m, (W, H))
                farm_mask_u8 = (m > 0.5).astype(np.uint8) * 255

                k_open = np.ones((CUT_OPEN_KERNEL, CUT_OPEN_KERNEL), np.uint8)
                farm_mask_cut = cv2.morphologyEx(farm_mask_u8, cv2.MORPH_OPEN, k_open)
                
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(farm_mask_cut, connectivity=8)
                
                components_info = []
                filtered_count = 0
                for label_id in range(1, num_labels):
                    area_px = stats[label_id, cv2.CC_STAT_AREA]
                    if area_px < MIN_COMPONENT_AREA:
                        filtered_count += 1
                        continue
                    
                    # 🌟 实时将像素面积转化为亩 🌟
                    area_mu = area_px * PX_TO_MU
                    
                    components_info.append({
                        'cx': float(centroids[label_id][0]),
                        'cy': float(centroids[label_id][1]),
                        'area_px': int(area_px),       # 保留像素，写给 JSON
                        'area_mu': area_mu,             # 亩数，写给图和 JSON
                        'mask': (labels == label_id)
                    })

                matched_comps, active_farms, next_global_id = assign_global_ids(
                    components_info, active_farms, next_global_id
                )

                OUT_OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
                out_path = OUT_OVERLAY_DIR / f"{Path(frame_paths[frame_idx]).stem}_overlay.jpg"
                
                mask_list_for_draw = [(c['global_id'], c['mask']) for c in matched_comps]
                # 🌟 传入亩数字典用于画图 🌟
                area_mu_dict_for_draw = {c['global_id']: c['area_mu'] for c in matched_comps}
                save_overlay(bgr, mask_list_for_draw, area_mu_dict_for_draw, out_path)

                # 写给 JSON 的数据（现在直接带真实面积了，Step5 可以直接读或者废弃）
                farmlands_stats = [{"id": c['global_id'], "area_pixels": c['area_px'], "area_mu": round(c['area_mu'], 2)} for c in matched_comps]
                area_records.append({
                    "frame_index": frame_idx,
                    "frame_name": Path(frame_paths[frame_idx]).name,
                    "farmlands": farmlands_stats
                })

                print(f"   帧 {frame_idx}: 拆分出 {len(matched_comps)} 块农田 (直接输出亩数)")

    def convert_to_native(obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        elif isinstance(obj, dict): return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list): return [convert_to_native(i) for i in obj]
        return obj

    with open(AREA_JSON, 'w', encoding='utf-8') as f:
        json.dump(convert_to_native(area_records), f, ensure_ascii=False, indent=2)
    print(f"\n🎉 处理完成！图片已直接标注'亩'数，JSON也同步更新: {AREA_JSON}")

if __name__ == "__main__":
    main()