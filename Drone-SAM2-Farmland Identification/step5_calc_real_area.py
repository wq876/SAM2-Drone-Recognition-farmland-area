"""
空间计算模块 V2.0：
1. 优先使用大疆 XMP 真实数据计算物理面积
2. 时序平滑滤波：消除追踪碎斑导致的面积突变
3. 优雅降级：无真实数据时自动走估算逻辑
"""
import json
import math
import os
import sys
from pathlib import Path

try:
    import exifread
    HAS_EXIF = True
except ImportError:
    HAS_EXIF = False

FRAMES_DIR = Path(r"E:\SAM_data\frames")
FIRST_FRAME = Path(r"E:\SAM_data\frames\frame_0001.jpg")
AREA_JSON = Path(r"E:\SAM_data\area_list.json")
OUT_REAL_AREA_JSON = Path(r"E:\SAM_data\real_area_list.json")

DEFAULT_GSD = 0.05  # 默认降级参数

# ================= 时序滤波参数 =================
# 如果当前帧面积突然小于上一帧面积的 30%，认为是碎斑干扰
DROP_THRESHOLD_RATIO = 0.3  

def get_dji_xmp_tags(image_path: Path) -> dict:
    """提取大疆底层的 XMP 物理参数"""
    if not HAS_EXIF or not image_path.is_file():
        return {}
    tags = {}
    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f, details=False)
    except Exception:
        return {}

    result = {}
    for key, val in tags.items():
        if 'XMP' in key:
            try:
                result[key] = float(val.values[0]) if hasattr(val, 'values') else float(val)
            except:
                pass
    return result

def calculate_real_gsd(xmp_data: dict, img_width_px: int) -> float | None:
    """
    🌟 核心物理公式计算真实 GSD (Ground Sample Distance)
    公式：GSD = (相对航高 * 传感器宽度) / (焦距 * 图像宽度像素)
    """
    try:
        rel_alt = xmp_data.get('XMP RelativeAltitude') # 相对起飞点高度
        sensor_w = xmp_data.get('XMP SensorWidth')      # 传感器物理宽度
        focal_len = xmp_data.get('XMP CalibratedFocalLength') # 标定焦距
        
        if rel_alt and sensor_w and focal_len and img_width_px > 0:
            gsd = (rel_alt * sensor_w) / (focal_len * img_width_px)
            return gsd
    except Exception:
        pass
    return None

def get_image_width_px(image_path: Path) -> int:
    """获取图片像素宽度"""
    try:
        import cv2
        img = cv2.imread(str(image_path))
        if img is not None:
            return img.shape[1]
    except:
        pass
    return 0

def main():
    if not AREA_JSON.is_file():
        print(f"错误: 找不到 {AREA_JSON}")
        sys.exit(1)

    print("🚀 启动高级空间分析引擎...")
    
    # --- 1. GSD 计算策略 ---
    gsd = None
    img_w = get_image_width_px(FIRST_FRAME)
    
    if FIRST_FRAME.is_file() and img_w > 0:
        print(f"> 检测到首帧图像宽度: {img_w}px")
        xmp = get_dji_xmp_tags(FIRST_FRAME)
        if xmp:
            print(f"  ✅ 成功提取无人机底层物理参数:")
            print(f"     - 相对航高: {xmp.get('XMP RelativeAltitude', 'N/A')} 米")
            print(f"     - 焦距: {xmp.get('XMP CalibratedFocalLength', 'N/A')} mm")
            print(f"     - 传感器宽: {xmp.get('XMP SensorWidth', 'N/A')} mm")
            
            gsd = calculate_real_gsd(xmp, img_w)
            if gsd and gsd > 0:
                print(f"  🎯 基于物理公式计算真实 GSD = {gsd:.5f} 米/像素")
            else:
                print("  ⚠️ 参数不完整，无法精确计算GSD。")
        else:
            if not HAS_EXIF:
                print("  ⚠️ 缺少 exifread 库。")
            else:
                print("  ⚠️ 图片中未检测到大疆 XMP 数据。")

    if gsd is None or gsd <= 0:
        gsd = DEFAULT_GSD
        print(f"> 🔄 触发降级策略：使用估算 GSD = {gsd} 米/像素")

    gsd_sq = gsd ** 2
    print(f"> 面积换算系数：1 像素 ≈ {gsd_sq:.6f} 平方米\n")

    # --- 2. 读取数据并进行时序滤波 ---
    with open(AREA_JSON, 'r', encoding='utf-8') as f:
        area_list = json.load(f)

    real_area_list = []
    max_areas = {}         # 记录全局最大面积
    prev_areas = {}        # 🌟 记录上一帧的面积，用于时序滤波防碎斑
    smooth_log = []        # 记录被平滑掉的日志

    for frame_data in area_list:
        frame_idx = frame_data["frame_index"]
        real_farmlands = []
        
        for farm in frame_data["farmlands"]:
            fid = farm["id"]
            raw_px = farm["area_pixels"]
            
            # 🌟🌟🌟 时序滤波核心逻辑 🌟🌟🌟
            filtered_px = raw_px
            if fid in prev_areas:
                prev_px = prev_areas[fid]
                # 如果上一帧有追踪到，但这帧面积突然断崖式下跌
                if raw_px < (prev_px * DROP_THRESHOLD_RATIO):
                    filtered_px = prev_px  # 强制保持上一帧的面积，屏蔽碎斑干扰
                    smooth_log.append(f"帧 {frame_idx} ID {fid}: 面积突变 ({prev_px}->{raw_px})，已平滑拦截")
            
            # 更新历史记录
            if filtered_px > 0:
                prev_areas[fid] = filtered_px
            else:
                # 如果面积为0（飞出画面了），清除历史，避免下次回来时误判为突变
                prev_areas.pop(fid, None)

            # 计算真实面积
            real_sqm = filtered_px * gsd_sq
            real_farmlands.append({
                "id": fid,
                "raw_area_pixels": raw_px,      # 保留原始值供对比
                "filtered_area_pixels": filtered_px, # 滤波后的像素
                "area_sqm": round(real_sqm, 2)
            })
            
            # 更新全局最大面积
            if real_sqm > max_areas.get(fid, 0):
                max_areas[fid] = real_sqm
                
        real_area_list.append({
            "frame_index": frame_idx,
            "frame_name": frame_data["frame_name"],
            "farmlands": real_farmlands
        })

    # --- 3. 输出结果 ---
    if smooth_log:
        print("🛡️ 时序防碎斑日志:")
        for log in smooth_log[:5]: # 只打印前5条避免刷屏
            print(f"   - {log}")
        if len(smooth_log) > 5:
            print(f"   ... 共拦截了 {len(smooth_log)} 次碎斑突变")
        print()

    print("="*60)
    print("📊 最终农田面积估算结果 (已过滤追踪碎斑):")
    for fid in sorted(max_areas.keys()):
        sqm = max_areas[fid]
        mu = sqm / 666.67  # 平方米转亩 (1亩 ≈ 666.67平方米)
        print(f"  农田 ID {fid}: 最大估算面积 {sqm:.2f} 平方米 (约合 {mu:.2f} 亩)")
    print("="*60)

    with open(OUT_REAL_AREA_JSON, 'w', encoding='utf-8') as f:
        json.dump(real_area_list, f, ensure_ascii=False, indent=2)
        
    print(f"\n💾 最终真实面积数据已保存至: {OUT_REAL_AREA_JSON}")

if __name__ == "__main__":
    main()