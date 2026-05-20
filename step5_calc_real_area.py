# -*- coding: utf-8 -*-
"""
Step5 ：时序滤波 + 最终汇总报告
依赖：step4 生成的 area_list.json
支持平方米/亩可选输出
"""
import json
import sys
from pathlib import Path
import os
from config import BASE_DIR
os.chdir(BASE_DIR)

from config import (
    AREA_JSON, REPORT_JSON,
    DROP_THRESHOLD_RATIO, MIN_AREA_FILTER, UNIT, MU_TO_SQM
)
OUT_REPORT = REPORT_JSON
MIN_AREA = MIN_AREA_FILTER


def format_area(value: float) -> str:
    """根据 UNIT 格式化面积"""
    if UNIT == "mu":
        return f"{value:.2f} 亩"
    else:
        return f"{value:.1f} 平方米"


def main():
    if not AREA_JSON.is_file():
        print(f"错误: 找不到 {AREA_JSON}")
        sys.exit(1)

    with open(AREA_JSON, 'r', encoding='utf-8') as f:
        area_list = json.load(f)

    max_areas = {}       # {id: 最大面积}
    prev_areas = {}      # {id: 上一帧面积}
    smooth_log = []      # 滤波日志

    for frame in area_list:
        for farm in frame.get("farmlands", []):
            fid = farm["id"]

            # 读取面积：优先 area_mu（亩），其次 area_pixels
            if "area_mu" in farm:
                raw_mu = farm["area_mu"]
            elif "area_pixels" in farm:
                raw_mu = farm["area_pixels"] * 0.00000375  # 降级估算
            else:
                continue

            # 根据单位选择使用的值
            raw = raw_mu if UNIT == "mu" else raw_mu * MU_TO_SQM

            # 时序滤波
            filtered = raw
            if fid in prev_areas:
                if raw < prev_areas[fid] * DROP_THRESHOLD_RATIO:
                    filtered = prev_areas[fid]
                    smooth_log.append(
                        f"帧 {frame['frame_index']} ID {fid}: "
                        f"{format_area(prev_areas[fid])} → {format_area(raw)}，已平滑"
                    )

            if filtered > MIN_AREA:
                prev_areas[fid] = filtered
            else:
                prev_areas.pop(fid, None)

            # 更新最大面积
            if filtered > max_areas.get(fid, 0):
                max_areas[fid] = filtered

    # 打印滤波日志
    if smooth_log:
        print("🛡️ 时序防碎斑日志:")
        for log in smooth_log[:5]:
            print(f"   {log}")
        if len(smooth_log) > 5:
            print(f"   ... 共 {len(smooth_log)} 条")
        print()

    # 最终汇总
    print("=" * 50)
    unit_label = "亩" if UNIT == "mu" else "平方米"
    print(f"📊 各农田最大估算面积 (单位: {unit_label}):")
    total = 0
    for fid in sorted(max_areas.keys()):
        val = max_areas[fid]
        total += val
        print(f"  农田 ID {fid}: {format_area(val)}")
    print(f"  总计: {format_area(total)}")
    print("=" * 50)

    # 保存报告
    report = {
        "unit": UNIT,
        "total_farmlands": len(max_areas),
        "total_area": round(total, 2),
        "farmlands": [
            {"id": fid, "max_area": round(val, 2)}
            for fid, val in sorted(max_areas.items())
        ]
    }
    with open(OUT_REPORT, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n📄 汇总报告已保存: {OUT_REPORT}")


if __name__ == "__main__":
    main()