# -*- coding: utf-8 -*-
"""
Step 7: Streamlit 极简数据看板
运行方式: 在终端输入 streamlit run step7_dashboard.py
"""
import streamlit as st
import json
import pandas as pd
from pathlib import Path

# ================= 页面配置 =================
st.set_page_config(page_title="无人机农田智慧分析系统", layout="wide")

# ================= 数据加载 =================
AREA_JSON = Path(r"E:\SAM_data\real_area_list.json")
VIDEO_PATH = Path(r"E:\SAM_data\demo_comparison.mp4")

@st.cache_data
def load_data():
    if not AREA_JSON.exists():
        return None
    with open(AREA_JSON, 'r', encoding='utf-8') as f:
        return json.load(f)

data = load_data()

# ================= UI 布局 =================
st.title("🌾 无人机影像农田面积智能估算系统")
st.markdown("""基于 Meta SAM2 视觉大模型，融合大疆 XMP 底层物理参数，实现非接触式、亚像素级农田面积测算。""")

# 区域1：视频演示
st.subheader("📹 核心效果演示 (左: 原始影像 | 右: 实例分割与追踪)")
if VIDEO_PATH.exists():
    st.video(str(VIDEO_PATH))
else:
    st.warning("未找到 demo_comparison.mp4，请先运行 Step6。")

st.divider()

# 区域2：数据分析 (仅在数据存在时显示)
if data:
    st.subheader("📊 时序面积数据分析")
    
    # 将嵌套的 JSON 展平为 DataFrame
    rows = []
    for frame in data:
        frame_idx = frame["frame_index"]
        frame_name = frame["frame_name"]
        for farm in frame["farmlands"]:
            rows.append({
                "帧序号": frame_idx,
                "文件名": frame_name,
                "农田ID": farm["id"],
                "原始像素面积": farm["raw_area_pixels"],
                "滤波后像素面积": farm["filtered_area_pixels"],
                "真实面积(㎡)": farm["area_sqm"],
                "真实面积(亩)": round(farm["area_sqm"] / 666.67, 2)
            })
    
    df = pd.DataFrame(rows)
    
    # 图表1：每帧检测到的农田总面积变化（体现时序滤波的稳定性）
    st.markdown("##### 1. 农田总面积时序变化趋势 (㎡)")
    # 计算每帧所有农田的面积总和
    area_trend = df.groupby("帧序号")["真实面积(㎡)"].sum().reset_index()
    st.line_chart(area_trend, x="帧序号", y="真实面积(㎡)", height=300)
    st.caption("💡 曲线越平稳，说明 Step5 中的“时序防碎斑滤波”发挥的作用越好。")
    
    # 图表2：柱状图展示各农田最大面积
    st.markdown("##### 2. 各独立农田最大估算面积 (亩)")
    max_area = df.groupby("农田ID")["真实面积(亩)"].max().reset_index()
    st.bar_chart(max_area, x="农田ID", y="真实面积(亩)", height=300)
    
    # 数据明细表
    with st.expander("🔍 查看全量逐帧数据明细"):
        st.dataframe(df, width="stretch")
        
else:
    st.error("未找到 real_area_list.json，请先完成 Step5 空间计算。")

# 底部技术栈说明
st.divider()
st.caption("Tech Stack: Python / OpenCV / PyTorch / SAM 2.1 (Hiera-S) / Streamlit / Morphology Math")