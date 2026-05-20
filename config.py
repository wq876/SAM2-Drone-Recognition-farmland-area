# config.py
"""
项目统一路径与参数配置
所有路径均基于本文件所在目录（项目根目录），无需修改绝对路径。
"""
from pathlib import Path

# 项目根目录（本文件所在文件夹）
BASE_DIR = Path(__file__).resolve().parent

# ===================== 目录结构 =====================
DATA_DIR = BASE_DIR / "data"                 # 原始数据（视频等）
FRAMES_DIR = DATA_DIR / "frames"            # 抽帧图像输出 / 读取
WEIGHTS_DIR = BASE_DIR / "weights"          # 模型权重
MASK_DIR = BASE_DIR / "farm_masks"          # 手动标注的 mask
OVERLAY_DIR = BASE_DIR / "masks_frame"      # 每帧追踪叠加图
ASSETS_DIR = BASE_DIR / "assets"            # 演示素材（视频、截图）
CONFIG_DIR = BASE_DIR / "configs"           # SAM2 配置文件目录

# ===================== 文件路径 =====================
# --- 视频与帧 ---
VIDEO_PATH = DATA_DIR / "Farmland.mp4"                  # 原始视频
FIRST_FRAME = FRAMES_DIR / "frame_0001.jpg"             # 第一帧
FRAMES_GLOB = str(FRAMES_DIR / "frame_*.jpg")           # 帧文件匹配模式

# --- 模型 ---
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"       # 相对于 BASE_DIR
WEIGHT_PATH = WEIGHTS_DIR / "sam2.1_hiera_small.pt"

# --- 标注与掩码 ---
LOCS_PATH = BASE_DIR / "locs.json"                      # 单块农田标注坐标
OVERLAY_PATH = BASE_DIR / "mask_overlay.jpg"            # 第一帧叠加图
UNION_MASK_PATH = MASK_DIR / "farm_mask_union.png"      # 联合掩码（step3 生成）
UNION_LOCS_PATH = BASE_DIR / "locs_union.json"          # 联合掩码中心坐标

# --- 输出 ---
AREA_JSON = BASE_DIR / "area_list.json"                 # 每帧面积（step4 生成）
REPORT_JSON = BASE_DIR / "final_report.json"            # 最终汇总报告（step5 生成）
DEMO_VIDEO = BASE_DIR / "demo_comparison.mp4"           # 演示视频（step6 生成）

# ===================== 可调参数 =====================
# 抽帧
OUTPUT_FPS = 2                                          # 每秒提取帧数

# 交互式标注
PREVIEW_ALPHA = 0.45
CONFIRMED_ALPHA = 0.22
PT_RADIUS = 6
BGR_POS = (0, 255, 0)                                   # 正样本点颜色（绿）
BGR_NEG = (0, 0, 255)                                   # 负样本点颜色（红）

# 联合掩码追踪（step4）
OVERLAY_ALPHA = 0.45                                    # 追踪叠加透明度
MIN_COMPONENT_AREA_PX = 100                             # 最小连通域像素面积（动态调整前默认值）
MATCH_MAX_DIST = 100                                    # 跨帧匹配最大中心距离
BRIDGE_CLOSE_KERNEL = 21                                # 闭运算核（搭桥）
CUT_OPEN_KERNEL = 7                                     # 开运算核（断桥）
DEFAULT_GSD = 0.05                                      # 无XMP时默认地面采样距离（米/像素）
MIN_AREA_SQM = 20.0                                     # 低于此面积的碎斑不显示/不记录

# 空间计算（step5）
DROP_THRESHOLD_RATIO = 0.3                              # 时序滤波：面积突降比例
MIN_AREA_FILTER = 1.0                                   # 最终报告忽略的最小面积（m² 或 亩）
UNIT = "sqm"                                            # 面积单位："sqm" 或 "mu"
MU_TO_SQM = 666.67                                      # 1亩 = 666.67 平方米

# 演示视频（step6）
VIDEO_FPS = 24.0
FRAME_REPEAT = 12
TEXT_FONT_NAME = "FONT_HERSHEY_SIMPLEX"                 # OpenCV 字体名（字符串）