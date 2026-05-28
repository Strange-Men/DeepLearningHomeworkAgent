"""全局配置模块 - 集中管理所有配置项。"""

import os

from dotenv import load_dotenv

# 加载 .env 文件（如果存在）
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    load_dotenv(_env_path)

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 模型配置
MODEL_PATH = os.path.join(BASE_DIR, "runs", "detect", "train4", "weights", "best.pt")
MODEL_IMAGE_SIZE = 640
MODEL_CONFIDENCE_THRESHOLD = 0.25

# 手势类别名称
GESTURE_CLASSES = [
    "A", "D", "I", "L", "V", "W", "Y",
    "number 5", "number 7", "I love you"
]

# 数据库配置
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "history.db")
KNOWLEDGE_BASE_PATH = os.path.join(DATA_DIR, "knowledge_base.json")

# 国际化配置
DEFAULT_LANGUAGE = "zh-CN"
LANG_NAMES = {
    "zh-CN": "简体中文",
    "zh-TW": "繁體中文",
    "en": "English",
    "fr": "Français",
}
KNOWLEDGE_BASE_PATHS = {
    "zh-CN": os.path.join(DATA_DIR, "knowledge_base.json"),
    "zh-TW": os.path.join(DATA_DIR, "knowledge_base_zh-TW.json"),
    "en": os.path.join(DATA_DIR, "knowledge_base_en.json"),
    "fr": os.path.join(DATA_DIR, "knowledge_base_fr.json"),
}

# 检测结果保存目录
RESULTS_DIR = os.path.join(BASE_DIR, "results")
HISTORY_VIDEOS_DIR = os.path.join(DATA_DIR, "history_videos")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(HISTORY_VIDEOS_DIR, exist_ok=True)

# 应用配置
APP_HOST = "127.0.0.1"
APP_PORT = 7860
APP_SHARE = False

# Agent配置
AGENT_SIMILARITY_THRESHOLD = 0.2
AGENT_RULE_HIGH_THRESHOLD = 0.3
AGENT_RULE_LOW_THRESHOLD = 0.2
AGENT_LLM_ENABLED = True
AGENT_LLM_TIMEOUT = 30
AGENT_LLM_MAX_RETRIES = 2
AGENT_LLM_MAX_HISTORY = 5

# MiMo API 配置
MIMO_API_KEY = os.getenv("MIMO_API_KEY", "")
MIMO_BASE_URL = os.getenv("MIMO_BASE_URL", "https://api.mimo.com/v1")
MIMO_MODEL_NAME = os.getenv("MIMO_MODEL_NAME", "mimo-2.5-pro")

if not MIMO_API_KEY:
    print("[WARNING] MIMO_API_KEY not set. LLM layer disabled. "
          "Set it in .env file.")

# Agent RAG 配置
AGENT_RAG_ENABLED = True
AGENT_RAG_MAX_CONTEXT_LEN = 1500

# 模型训练结果路径
TRAIN_RESULTS_PATH = os.path.join(
    BASE_DIR, "runs", "detect", "train4", "results.csv"
)
TRAIN_ARGS_PATH = os.path.join(
    BASE_DIR, "runs", "detect", "train4", "args.yaml"
)
