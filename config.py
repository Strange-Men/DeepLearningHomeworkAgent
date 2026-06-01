"""全局配置模块 - 集中管理所有配置项。"""

import os
import logging
from logging.handlers import RotatingFileHandler

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

# 日志配置
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "agent.log")
os.makedirs(LOG_DIR, exist_ok=True)

# 应用配置
APP_HOST = "127.0.0.1"
APP_PORT = 7860
APP_SHARE = False

# Agent配置
AGENT_SIMILARITY_THRESHOLD = 0.2
AGENT_RULE_HIGH_THRESHOLD = 0.3
AGENT_RULE_LOW_THRESHOLD = 0.2
AGENT_LLM_ENABLED = True
AGENT_LLM_TIMEOUT = 15
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

# 检索器后端配置
RETRIEVER_BACKEND = "tfidf"  # 可选: "tfidf", "vector"

# RAG 向量检索配置
CHROMA_PERSIST_DIR = os.path.join(DATA_DIR, "chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RAG_CHUNK_SIZE = 500
RAG_CHUNK_OVERLAP = 50
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_ENDPOINT"] = HF_ENDPOINT

# Agent Loop 配置
AGENT_MAX_TURNS = 3
AGENT_MAX_TOOL_CALLS = 6
AGENT_MAX_SAME_CALLS = 2
AGENT_MAX_PARSE_FAILS = 2
AGENT_FOLLOWUP_MAX_TOKENS = 256
AGENT_FOLLOWUP_TIMEOUT = 10

# 模型训练结果路径
TRAIN_RESULTS_PATH = os.path.join(
    BASE_DIR, "runs", "detect", "train4", "results.csv"
)
TRAIN_ARGS_PATH = os.path.join(
    BASE_DIR, "runs", "detect", "train4", "args.yaml"
)


def setup_logging():
    """初始化全局日志配置：控制台 WARNING + 文件 INFO。"""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 文件处理器：INFO 级别，单文件最大 5MB，保留 3 个备份
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)

    # 控制台处理器：WARNING 级别（避免刷屏）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(fmt)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
