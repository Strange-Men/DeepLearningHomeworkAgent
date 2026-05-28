"""国际化模块 - 多语言支持核心。"""

import json
import logging
import os

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = ("zh-CN", "zh-TW", "en", "fr")
DEFAULT_LANGUAGE = "zh-CN"

_current_lang = DEFAULT_LANGUAGE
_translations = {}


def _locales_dir():
    """返回 locales 目录的绝对路径。"""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "locales")


def init():
    """加载 locales/ 下所有 JSON 翻译文件。"""
    global _translations
    _translations = {}
    locales_dir = _locales_dir()
    if not os.path.isdir(locales_dir):
        logger.warning("Locales directory not found: %s", locales_dir)
        return
    for lang in SUPPORTED_LANGUAGES:
        path = os.path.join(locales_dir, f"{lang}.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    _translations[lang] = json.load(f)
                logger.debug("Loaded locale: %s", lang)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load locale %s: %s", lang, e)
        else:
            logger.warning("Locale file not found: %s", path)


def set_language(lang):
    """设置当前语言。

    Args:
        lang: 语言代码，必须在 SUPPORTED_LANGUAGES 中

    Raises:
        ValueError: 不支持的语言代码
    """
    global _current_lang
    if lang not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language: {lang}. "
            f"Supported: {SUPPORTED_LANGUAGES}"
        )
    _current_lang = lang


def get_language():
    """返回当前语言代码。"""
    return _current_lang


def t(key, **kwargs):
    """翻译函数。回退链：当前语言 -> zh-CN -> 原始 key。

    Args:
        key: 翻译键名
        **kwargs: 用于 str.format() 的命名参数

    Returns:
        str: 翻译后的文本
    """
    value = None
    # 当前语言
    if _current_lang in _translations:
        value = _translations[_current_lang].get(key)
    # 回退到 zh-CN
    if value is None and _current_lang != DEFAULT_LANGUAGE:
        if DEFAULT_LANGUAGE in _translations:
            value = _translations[DEFAULT_LANGUAGE].get(key)
    # 最终回退：原始 key
    if value is None:
        value = key

    if kwargs:
        try:
            return value.format(**kwargs)
        except (KeyError, IndexError, AttributeError):
            return value
    return value
