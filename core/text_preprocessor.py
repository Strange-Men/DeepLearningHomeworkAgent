"""文本预处理模块 - 分词、去停用词、归一化。支持中/英/法多语言。"""

import os
import re

import config

# 内置英文/法文停用词集合
_LATIN_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must",
    "i", "me", "my", "we", "our", "you", "your", "he", "him", "his",
    "she", "her", "it", "its", "they", "them", "their",
    "this", "that", "these", "those", "what", "which", "who", "whom",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "into", "about", "between", "through", "after", "before", "above",
    "below", "up", "down", "out", "off", "over", "under",
    "and", "or", "not", "no", "nor", "but", "if", "then", "so", "than",
    "too", "very", "just", "also", "more", "most", "other", "some", "any",
    "each", "every", "all", "both", "few", "many", "much",
    "how", "when", "where", "why", "what", "which",
    "le", "la", "les", "de", "des", "du", "un", "une",
    "et", "ou", "mais", "donc", "car", "ni",
    "est", "sont", "a", "ont", "fait", "faire",
    "je", "tu", "il", "elle", "nous", "vous", "ils", "elles",
    "ce", "cette", "ces", "mon", "ton", "son", "notre", "votre", "leur",
    "qui", "que", "quoi", "dont", "où",
    "dans", "sur", "sous", "avec", "pour", "par", "en", "au", "aux",
    "pas", "plus", "aussi", "très", "bien", "tout", "tous",
    "comment", "quand", "où", "pourquoi", "quel", "quelle",
})


class TextPreprocessor:
    """多语言文本预处理器。"""

    def __init__(self, lang=None):
        self.lang = lang or config.DEFAULT_LANGUAGE
        self.stopwords = set()
        self._load_stopwords()
        if self.lang == "zh-CN":
            self._load_custom_dict()

    def _load_stopwords(self):
        """加载停用词表。"""
        stopwords_path = os.path.join(config.DATA_DIR, "stopwords.txt")
        if not os.path.exists(stopwords_path):
            return
        with open(stopwords_path, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if word:
                    self.stopwords.add(word)

    def _load_custom_dict(self):
        """加载 jieba 自定义词典（仅中文）。"""
        if self.lang not in ("zh-CN", "zh-TW"):
            return
        try:
            import jieba
            dict_path = os.path.join(config.DATA_DIR, "custom_dict.txt")
            if os.path.exists(dict_path):
                jieba.load_userdict(dict_path)
        except ImportError:
            pass

    def normalize(self, text):
        """文本归一化：去标点、统一空白、转小写。"""
        text = text.strip()
        text = re.sub(r"[？?！!。，,、；;：:\s]+", " ", text)
        text = text.lower().strip()
        return text

    def tokenize(self, text):
        """分词并去停用词，返回词列表。中文用 jieba，西文用正则。"""
        normalized = self.normalize(text)
        if not normalized:
            return []

        if self.lang in ("zh-CN", "zh-TW"):
            # 中文分词
            try:
                import jieba
                words = jieba.lcut(normalized)
            except ImportError:
                words = normalized.split()
        else:
            # 英文/法文分词：提取字母+数字 token
            words = re.findall(r"[a-zA-Z0-9]+", normalized)

        # 去停用词
        if self.lang in ("zh-CN", "zh-TW"):
            filtered = [
                w for w in words if w.strip() and w not in self.stopwords
            ]
        else:
            filtered = [
                w for w in words
                if w and w not in _LATIN_STOPWORDS
            ]
        return filtered

    def process(self, text):
        """完整预处理流水线，返回清洗后的文本（空格分隔）。"""
        tokens = self.tokenize(text)
        return " ".join(tokens)
