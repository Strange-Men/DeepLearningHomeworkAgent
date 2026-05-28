"""文本预处理模块 - 分词、去停用词、归一化。"""

import os
import re

import jieba

import config


class TextPreprocessor:
    """中文文本预处理器。"""

    def __init__(self):
        self.stopwords = set()
        self._load_stopwords()
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
        """加载 jieba 自定义词典。"""
        dict_path = os.path.join(config.DATA_DIR, "custom_dict.txt")
        if os.path.exists(dict_path):
            jieba.load_userdict(dict_path)

    def normalize(self, text):
        """文本归一化：去标点、统一空白、转小写。"""
        text = text.strip()
        text = re.sub(r"[？?！!。，,、；;：:\s]+", " ", text)
        text = text.lower().strip()
        return text

    def tokenize(self, text):
        """分词并去停用词，返回词列表。"""
        normalized = self.normalize(text)
        if not normalized:
            return []
        words = jieba.lcut(normalized)
        filtered = [w for w in words if w.strip() and w not in self.stopwords]
        return filtered

    def process(self, text):
        """完整预处理流水线，返回清洗后的文本（空格分隔）。"""
        tokens = self.tokenize(text)
        return " ".join(tokens)
