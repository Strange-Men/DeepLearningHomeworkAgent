"""Chroma 向量 RAG 检索器 - 基于 sentence-transformers + ChromaDB。"""

import json
import logging
import os
import re
import time

import config

logger = logging.getLogger(__name__)

_MODEL_DOWNLOAD_MAX_RETRIES = 3
_MODEL_DOWNLOAD_TIMEOUT = 120


def _split_text(text, chunk_size=500, overlap=50):
    """将文本按 chunk_size 分块，overlap 为重叠字符数。"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
    return chunks


def _extract_code_functions(base_dir):
    """从核心源码中提取函数/类定义及 docstring。"""
    search_files = [
        os.path.join(base_dir, "core", "detector.py"),
        os.path.join(base_dir, "core", "history.py"),
        os.path.join(base_dir, "core", "agent.py"),
        os.path.join(base_dir, "core", "llm.py"),
        os.path.join(base_dir, "core", "tools.py"),
        os.path.join(base_dir, "core", "rag_retriever.py"),
        os.path.join(base_dir, "app.py"),
        os.path.join(base_dir, "config.py"),
    ]
    docs = []
    for fpath in search_files:
        if not os.path.exists(fpath):
            continue
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            continue
        fname = os.path.relpath(fpath, base_dir)
        # 提取函数和类定义及其 docstring
        lines = content.split("\n")
        for i, line in enumerate(lines):
            m = re.match(r"\s*(?:def|class)\s+(\w+)", line)
            if not m:
                continue
            name = m.group(1)
            docstring = ""
            # 在函数定义行之后查找紧跟的 docstring
            for j in range(i + 1, min(i + 5, len(lines))):
                stripped = lines[j].strip()
                if not stripped:
                    continue
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    quote = stripped[:3]
                    # 单行 docstring
                    if stripped.count(quote) >= 2:
                        docstring = stripped[3:-3].strip()
                    else:
                        # 多行 docstring
                        doc_lines = [stripped[3:]]
                        for k in range(j + 1, len(lines)):
                            if quote in lines[k]:
                                doc_lines.append(
                                    lines[k].split(quote)[0]
                                )
                                break
                            doc_lines.append(lines[k])
                        docstring = "\n".join(doc_lines).strip()
                break
            if docstring:
                text = (
                    f"文件 {fname} 中的 {name} 函数: {docstring}"
                )
                docs.append(text)
    return docs


def _load_md_documents(base_dir):
    """加载 .md 文档内容。"""
    md_files = [
        os.path.join(base_dir, "CLAUDE.md"),
        os.path.join(base_dir, "README.md"),
        os.path.join(base_dir, "AgentMaker.md"),
    ]
    docs = []
    for fpath in md_files:
        if not os.path.exists(fpath):
            continue
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            fname = os.path.basename(fpath)
            chunks = _split_text(
                content,
                chunk_size=config.RAG_CHUNK_SIZE,
                overlap=config.RAG_CHUNK_OVERLAP,
            )
            for chunk in chunks:
                docs.append(f"[{fname}] {chunk}")
        except Exception as e:
            logger.warning("Failed to load %s: %s", fpath, e)
    return docs


def _load_essay_docx(base_dir):
    """加载 Essay.docx 论文内容。"""
    docx_path = os.path.join(base_dir, "Essay.docx")
    if not os.path.exists(docx_path):
        return []
    try:
        from docx import Document
        doc = Document(docx_path)
        paragraphs = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                paragraphs.append(text)
        full_text = "\n".join(paragraphs)
        chunks = _split_text(
            full_text,
            chunk_size=config.RAG_CHUNK_SIZE,
            overlap=config.RAG_CHUNK_OVERLAP,
        )
        return [f"[Essay] {c}" for c in chunks]
    except ImportError:
        logger.warning(
            "python-docx not installed, skipping Essay.docx"
        )
        return []
    except Exception as e:
        logger.warning("Failed to load Essay.docx: %s", e)
        return []


def _load_knowledge_base(base_dir):
    """加载 knowledge_base.json 问答对。"""
    kb_path = os.path.join(base_dir, "data", "knowledge_base.json")
    if not os.path.exists(kb_path):
        return []
    try:
        with open(kb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        docs = []
        for qa in data.get("qa_pairs", []):
            q = qa.get("question", "")
            a = qa.get("answer", "")
            category = qa.get("category", "")
            text = f"问题: {q}\n分类: {category}\n回答: {a}"
            docs.append(text)
        return docs
    except Exception as e:
        logger.warning("Failed to load knowledge_base.json: %s", e)
        return []


class ChromaRetriever:
    """基于 ChromaDB + sentence-transformers 的向量检索器。"""

    def __init__(self):
        self._collection = None
        self._model = None
        self._client = None
        self._quick_questions = []
        self._default_response = ""
        self._initialized = False
        # 主动加载嵌入模型（首次启动时通过镜像下载并缓存）
        try:
            self._ensure_model()
        except Exception:
            # 模型加载失败不阻塞启动，后续 build_index 会再次尝试
            pass

    def _ensure_model(self):
        """延迟加载嵌入模型（含重试机制）。"""
        if self._model is None:
            t0 = time.time()
            logger.debug("ChromaRetriever._ensure_model: loading model...")
            import os
            # 设置 HuggingFace 镜像
            if config.HF_ENDPOINT:
                os.environ["HF_ENDPOINT"] = config.HF_ENDPOINT
            cache_dir = os.path.join(config.DATA_DIR, "hf_models")
            from sentence_transformers import SentenceTransformer

            last_err = None
            for attempt in range(1, _MODEL_DOWNLOAD_MAX_RETRIES + 1):
                try:
                    self._model = SentenceTransformer(
                        config.EMBEDDING_MODEL,
                        cache_folder=cache_dir,
                    )
                    logger.info(
                        "Loaded embedding model: %s",
                        config.EMBEDDING_MODEL,
                    )
                    logger.debug("ChromaRetriever._ensure_model: model loaded in %.3fs", time.time()-t0)
                    return
                except Exception as e:
                    last_err = e
                    logger.warning(
                        "Embedding model load attempt %d/%d "
                        "failed: %s",
                        attempt, _MODEL_DOWNLOAD_MAX_RETRIES, e,
                    )
                    if attempt < _MODEL_DOWNLOAD_MAX_RETRIES:
                        time.sleep(3)

            # 所用重试均失败
            hf_endpoint = config.HF_ENDPOINT or "未设置"
            logger.error(
                "嵌入模型加载失败（已重试%d次）。\n"
                "  模型: %s\n"
                "  HF_ENDPOINT: %s\n"
                "  缓存目录: %s\n"
                "  错误: %s\n"
                "请检查：\n"
                "  1. 网络是否可访问 %s\n"
                "  2. 如需手动下载，请先设置环境变量：\n"
                "     set HF_ENDPOINT=https://hf-mirror.com\n"
                "     然后运行：\n"
                "     python -c \"from sentence_transformers "
                "import SentenceTransformer; "
                "SentenceTransformer('%s', "
                "cache_folder='%s')\"\n"
                "  3. 也可直接从镜像站下载模型文件到上述缓存目录",
                _MODEL_DOWNLOAD_MAX_RETRIES,
                config.EMBEDDING_MODEL,
                hf_endpoint,
                cache_dir,
                last_err,
                hf_endpoint,
                config.EMBEDDING_MODEL,
                cache_dir,
            )
            raise last_err

    def _ensure_chroma(self):
        """延迟初始化 ChromaDB。"""
        if self._client is None:
            t0 = time.time()
            import chromadb
            self._client = chromadb.PersistentClient(
                path=config.CHROMA_PERSIST_DIR
            )
            logger.info(
                "ChromaDB initialized at %s", config.CHROMA_PERSIST_DIR
            )
            logger.debug("ChromaRetriever._ensure_chroma: initialized in %.3fs", time.time()-t0)

    def build_index(self, lang=None):
        """构建向量索引。从多种数据源加载文档并嵌入。"""
        t0 = time.time()
        logger.debug("ChromaRetriever.build_index: START | lang=%s", lang)
        try:
            self._ensure_model()
        except Exception as e:
            logger.warning(
                "Failed to load embedding model: %s. "
                "RAG retrieval disabled. Set HF_ENDPOINT or "
                "download model manually.", e
            )
            self._initialized = True
            return
        self._ensure_chroma()

        collection_name = f"knowledge_{lang or config.DEFAULT_LANGUAGE}"
        # 删除旧集合重建
        try:
            self._client.delete_collection(collection_name)
        except Exception:
            pass
        self._collection = self._client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # 加载所有数据源
        all_docs = []
        all_ids = []
        doc_id = 0

        # 1. Essay.docx
        essay_docs = _load_essay_docx(config.BASE_DIR)
        for d in essay_docs:
            all_docs.append(d)
            all_ids.append(f"essay_{doc_id}")
            doc_id += 1

        # 2. .md 文档
        md_docs = _load_md_documents(config.BASE_DIR)
        for d in md_docs:
            all_docs.append(d)
            all_ids.append(f"md_{doc_id}")
            doc_id += 1

        # 3. 代码函数
        code_docs = _extract_code_functions(config.BASE_DIR)
        for d in code_docs:
            all_docs.append(d)
            all_ids.append(f"code_{doc_id}")
            doc_id += 1

        # 4. knowledge_base.json
        kb_path = config.KNOWLEDGE_BASE_PATHS.get(
            lang, config.KNOWLEDGE_BASE_PATH
        )
        if os.path.exists(kb_path):
            try:
                with open(kb_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._quick_questions = [
                    qa["question"]
                    for qa in data.get("qa_pairs", [])[:10]
                ]
                self._default_response = data.get(
                    "default_response", ""
                )
                for qa in data.get("qa_pairs", []):
                    q = qa.get("question", "")
                    a = qa.get("answer", "")
                    category = qa.get("category", "")
                    text = (
                        f"问题: {q}\n分类: {category}\n回答: {a}"
                    )
                    all_docs.append(text)
                    all_ids.append(f"kb_{doc_id}")
                    doc_id += 1
            except Exception as e:
                logger.warning("Failed to load KB: %s", e)

        if not all_docs:
            logger.warning("No documents to index")
            self._initialized = True
            logger.debug("ChromaRetriever.build_index: no docs, DONE in %.3fs", time.time()-t0)
            return

        logger.debug("ChromaRetriever.build_index: embedding %d docs...", len(all_docs))
        # 批量嵌入并写入 ChromaDB
        batch_size = 100
        for i in range(0, len(all_docs), batch_size):
            batch_docs = all_docs[i:i + batch_size]
            batch_ids = all_ids[i:i + batch_size]
            embeddings = self._model.encode(
                batch_docs, show_progress_bar=False
            ).tolist()
            self._collection.add(
                documents=batch_docs,
                embeddings=embeddings,
                ids=batch_ids,
            )

        self._initialized = True
        logger.info(
            "Chroma index built: %d documents in collection '%s'",
            len(all_docs), collection_name,
        )
        logger.debug("ChromaRetriever.build_index: DONE in %.3fs | %d docs", time.time()-t0, len(all_docs))

    def search(self, query, top_k=3):
        """向量检索，返回相关文档列表。"""
        if not self._initialized or self._collection is None:
            logger.debug("ChromaRetriever.search: not initialized, return []")
            return []
        if not query or not query.strip():
            return []

        t0 = time.time()
        self._ensure_model()
        query_embedding = self._model.encode(
            [query], show_progress_bar=False
        ).tolist()
        t_encode = time.time() - t0

        t1 = time.time()
        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self._collection.count()),
        )
        t_query = time.time() - t1

        docs = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                distance = (
                    results["distances"][0][i]
                    if results.get("distances")
                    else 0.0
                )
                # ChromaDB cosine distance: 0=identical, 2=opposite
                score = max(0.0, 1.0 - distance)
                docs.append({"text": doc, "score": score})
        logger.debug(
            "ChromaRetriever.search: DONE | encode=%.3fs | query=%.3fs "
            "| total=%.3fs | results=%d",
            t_encode, t_query, time.time()-t0, len(docs)
        )
        return docs

    def get_quick_questions(self, count=6):
        """获取快捷问题列表。"""
        return self._quick_questions[:count]

    def get_default_response(self):
        """获取默认兜底回答。"""
        return self._default_response


def create_rag_retriever():
    """创建 ChromaRetriever 实例。"""
    return ChromaRetriever()
