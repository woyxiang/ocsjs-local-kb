import os
import re
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


# ─── Embedding 客户端 ───────────────────────────────────────────────────────


def _get_embedding_client():
    """创建 embedding API 客户端，优先用环境变量配置。"""
    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("EMBEDDING_BASE_URL", "").rstrip("/")
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    if not base_url:
        return None, None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
        return client, model
    except Exception:
        return None, None


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """计算两个向量的 cosine similarity。"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b + 1e-9)


# ─── 分词器 ─────────────────────────────────────────────────────────────────


_STOPWORDS = {"的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都",
              "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你",
              "会", "着", "没有", "看", "好", "自己", "这", "如何", "怎么",
              "怎样", "为什么", "啥", "吗", "呢", "吧", "啊", "哦", "嗯"}


def _tokenize(text: str) -> list[str]:
    """中文分词：优先 jieba，无则用字符 bigram。"""
    text_clean = re.sub(r"[^\w]", " ", text)
    try:
        import jieba
        return [w for w in jieba.cut(text_clean) if len(w) > 1 and w not in _STOPWORDS]
    except ImportError:
        chars = [c for c in text_clean if c.strip()]
        bigrams = [text_clean[i:i+2] for i in range(len(text_clean) - 1)]
        return [b for b in bigrams if b not in _STOPWORDS]


# ─── BM25 ───────────────────────────────────────────────────────────────────


class BM25:
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.N = len(corpus)
        self.avgdl = sum(len(doc) for doc in corpus) / self.N if self.N else 0
        df: Counter = Counter()
        for doc in corpus:
            df.update(set(doc))
        self.df = dict(df)
        self.idf = {}
        for term, freq in self.df.items():
            self.idf[term] = math.log((self.N - freq + 0.5) / (freq + 0.5) + 1)

    def score(self, query: list[str], doc: list[str]) -> float:
        doc_tf = Counter(doc)
        score = 0.0
        doc_len = len(doc)
        for term in query:
            if term not in self.idf:
                continue
            tf = doc_tf.get(term, 0)
            if tf == 0:
                continue
            idf = self.idf[term]
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / (self.avgdl + 1e-5))
            score += idf * numerator / denominator
        return score


# ─── 章节结构 ─────────────────────────────────────────────────────────────


@dataclass
class Section:
    title: str
    level: int
    content: str
    path: str
    tokens: list[str] = field(default_factory=list)
    embedding: list[float] | None = field(default=None)


# ─── 知识库 ─────────────────────────────────────────────────────────────────


class KnowledgeBase:
    """
    加载 markdown 教材，支持 BM25 + embedding 混合检索。
    embedding 可选，有则用，无则只用 BM25。
    """

    def __init__(self, kb_path: str | Path | None = None,
                 embedding_api_key: str | None = None,
                 embedding_base_url: str | None = None,
                 embedding_model: str | None = None):
        self.sections: list[Section] = []
        self._bm25: BM25 | None = None
        self._emb_client = None
        self._emb_model = embedding_model or "text-embedding-3-small"

        if embedding_base_url:
            try:
                from openai import OpenAI
                self._emb_client = OpenAI(
                    api_key=embedding_api_key or os.getenv("OPENAI_API_KEY", ""),
                    base_url=embedding_base_url.rstrip("/"),
                )
            except Exception:
                self._emb_client = None

        if kb_path:
            self.load(kb_path)

    def load(self, kb_path: str | Path) -> None:
        path = Path(kb_path)
        text = path.read_text(encoding="utf-8")
        self.sections = self._parse(text)
        self._build_bm25_index()
        self._kb_path = path
        if self._emb_client:
            self._build_embedding_index()

    def _parse(self, text: str) -> list[Section]:
        lines = text.splitlines()
        sections: list[Section] = []
        heading_stack: list[str] = []
        level_stack: list[int] = []

        for i, line in enumerate(lines):
            m = re.match(r"^(#{1,4})\s+(.+)", line)
            if not m:
                continue
            level = len(m.group(1))
            title = m.group(2).strip()

            skip = ["目录", "前言", "参考文献", "习题", "复习思考题",
                    "OCR", "Images have been", '"filename"']
            if any(k in title for k in skip):
                continue

            while level_stack and level_stack[-1] >= level:
                heading_stack.pop()
                level_stack.pop()
            heading_stack.append(title)
            level_stack.append(level)
            path_str = " / ".join(heading_stack)

            content_lines: list[str] = []
            j = i + 1
            while j < len(lines):
                next_line = lines[j]
                m2 = re.match(r"^(#{1,4})\s+(.+)", next_line)
                if m2 and len(m2.group(1)) <= level:
                    break
                stripped = next_line.strip()
                if stripped and not stripped.startswith(("!", "图", "$", "\"")):
                    content_lines.append(stripped)
                j += 1

            content = "\n".join(content_lines).strip()
            if content:
                sections.append(Section(
                    title=title,
                    level=level,
                    content=content,
                    path=path_str,
                ))
        return sections

    def _build_bm25_index(self) -> None:
        for s in self.sections:
            s.tokens = _tokenize(s.title + " " + s.path + " " + s.content)
        corpus = [s.tokens for s in self.sections]
        self._bm25 = BM25(corpus)

    def _build_embedding_index(self) -> None:
        """批量计算所有章节的 embedding 向量（带缓存）。"""
        if not self._emb_client or not self.sections:
            return

        cache_path = self._kb_path.with_suffix(self._kb_path.suffix + ".embeddings.json")
        cached = self._load_embedding_cache(cache_path)
        if cached:
            emb_model, emb_hash, embeddings = cached
            if emb_model == self._emb_model and emb_hash == self._content_hash():
                print(f"[KB] embedding 缓存加载成功 ({len(embeddings)} 个章节)")
                for s, emb in zip(self.sections, embeddings):
                    s.embedding = emb
                return
            print("[KB] embedding 缓存已失效，重新计算...")

        # 分批：每批最多 32 个 text
        batch_size = 32
        texts = [s.title + "\n" + s.content for s in self.sections]
        embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                resp = self._emb_client.embeddings.create(
                    model=self._emb_model,
                    input=batch,
                    timeout=120.0,
                )
                embeddings.extend([item.embedding for item in resp.data])
            except Exception as e:
                print(f"[KB] embedding batch {i//batch_size} failed: {e}")
                # 降格：用 zero 向量
                dim = 1536
                embeddings.extend([[0.0] * dim] * len(batch))
            time.sleep(0.1)

        for s, emb in zip(self.sections, embeddings):
            s.embedding = emb

        self._save_embedding_cache(cache_path, embeddings)

    def _content_hash(self) -> str:
        """对所有章节内容算 hash，用于判断缓存是否过期。"""
        import hashlib
        h = hashlib.sha256()
        for s in self.sections:
            h.update((s.title + s.path + s.content).encode())
        return h.hexdigest()

    def _load_embedding_cache(self, cache_path: Path) -> tuple[str, str, list[list[float]]] | None:
        """尝试加载缓存。优先 npz 压缩格式，fallback 到 json。"""
        if not cache_path.exists():
            return None
        # 尝试 npz 压缩格式
        npz_path = cache_path.with_suffix(".embeddings.npz")
        if npz_path.exists():
            try:
                import numpy as np
                data = np.load(npz_path)
                embeddings = [data[f"emb_{i}"].tolist() for i in range(len(data.files))]
                return str(data["model"]), str(data["content_hash"]), embeddings
            except Exception:
                pass
        # fallback json 格式
        try:
            import json
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            return data["model"], data["content_hash"], data["embeddings"]
        except Exception:
            return None

    def _save_embedding_cache(self, cache_path: Path, embeddings: list[list[float]]) -> None:
        """保存 embeddings 到缓存文件（npz 压缩格式）。"""
        try:
            import numpy as np
            npz_path = cache_path.with_suffix(".embeddings.npz")
            arr = np.array(embeddings, dtype=np.float32)
            np.savez_compressed(
                npz_path,
                model=self._emb_model,
                content_hash=self._content_hash(),
                **{f"emb_{i}": arr[i] for i in range(len(embeddings))},
            )
            print(f"[KB] embedding 缓存已保存 ({len(embeddings)} 个章节, npz 压缩)")
        except Exception as e:
            print(f"[KB] embedding 缓存保存失败 (npz): {e}")
            # fallback json 格式
            try:
                import json
                cache_path.write_text(
                    json.dumps({
                        "model": self._emb_model,
                        "content_hash": self._content_hash(),
                        "embeddings": embeddings,
                    }, ensure_ascii=False),
                    encoding="utf-8",
                )
                print(f"[KB] embedding 缓存已保存 ({len(embeddings)} 个章节, json fallback)")
            except Exception as e2:
                print(f"[KB] embedding 缓存保存失败 (json): {e2}")

    def _search_by_embedding(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """纯 embedding 语义搜索。"""
        if not self._emb_client:
            return []
        if not self.sections or self.sections[0].embedding is None:
            return []

        try:
            resp = self._emb_client.embeddings.create(
                model=self._emb_model,
                input=[query],
                timeout=60.0,
            )
            q_emb = resp.data[0].embedding
        except Exception:
            return []

        scored = []
        for idx, section in enumerate(self.sections):
            if section.embedding is None:
                continue
            sim = _cosine_sim(q_emb, section.embedding)
            scored.append((idx, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        混合检索：embedding 语义相似度 + BM25 关键词。
        embedding 有则混合使用，无则只用 BM25。
        """
        if not self.sections:
            return []

        # BM25 打分
        bm25_scored: list[tuple[int, float]] = []
        if self._bm25:
            query_tokens = _tokenize(query)
            if not query_tokens:
                query_tokens = _tokenize(query)  # 不过滤

            for idx, section in enumerate(self.sections):
                bm25_score = self._bm25.score(query_tokens, section.tokens)
                title_hit = sum(1 for t in query_tokens if t in section.title.lower())
                path_hit = sum(1 for t in query_tokens if t in section.path.lower())
                bonus = title_hit * bm25_score * 0.8 + path_hit * bm25_score * 0.3
                total = bm25_score + bonus
                if total > 0:
                    bm25_scored.append((idx, total))

        # Embedding 打分
        emb_scored: list[tuple[int, float]] = []
        if self._emb_client and self.sections and self.sections[0].embedding is not None:
            emb_scored = self._search_by_embedding(query, top_k * 3)

        # 合并两个分数
        all_scores: dict[int, float] = {}
        # BM25 归一化
        if bm25_scored:
            max_bm25 = max(s for _, s in bm25_scored)
            if max_bm25 > 0:
                for idx, score in bm25_scored:
                    all_scores[idx] = all_scores.get(idx, 0) + score / max_bm25 * 0.4
        # Embedding 归一化
        if emb_scored:
            max_emb = max(s for _, s in emb_scored)
            if max_emb > 0:
                for idx, score in emb_scored:
                    all_scores[idx] = all_scores.get(idx, 0) + score / max_emb * 0.6

        # 无 embedding 时只用 BM25
        if not emb_scored and bm25_scored:
            sorted_bm25 = sorted(bm25_scored, key=lambda x: x[1], reverse=True)
            return [
                {
                    "title": self.sections[idx].title,
                    "path": self.sections[idx].path,
                    "level": self.sections[idx].level,
                    "content": self.sections[idx].content[:4000],
                }
                for idx, _ in sorted_bm25[:top_k]
            ]

        # 混合排序
        ranked = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        return [
            {
                "title": self.sections[idx].title,
                "path": self.sections[idx].path,
                "level": self.sections[idx].level,
                "content": self.sections[idx].content[:4000],
            }
            for idx, _ in ranked[:top_k]
        ]


# ─── 全局单例 ─────────────────────────────────────────────────────────────


_kb: KnowledgeBase | None = None


def get_kb() -> KnowledgeBase:
    global _kb
    if _kb is None:
        import pathlib
        kb_dir = pathlib.Path(__file__).parent / "knowledge_base"
        md_files = list(kb_dir.glob("*.md"))
        if not md_files:
            raise FileNotFoundError("未在 knowledge_base/ 目录下找到 markdown 文件")
        emb_base = os.getenv("EMBEDDING_BASE_URL", "")
        emb_key = os.getenv("OPENAI_API_KEY", "")
        emb_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        _kb = KnowledgeBase(
            max(md_files, key=lambda p: p.stat().st_size),
            embedding_api_key=emb_key,
            embedding_base_url=emb_base,
            embedding_model=emb_model,
        )
    return _kb
