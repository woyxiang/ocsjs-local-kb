# 使用说明

## 快速启动

```bash
# 安装依赖
uv sync

# 配置 API Key
cp .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY

# 启动服务
uvicorn main:app --reload --port 8000
```

## 知识库模式

通过 `OPENAI_KB_MODE` 环境变量切换：

| 模式 | 值 | 说明 |
|------|---|------|
| 不用知识库 | `OFF` | AI 直接用自己的知识回答，无知识库依赖，最快 |
| 搜索模式 | `SEARCH`（默认） | AI 搜索相关章节，结合章节内容回答 |

```bash
# 关闭知识库
echo "OPENAI_KB_MODE=OFF" >> .env
```

## SEARCH 模式原理

```
用户提问
    ↓
AI 调用 kb_search_chapters 工具
    ↓
知识库.search() 返回 top-5 相关章节（BM25 + jieba 分词）
    ↓
章节全文（标题 + 路径 + 内容，最多 5×4000 字）注入上下文
    ↓
AI 基于上下文用自己的语言回答
```

**关键改进：** 章节全文直接进上下文，不依赖 AI 自行综合搜索结果，减少回答格式问题。

## API 接口

### `GET /query` — AI 答题

**参数：**

| 参数 | 必填 | 说明 |
|------|------|------|
| `token` | 是 | 用户凭证 |
| `title` / `q` / `question` | 是 | 题目（三选一） |
| `options` | 否 | 选项内容 |
| `type` | 否 | 题目类型：`single` / `multiple` / `judgement` / `completion` |

**题型与 answer 格式：**

| 题型 | type 参数 | answer 示例 |
|------|-----------|-------------|
| 单选 | `single` | `"A"` |
| 多选 | `multiple` | `"AC"` |
| 判断 | `judgement` | `"对"` / `"错"` |
| 填空 | `completion` | `"集总热容法"` |
| 问答/计算 | 其他 | 完整回答/公式 |

### `GET /tools` — 查看可用工具

```bash
curl http://localhost:8000/tools
```

当前可用工具：`kb_search_chapters`（搜索教材章节全文）

### `GET /info` — 统计数据

```bash
curl "http://localhost:8000/info?token=test"
```

## 知识库管理

教材文件放在 `knowledge_base/` 目录下：

```
knowledge_base/
└── *.md   ← 放置 markdown 教材
```

**检索算法：** BM25 + jieba 中文分词，按 title/path/content 加权打分，返回 top-5 相关章节。

### 向量索引缓存（可选）

当 `EMBEDDING_BASE_URL` 配置后，首次加载知识库会自动生成 embedding 缓存（`knowledge_base/*.md.embeddings.json`），后续请求直接加载缓存，无需重复调用 embedding 接口。

**预生成缓存（离线）**

```bash
# 启动服务前先生成缓存，--workers 并发数可调
uv run tools/build_embeddings.py --workers 4
```

**检索算法（混合）：** BM25（40%）+ Embedding 语义相似度（60%），无 embedding 时只用 BM25。

## 环境变量

```bash
OPENAI_API_KEY=your_api_key_here      # API 密钥
OPENAI_MODEL=gpt-4o-mini              # 模型名称（含 "minimax" 则启用 reasoning_split）
OPENAI_BASE_URL=https://api.openai.com/v1  # API 地址（支持 OpenAI 兼容的第三方接口）
OPENAI_KB_MODE=SEARCH                # 知识库模式：OFF | SEARCH
EMBEDDING_BASE_URL=                   # Embedding API 地址（不填则只用 BM25，不生成向量索引）
EMBEDDING_MODEL=text-embedding-3-small  # Embedding 模型
```

## 调试日志

LLM 原始输出和思考过程会写入 `logs/llm_debug.log`，同时输出到 uvicorn 终端：

```bash
# 实时查看
tail -f logs/llm_debug.log
```

日志包含：模式（OFF/SEARCH）、问题、注入章节数（SEARCH）、原始 LLM 输出。

## 项目结构

```
tiku/
├── main.py              # FastAPI 入口，/query、/info、/tools 路由
├── ai.py                # AIGenerator，OFF/SEARCH 两种模式，含 LLM 调试日志
├── knowledge_base.py    # 知识库加载，BM25 + Embedding 混合检索
├── tools.py             # kb_search_chapters 工具定义与执行
├── docs/usage.md        # 本文档
├── tools/
│   └── build_embeddings.py  # 预生成 embedding 缓存脚本
├── knowledge_base/       # 知识库 markdown 教材目录
│   ├── *.md
│   └── *.md.embeddings.json  # embedding 缓存（自动生成）
├── logs/
│   └── llm_debug.log    # LLM 调试日志
└── .env                 # API Key（不提交到 git）
```

## 常见问题

**Q: 搜索结果不相关？**
A: BM25 对关键词敏感，可尝试精简问题核心词（如"光合作用"而非"请介绍一下光合作用是如何进行的"）。

**Q: OFF 模式如何使用？**
A: 设置 `OPENAI_KB_MODE=OFF`，AI 完全不访问知识库。
