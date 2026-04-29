## 项目概述

AI 知识库问答代理，兼容 `tk.enncy.cn` API 格式。本地运行，调用 AI 生成答案而非查询数据库。

### 技术栈

- **FastAPI** — Web 框架
- **OpenAI SDK** — AI 调用（支持 OpenAI 及兼容 API 的第三方提供商）
- **python-dotenv** — 环境变量管理
- **uv** — 包管理器

### API 兼容性

目标 API：`https://tk.enncy.cn`

- `GET /query` — AI 生成答案
  - 参数：`token`（必填）、`title`/`q`/`question`（必填）、`options`、`type`、`more`
  - 返回：`{ code, data: { question, answer, ai: true, times }, message }`
- `GET /info` — 获取题库信息（本地统计）
  - 参数：`token`（必填）
  - 返回：`{ code, data: { times, user_times, success_times }, message }`

### 项目结构

```
tiku/
├── main.py              # FastAPI 入口，/query、/info、/tools 路由
├── ai.py                # AIGenerator，prompt 构造、function calling
├── kb.py                # 知识库加载与关键词检索
├── tools.py            # Function Calling 工具定义与执行
├── kb/                  # Markdown 格式教材目录
│   └── *.md
├── docs/usage.md        # 使用说明文档
├── apiDoc.html          # API 文档页面
├── pyproject.toml       # uv 项目配置
├── .env.example         # 环境变量模板
└── .env                 # API Key 配置（不提交到 git）
```

### 知识库模式（OPENAI_KB_MODE）

两种模式，通过环境变量切换：

- `SEARCH`（默认）— AI 调用 `kb_search_chapters` 工具搜索相关章节，章节全文注入上下文后生成答案
- `OFF` — 不使用知识库，AI 直接回答

SEARCH 流程：`kb_search_chapters` → top-5 章节全文（每章节最多 4000 字）→ 注入上下文 → AI 回答

- `kb.py` — BM25 + jieba 分词解析教材章节
- `tools.py` — `kb_search_chapters` 工具（SEARCH 模式注册）
- `ai.py` — `_do_generate_off` / `_do_generate_search`，`_clean_answer()` 兜底过滤思考内容
- MiniMax 模型自动启用 `reasoning_split=True`

### 注意事项

- `request_count` 存储在内存中，单 worker 模式下的统计数据准确
- 支持任意 OpenAI 兼容 API（通过 `OPENAI_BASE_URL` 配置）
- `OPENAI_MODEL` 含 `minimax` 时自动启用 reasoning_split
- `OPENAI_KB_MODE` 支持 `OFF` / `SEARCH`

### 启动命令

```bash
# 安装依赖
uv sync

# 配置 API Key（复制 .env.example 为 .env 后填入）
cp .env.example .env

# 启动服务（默认 localhost:8000）
uvicorn main:app --reload --port 8000
```

详细使用说明见 `docs/usage.md`。
