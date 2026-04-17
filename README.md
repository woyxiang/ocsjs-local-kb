# ocsjs-local-kb

本地知识库 AI 题库 API 服务，兼容 [OCS 网课助手](https://github.com/ocsjs/ocsjs) 的 tk.enncy.cn 接口格式。

## 快速开始

```bash
# 安装依赖
uv sync

# 配置环境变量
cp .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY 等配置

# 启动服务
uvicorn main:app --reload --port 8000
```

## API 兼容

目标接口：`https://tk.enncy.cn`

| 接口 | 说明 |
|------|------|
| `GET /query` | AI 生成答案 |
| `GET /info` | 统计数据 |
| `GET /tools` | 可用工具列表 |

详细接口文档见 [docs/usage.md](docs/usage.md)。

## 许可证

MIT
