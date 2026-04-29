# Changelog

## v0.2 (2026-04-29)

**Features**

- embedding 缓存改为 npz 压缩格式，压缩率约 70%
- json 格式保留作为 fallback，兼容现有缓存
- embedding API 调用添加 timeout（批量 120s，单次查询 60s）

---

## v0.1 (2026-04-18)

- Initial release
- 支持 `SEARCH` / `OFF` 两种知识库模式
- 兼容 `tk.enncy.cn` API 格式
- 支持任意 OpenAI 兼容 API
