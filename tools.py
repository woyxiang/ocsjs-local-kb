"""
OpenAI Function Calling 工具定义。

暴露给 AI 的工具列表，以及工具执行入口。
"""

from typing import Any
from dataclasses import dataclass


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]


def get_tool_schemas() -> list[dict[str, Any]]:
    """返回 OpenAI function calling 所需的 tools 参数。"""
    return [TOOLS["kb_search_chapters"].to_openai_schema()]


def get_tool(name: str) -> "ToolImpl":
    """按名称查找工具。"""
    t = TOOLS.get(name)
    if t is None:
        raise ValueError(f"未知工具: {name}")
    return t


# ─── 工具实现 ───────────────────────────────────────────────────────────────


def _do_kb_search_chapters(query: str, top_k: int = 5) -> dict[str, Any]:
    """
    搜索教材章节，返回与问题最相关的章节内容（全文，不截断内容）。
    """
    from knowledge_base import get_kb

    kb = get_kb()
    results = kb.search(query, top_k=top_k)
    return {
        "query": query,
        "count": len(results),
        "results": results,
    }


def _format_chapters_for_context(data: dict[str, Any]) -> str:
    """
    将章节列表格式化为易读的文本，供注入上下文使用。
    每个章节包含标题、路径和完整正文。
    """
    if not data["results"]:
        return "（未找到相关章节内容）"

    lines = []
    for i, r in enumerate(data["results"], 1):
        lines.append(f"=== 第{i}章 ===")
        lines.append(f"标题：{r['title']}")
        lines.append(f"位置：{r['path']}")
        lines.append(f"内容：{r['content']}")
        lines.append("")
    return "\n".join(lines)


# ─── 工具注册表 ─────────────────────────────────────────────────────────────


class ToolImpl:
    """可执行的工具实现。"""

    def __init__(self, name: str, description: str, parameters: dict[str, Any]):
        self.name = name
        self.description = description
        self.parameters = parameters

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def execute(self, arguments: dict[str, Any]) -> str:
        if self.name == "kb_search_chapters":
            return _format_chapters_for_context(_do_kb_search_chapters(**arguments))
        raise ValueError(f"未实现的工具: {self.name}")


TOOLS: dict[str, ToolImpl] = {
    "kb_search_chapters": ToolImpl(
        name="kb_search_chapters",
        description=(
            "搜索教材章节，返回与问题最相关的章节全文内容。\n"
            "适用场景：所有需要引用教材内容来回答的问题。\n"
            "返回结果包含章节标题、路径和完整正文（可能很长），"
            "AI 应基于这些内容用自己的语言回答，不要复制原文。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词，从用户问题中提取核心术语",
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回的相关章节数量，默认 5",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    ),
}
