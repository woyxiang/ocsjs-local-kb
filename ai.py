import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ─── 调试日志 ────────────────────────────────────────────────────────────────

def _setup_logger() -> logging.Logger:
    log_path = Path(__file__).parent / "logs" / "llm_debug.log"
    log_path.parent.mkdir(exist_ok=True)

    logger = logging.getLogger("llm_debug")
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        # 同时输出到 uvicorn 终端
        sh = logging.StreamHandler()
        sh.setLevel(logging.DEBUG)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    return logger


_llm_logger = _setup_logger()


@dataclass
class AIConfig:
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    # 知识库模式：OFF | SEARCH（默认）
    kb_mode: str = os.getenv("OPENAI_KB_MODE", "SEARCH").upper()

    @property
    def is_minimax(self) -> bool:
        return "minimax" in self.model.lower()


class AIGenerator:
    def __init__(self, config: AIConfig | None = None):
        self.config = config or AIConfig()
        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )

    def build_prompt(
        self,
        question: str,
        options: str | None = None,
        qtype: Literal["single", "multiple", "judgement", "completion", "unknown"]
        | None = None,
    ) -> str:
        parts = [f"请回答以下问题：\n{question}"]

        if options:
            parts.append(f"\n选项如下：\n{options}")

        format_rules = {
            "single": "【输出格式】只输出选项字母，如 A、B、C、D，不要输出任何其他内容。",
            "multiple": "【输出格式】只输出选项字母组合，如 AC、AD、BCD，不要输出任何其他内容。",
            "judgement": "【输出格式】只输出'对'或'错'，不要输出任何其他内容。",
            "completion": "【输出格式】只输出答案，不要有任何解释。",
        }
        if qtype in format_rules:
            parts.append("\n" + format_rules[qtype])
        elif qtype == "unknown":
            parts.append("\n\n请直接给出答案，无需解释。")

        return "\n".join(parts)

    def build_system_prompt(self, for_off_mode: bool = False) -> str:
        if for_off_mode:
            return (
                "你是一个专业的知识库问答助手。\n"
                "重要规则：\n"
                "1. 直接给出准确答案，用你自己的语言组织，不要废话。\n"
                "2. 如果不确定答案，说明'无法确定'。"
            )
        return (
            "你是一个专业的知识库问答助手。\n"
            "重要规则：\n"
            "1. 答案必须是你自己组织的语言，结合教材内容给出准确回答，"
            "绝对不能直接输出或复制教材原文。\n"
            "2. 如果教材中没有相关内容，请基于你的专业知识回答，"
            "但需说明'未在教材中找到相关内容'。"
        )

    @staticmethod
    def _clean_answer(content: str, qtype: str | None = None) -> str:
        """
        严格提取答案：
        1. 去掉思考标签
        2. 按题型强制提纯
        """
        rn_open = chr(0x3C) + chr(0x74) + chr(0x68) + chr(0x69) + chr(0x6E) + chr(0x6B) + chr(0x3E)
        rn_close = chr(0x3C) + chr(0x2F) + chr(0x74) + chr(0x68) + chr(0x69) + chr(0x6E) + chr(0x6B) + chr(0x3E)
        cleaned = re.sub(r"```[\s\S]*?```", "", content, flags=re.DOTALL)
        cleaned = re.sub(re.escape(rn_open) + r"[\s\S]*?" + re.escape(rn_close), "", cleaned, flags=re.DOTALL)
        cleaned = cleaned.strip()

        # 选择/判断题：直接提纯字母，不允许解释存在
        if qtype in ("single", "multiple"):
            # \b 对中文无效，改用字符类直接匹配
            letters = re.findall(r"[A-D]", cleaned)
            if letters:
                return "".join(letters)
            # 备选：搜 ABCD 这样连续的字串
            m = re.search(r"[A-D]{1,4}", cleaned)
            if m:
                return m.group(0)

        if qtype == "judgement":
            if "对" in cleaned or "错" in cleaned:
                # 提取第一个对/错
                m = re.search(r"[对错]", cleaned)
                return m.group(0) if m else cleaned

        return cleaned

    def _do_generate_off(self, question: str, options: str | None, qtype: str | None) -> dict[str, str]:
        """OFF 模式：不使用知识库，直接回答。"""
        messages = [
            {"role": "system", "content": self.build_system_prompt(for_off_mode=True)},
            {"role": "user", "content": self.build_prompt(question, options, qtype)},
        ]
        extra_body = {"reasoning_split": True} if self.config.is_minimax else None
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=0.3,
            extra_body=extra_body,
        )
        msg = response.choices[0].message
        raw = msg.content or ""

        _llm_logger.debug("===== OFF 模式 =====")
        _llm_logger.debug(f"问题: {question[:100]}")
        _llm_logger.debug(f"原始输出:\n{raw}")
        _llm_logger.debug(f"完整response:\n{response.model_dump_json(indent=2)}")

        return {"answer": self._clean_answer(raw, qtype)}

    def _do_generate_search(self, question: str, options: str | None, qtype: str | None) -> dict[str, str]:
        """
        SEARCH 模式（简化 RAG）：
        1. kb.search() 搜索相关章节
        2. 章节全文直接注入 user 消息
        3. 单轮 API 调用，AI 直接生成答案
        无 function calling，避免模型退化问题。
        """
        from knowledge_base import get_kb
        from tools import _format_chapters_for_context, _do_kb_search_chapters

        # 搜索相关章节
        kb_results = _do_kb_search_chapters(question, top_k=5)
        chapters_text = _format_chapters_for_context(kb_results)

        system_prompt = (
            "你是一个专业的知识库问答助手。\n"
            "重要规则：\n"
            "1. 必须基于【相关章节内容】回答，不要复制原文，用你自己的语言组织答案。\n"
            "2. 如果章节内容与问题不相关或不足以回答，请基于你的专业知识回答，"
            "但需说明'未在教材中找到相关内容'。\n"
            "3. 绝对不能直接输出章节内容，只能引用其中信息组织回答。"
        )

        user_prompt = (
            "【相关章节内容】\n\n"
            + chapters_text
            + "\n\n"
            + self.build_prompt(question, options, qtype)
        )

        extra_body = {"reasoning_split": True} if self.config.is_minimax else None
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            extra_body=extra_body,
        )
        msg = response.choices[0].message
        raw = msg.content or ""

        _llm_logger.debug("===== SEARCH 模式 =====")
        _llm_logger.debug(f"问题: {question[:100]}")
        _llm_logger.debug(f"注入章节数: {len(kb_results)}")
        _llm_logger.debug(f"原始输出:\n{raw}")
        _llm_logger.debug(f"完整response:\n{response.model_dump_json(indent=2)}")

        return {"answer": self._clean_answer(raw, qtype)}

    def generate(
        self, question: str, options: str | None = None, qtype: str | None = None
    ) -> dict[str, str]:
        if self.config.kb_mode == "OFF":
            return self._do_generate_off(question, options, qtype)
        return self._do_generate_search(question, options, qtype)
