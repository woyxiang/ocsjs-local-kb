import math
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from ai import AIGenerator
from tools import get_tool_schemas


# ─── Request/Response models ───────────────────────────────────────────────

class QueryParams(BaseModel):
    token: str
    title: str | None = None
    q: str | None = None
    question: str | None = None
    options: str | None = None
    type: str | None = None
    more: bool = False

    def question_text(self) -> str:
        return self.title or self.q or self.question or ""


class QueryResponse(BaseModel):
    code: int
    data: dict
    message: str


class InfoResponse(BaseModel):
    code: int
    data: dict
    message: str


# ─── App lifecycle ─────────────────────────────────────────────────────────

ai_gen: AIGenerator | None = None
request_count: int = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ai_gen
    ai_gen = AIGenerator()
    yield
    ai_gen = None


app = FastAPI(title="AI 题库", lifespan=lifespan)


# ─── Routes ────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"status": "ok"}


@app.head("/")
async def root_head():
    return Response(status_code=200)


@app.get("/tools")
async def list_tools():
    """返回当前 AI 可调用的工具列表（OpenAI function calling schema）。"""
    return JSONResponse(content={
        "code": 1,
        "data": {"tools": get_tool_schemas()},
        "message": "请求成功",
    })


@app.get("/query")
async def query(
    token: str = Query(..., description="用户凭证"),
    title: str | None = Query(None, description="题目"),
    q: str | None = Query(None, description="题目（别名）"),
    question: str | None = Query(None, description="题目（别名）"),
    options: str | None = Query(None, description="选项"),
    type: str | None = Query(None, description="题目类型"),
    more: bool = Query(False, description="是否返回多个结果（暂不支持）"),
) -> JSONResponse:
    global request_count
    request_count += 1

    question_text = title or q or question
    if not question_text:
        return JSONResponse(
            status_code=400,
            content={"code": 0, "data": {"question": "", "answer": "", "times": -1}, "message": "题目不能为空"},
        )

    if not ai_gen:
        return JSONResponse(
            status_code=500,
            content={"code": 0, "data": {"question": question_text, "answer": "", "times": -1}, "message": "AI 服务未初始化"},
        )

    try:
        result = ai_gen.generate(question_text, options, type)
        return JSONResponse(content={
            "code": 1,
            "data": {
                "question": question_text,
                "answer": result["answer"],
                "ai": True,
                "times": -1,
            },
            "message": "请求成功",
        })
    except Exception as e:
        return JSONResponse(
            status_code=200,
            content={
                "code": 0,
                "data": {"question": question_text, "answer": "", "times": -1},
                "message": f"AI 生成失败：{e}",
            },
        )


@app.get("/info")
async def info(token: str = Query(..., description="用户凭证")) -> JSONResponse:
    global request_count
    return JSONResponse(content={
        "code": 1,
        "data": {
            "times": -1,
            "user_times": request_count,
            "success_times": request_count,
        },
        "message": "请求成功",
    })
