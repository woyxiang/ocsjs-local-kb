#!/usr/bin/env python3
"""
预生成 embedding 缓存。

用法:
    python tools/build_embeddings.py
    python tools/build_embeddings.py --workers 4

会在 knowledge_base/*.md 同级目录生成 *.md.embeddings.json
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

KB_DIR = Path(__file__).parent.parent / "knowledge_base"


def get_client():
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("EMBEDDING_BASE_URL", "").rstrip("/")
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    if not base_url:
        print("错误: EMBEDDING_BASE_URL 未设置")
        return None, None

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)
    return client, model


def content_hash(sections: list[dict]) -> str:
    import hashlib
    h = hashlib.sha256()
    for s in sections:
        h.update((s["title"] + s["path"] + s["content"]).encode())
    return h.hexdigest()


def parse_md(text: str) -> list[dict]:
    import re
    lines = text.splitlines()
    sections = []
    heading_stack, level_stack = [], []

    skip = {"目录", "前言", "参考文献", "习题", "复习思考题", "OCR", "Images have been", '"filename"'}

    for i, line in enumerate(lines):
        m = re.match(r"^(#{1,4})\s+(.+)", line)
        if not m:
            continue
        level = len(m.group(1))
        title = m.group(2).strip()

        if any(k in title for k in skip):
            continue

        while level_stack and level_stack[-1] >= level:
            heading_stack.pop()
            level_stack.pop()
        heading_stack.append(title)
        level_stack.append(level)
        path_str = " / ".join(heading_stack)

        content_lines = []
        j = i + 1
        while j < len(lines):
            m2 = re.match(r"^(#{1,4})\s+(.+)", lines[j])
            if m2 and len(m2.group(1)) <= level:
                break
            stripped = lines[j].strip()
            if stripped and not stripped.startswith(("!", "图", "$", "\"")):
                content_lines.append(stripped)
            j += 1

        content = "\n".join(content_lines).strip()
        if content:
            sections.append({"title": title, "path": path_str, "content": content})
    return sections


def build_cache(md_path: Path, client, model):
    text = md_path.read_text(encoding="utf-8")
    sections = parse_md(text)

    cache_path = md_path.with_suffix(md_path.suffix + ".embeddings.json")

    # 检查现有缓存
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("model") == model and cached.get("content_hash") == content_hash(sections):
                print(f"  跳过 (缓存有效): {md_path.name}")
                return
        except Exception:
            pass

    # 生成 embeddings
    texts = [s["title"] + "\n" + s["content"] for s in sections]
    embeddings = []
    batch_size = 32

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            resp = client.embeddings.create(model=model, input=batch)
            embeddings.extend([item.embedding for item in resp.data])
            print(f"  批次 {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size} 完成")
        except Exception as e:
            print(f"  批次失败: {e}")
            dim = 1536
            embeddings.extend([[0.0] * dim] * len(batch))
        time.sleep(0.1)

    # 保存
    cache_path.write_text(
        json.dumps({
            "model": model,
            "content_hash": content_hash(sections),
            "embeddings": embeddings,
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  已保存: {cache_path.name} ({len(embeddings)} 个章节)")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="预生成 embedding 缓存")
    parser.add_argument("--workers", type=int, default=4, help="并发数 (默认: 4)")
    args = parser.parse_args()

    md_files = list(KB_DIR.glob("*.md"))
    if not md_files:
        print(f"未找到 md 文件: {KB_DIR}")
        return

    print(f"找到 {len(md_files)} 个 md 文件，并发数: {args.workers}")

    client, model = get_client()
    if not client:
        return

    # ThreadPoolExecutor 共享同一个 client（线程安全）
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(build_cache, md_path, client, model): md_path for md_path in md_files}
        for future in as_completed(futures):
            md_path = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"  {md_path.name} 处理失败: {e}")

    print("\n完成!")


if __name__ == "__main__":
    main()