"""Notion 推送模块 — 将书籍数据推送至 Notion

提供两种使用方式：
1. push_single_book() — 单本书推送（供 sync.py 循环内调用）
2. push_to_notion()     — 批量推送（遍历 index.json，可独立运行）

推送策略：
- 通过 bookId 查询 Notion 数据库
- 不存在 → 创建新页面（全量写入）
- 存在 → 删除旧 blocks + 全量更新（不做时间判断）

用法：
  python scripts/notion_push.py
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import load_env, get_data_dir, get_index_path
from utils import load_json
from notion_client import NotionClient
from md_to_notion import md_to_blocks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("notion_push")

# 模块级 Notion 客户端（延迟初始化）
_notion_client: NotionClient | None = None


def _get_notion_client() -> NotionClient:
    """获取或初始化 Notion 客户端（单例）"""
    global _notion_client
    if _notion_client is None:
        load_env()
        _notion_client = NotionClient()
    return _notion_client


def build_page_properties(book_data: dict) -> dict:
    """构建 Notion 数据库页面属性

    Args:
        book_data: 书籍 JSON 数据

    Returns:
        Notion properties dict
    """
    meta = book_data.get("meta", {})
    title = meta.get("title", "")
    book_id = meta.get("bookId", "")
    author = meta.get("author", "")
    translator = meta.get("translator", "")
    category = meta.get("category", "")
    publisher = meta.get("publisher", "")
    cover = meta.get("cover", "")
    reading_progress = meta.get("readingProgress", "")
    note_count = meta.get("noteCount", 0)
    review_count = meta.get("reviewCount", 0)
    bookmark_count = meta.get("bookmarkCount", 0)
    total_notes = note_count + review_count + bookmark_count
    app_link = meta.get("appLink", "")

    # 阅读进度映射
    progress_val = meta.get("readingProgress", "")
    if progress_val == "100%":
        progress_select = "已读完"
    elif progress_val == "0%":
        progress_select = "未读"
    else:
        progress_select = "在读"

    properties = {
        "书名": {"title": [{"text": {"content": title}}]},
        "bookId": {"rich_text": [{"text": {"content": book_id}}]},
        "阅读进度": {"select": {"name": progress_select}},
        "笔记数": {"number": total_notes},
    }

    # 仅在非空时添加属性
    if author:
        properties["作者"] = {"rich_text": [{"text": {"content": author}}]}
    if translator:
        properties["译者"] = {"rich_text": [{"text": {"content": translator}}]}
    if publisher:
        properties["出版社"] = {"rich_text": [{"text": {"content": publisher}}]}
    if category:
        properties["分类"] = {"select": {"name": category}}

    # 清理 None 值
    properties = {k: v for k, v in properties.items() if v is not None}

    # 封面（仅在有有效 URL 时添加）
    if cover and cover.startswith("http"):
        properties["封面"] = {
            "files": [{"name": f"{title}_cover", "external": {"url": cover}}]
        }

    # App 链接（仅作为属性，不写入页面内容）
    if app_link:
        properties["App链接"] = {"url": app_link}

    return properties


def push_single_book(client: NotionClient, book_data: dict, json_path: Path) -> bool:
    """推送单本书到 Notion（直接覆盖）

    通过 bookId 查询 Notion 数据库：
    - 不存在 → 创建新页面（全量写入）
    - 存在 → 删除旧 blocks + 全量更新

    Args:
        client: Notion 客户端
        book_data: 书籍 JSON 数据
        json_path: JSON 文件路径（用于定位 .md 文件）

    Returns:
        是否成功
    """
    meta = book_data.get("meta", {})
    book_id = meta.get("bookId", "")
    title = meta.get("title", f"未知_{book_id}")

    try:
        # 查询 Notion 是否已存在
        page = client.find_page_by_book_id(book_id)

        properties = build_page_properties(book_data)

        # 读取 Markdown 并转换为 blocks
        md_path = json_path.with_suffix(".md")
        children = []
        if md_path.exists():
            with open(md_path, "r", encoding="utf-8") as f:
                md_content = f.read()
            children = md_to_blocks(md_content)

        if page is None:
            # 不存在 → 创建新页面
            first_batch = children[:100] if len(children) > 100 else children
            client.create_page(properties, children=first_batch)

            # 追加剩余 blocks
            if len(children) > 100:
                page_id = None  # create_page 返回的 id
                # 重新查询获取 page_id
                new_page = client.find_page_by_book_id(book_id)
                if new_page:
                    for i in range(100, len(children), 100):
                        batch = children[i:i + 100]
                        client.append_blocks(new_page["id"], batch)

            logger.info("Notion 新建: %s", title)
        else:
            # 存在 → 更新属性 + 全量替换内容
            page_id = page["id"]
            client.update_page_properties(page_id, properties)

            if children:
                # 删除旧 blocks
                try:
                    existing_blocks = client.get_page_blocks(page_id)
                    for block in existing_blocks:
                        block_id = block.get("id")
                        if block_id:
                            try:
                                client._request("DELETE", f"/blocks/{block_id}")
                            except Exception as e:
                                logger.debug("删除 block %s 失败: %s", block_id, e)
                except Exception as e:
                    logger.debug("获取页面 blocks 失败: %s", e)

                # 添加新 blocks（分批）
                for i in range(0, len(children), 100):
                    batch = children[i:i + 100]
                    client.append_blocks(page_id, batch)

            logger.info("Notion 更新: %s", title)

        return True

    except Exception as e:
        logger.error("Notion 推送失败: %s - %s", title, e)
        return False


def push_to_notion(book_ids: list[str] | None = None):
    """批量推送书籍到 Notion

    Args:
        book_ids: 指定推送的书籍 ID 列表。为 None 时推送 index.json 中所有书籍。
    """
    client = _get_notion_client()

    # 加载索引
    index_path = get_index_path()
    index = load_json(index_path)
    if not index:
        logger.info("无索引数据，跳过 Notion 推送")
        return

    books = index.get("books", {})
    if not books:
        logger.info("索引中无书籍数据，跳过 Notion 推送")
        return

    # 确定推送范围
    if book_ids:
        target_books = {bid: books[bid] for bid in book_ids if bid in books}
    else:
        target_books = books

    data_dir = get_data_dir()
    created = 0
    updated = 0
    failed = 0

    for book_id, book_info in target_books.items():
        title = book_info.get("title", f"未知_{book_id}")

        # 加载本地 JSON
        json_path = data_dir / book_info.get("path", "")
        if not json_path.exists():
            found = list(data_dir.rglob(f"{book_id}.json"))
            if found:
                json_path = found[0]
            else:
                logger.warning("找不到 JSON 文件: %s", book_id)
                failed += 1
                continue

        book_data = load_json(json_path)
        if not book_data:
            logger.warning("JSON 文件为空: %s", json_path)
            failed += 1
            continue

        # 判断是新建还是更新
        try:
            page = client.find_page_by_book_id(book_id)
            is_new = page is None
        except Exception:
            is_new = True

        success = push_single_book(client, book_data, json_path)
        if success:
            if is_new:
                created += 1
            else:
                updated += 1
        else:
            failed += 1

    logger.info(
        "Notion 推送完成: 新建 %d, 更新 %d, 失败 %d",
        created, updated, failed,
    )


if __name__ == "__main__":
    push_to_notion()
