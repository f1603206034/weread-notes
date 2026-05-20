"""更新 Notion 页面图标 - 使用书籍封面作为图标

用法：
  python scripts/update_notion_icons.py

此脚本会：
1. 从 Notion 数据库获取所有书籍页面
2. 获取每个书籍的 bookId 和当前图标状态
3. 从本地 JSON 或微信读书 API 获取封面 URL
4. 更新 Notion 页面的图标
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import load_env, get_data_dir, get_index_path
from utils import load_json
from notion_client import NotionClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("update_icons")


def get_cover_from_local(book_id: str) -> str | None:
    """从本地 JSON 文件获取封面 URL"""
    data_dir = get_data_dir()

    json_files = list(Path(data_dir).rglob(f"{book_id}.json"))
    if json_files:
        json_path = json_files[0]
        data = load_json(json_path)
        if data:
            cover = data.get("meta", {}).get("cover", "")
            if cover and cover.startswith("http"):
                return cover

    return None


def get_cover_from_weread(book_id: str) -> str | None:
    """从微信读书 API 获取封面 URL"""
    import os
    from urllib.parse import urlencode

    api_key = os.getenv("WEREAD_API_KEY")
    if not api_key:
        logger.warning("WEREAD_API_KEY 未设置，无法从 API 获取封面")
        return None

    base_url = "https://weread.qq.com/wrbook detail"
    headers = {
        "Cookie": f"wr_sk={api_key}",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }

    try:
        import requests
        url = f"{base_url}?bookId={book_id}"
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.ok:
            data = resp.json()
            book = data.get("data", {}).get("book", {})
            cover = book.get("cover", "")
            if cover:
                return cover
    except Exception as e:
        logger.debug("从微信读书 API 获取封面失败: %s", e)

    return None


def _build_icon(cover_url: str) -> dict | None:
    """构建 Notion 图标对象"""
    if cover_url and cover_url.startswith("http"):
        return {"type": "external", "external": {"url": cover_url}}
    return None


def update_page_icon(client: NotionClient, page_id: str, book_id: str, title: str) -> bool:
    """更新单个页面的图标"""
    cover_url = None

    cover_url = get_cover_from_local(book_id)

    if not cover_url:
        cover_url = get_cover_from_weread(book_id)

    if not cover_url:
        logger.warning("无法获取封面 [%s]: %s", book_id, title)
        return False

    icon = _build_icon(cover_url)
    if not icon:
        return False

    try:
        client._request("PATCH", f"/pages/{page_id}", json={"icon": icon})
        logger.info("更新图标成功: %s", title)
        return True
    except Exception as e:
        logger.error("更新图标失败 [%s]: %s - %s", book_id, title, e)
        return False


def main():
    load_env()

    client = NotionClient()

    logger.info("正在查询 Notion 数据库...")
    all_pages = []
    cursor = None

    while True:
        result = client.query_database(page_size=100, start_cursor=cursor)
        pages = result.get("results", [])
        all_pages.extend(pages)

        if not result.get("has_more"):
            break
        cursor = result.get("next_cursor")

    logger.info("共获取 %d 个页面", len(all_pages))

    updated = 0
    skipped = 0
    failed = 0

    for page in all_pages:
        page_id = page.get("id", "")
        title = "未知"
        book_id = ""

        props = page.get("properties", {})

        title_prop = props.get("书名", {})
        if title_prop.get("title"):
            title = title_prop["title"][0].get("plain_text", "未知")

        book_id_prop = props.get("bookId", {})
        if book_id_prop.get("rich_text"):
            book_id = book_id_prop["rich_text"][0].get("plain_text", "")

        if not book_id:
            logger.debug("跳过无 bookId 的页面: %s", title)
            skipped += 1
            continue

        has_icon = page.get("icon") is not None
        if has_icon:
            logger.debug("跳过已有图标的页面: %s", title)
            skipped += 1
            continue

        success = update_page_icon(client, page_id, book_id, title)
        if success:
            updated += 1
        else:
            failed += 1

    logger.info("完成！更新: %d, 跳过: %d, 失败: %d", updated, skipped, failed)


if __name__ == "__main__":
    main()
