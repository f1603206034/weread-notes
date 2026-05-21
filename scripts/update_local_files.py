"""本地文件更新脚本 — 拉取微信读书数据并更新本地 JSON/Markdown

用途：
  当代码更新引入新字段（如 contentHash）时，重新拉取所有书籍数据，
  更新本地 JSON 和 Markdown 文件，但不推送 Notion。

用法：
  export WEREAD_API_KEY=your_api_key
  python scripts/update_local_files.py
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from api import WeReadClient
from config import load_env, get_data_dir, get_index_path
from sync import fetch_book_data, save_book_data, load_index, save_index
from utils import extract_category, get_folder_name, load_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("update_local")


def update_local_files():
    """拉取所有书籍数据，更新本地 JSON 和 Markdown，不推送 Notion"""
    load_env()

    client = WeReadClient()
    index = load_index()

    notebooks = client.get_all_notebooks()
    logger.info("共 %d 本有笔记的书籍", len(notebooks))

    updated = 0
    failed = 0
    skipped = 0

    for nb in notebooks:
        book_id = nb.get("bookId", "")
        book_info = nb.get("book", {})
        title = book_info.get("title", f"未知书名_{book_id}")
        categories = book_info.get("categories", [])
        if categories and len(categories) > 0:
            category = categories[0].get("title", "")
        else:
            category = ""

        try:
            book_data = fetch_book_data(client, book_id)
            new_hash = book_data["meta"]["contentHash"]

            json_path = (
                get_data_dir()
                / extract_category(category)
                / get_folder_name(title, book_id)
                / f"{book_id}.json"
            )
            old_book_data = load_json(json_path)

            if old_book_data:
                old_hash = old_book_data.get("meta", {}).get("contentHash", "")
                if old_hash and old_hash == new_hash:
                    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                    book_dir_rel = Path(extract_category(category)) / get_folder_name(
                        title, book_id
                    )
                    index["books"][book_id] = {
                        "title": title,
                        "category": extract_category(category),
                        "path": str(book_dir_rel / f"{book_id}.json"),
                        "sort": nb.get("sort", 0),
                        "noteCount": nb.get("noteCount", 0),
                        "reviewCount": nb.get("reviewCount", 0),
                        "bookmarkCount": nb.get("bookmarkCount", 0),
                        "lastSync": now_utc,
                    }
                    index["lastGlobalSync"] = now_utc
                    save_index(index)
                    skipped += 1
                    logger.info("内容未变，跳过: %s", title)
                    continue

            save_book_data(book_data, title, category)

            now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            book_dir_rel = Path(extract_category(category)) / get_folder_name(
                title, book_id
            )
            index["books"][book_id] = {
                "title": title,
                "category": extract_category(category),
                "path": str(book_dir_rel / f"{book_id}.json"),
                "sort": nb.get("sort", 0),
                "noteCount": nb.get("noteCount", 0),
                "reviewCount": nb.get("reviewCount", 0),
                "bookmarkCount": nb.get("bookmarkCount", 0),
                "lastSync": now_utc,
            }
            index["lastGlobalSync"] = now_utc
            save_index(index)

            updated += 1
            logger.info("更新成功 [%d/%d]: %s", updated, len(notebooks), title)

        except Exception as e:
            failed += 1
            logger.error("更新失败: %s (%s) - %s", title, book_id, e)
            continue

    logger.info(
        "本地更新完成: 成功 %d, 失败 %d, 跳过 %d",
        updated,
        failed,
        skipped,
    )


if __name__ == "__main__":
    update_local_files()
