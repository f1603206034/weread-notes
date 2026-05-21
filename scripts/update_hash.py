"""为本地 JSON 文件添加 contentHash 字段

遍历 data/ 目录下所有书籍 JSON 文件，调用微信读书 API 重新拉取数据，
计算 contentHash 并更新到本地文件中。

用法:
    export WEREAD_API_KEY=your_api_key
    python scripts/update_hash.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from api import WeReadClient
from config import load_env, get_data_dir
from sync import fetch_book_data, get_book_file_path
from utils import atomic_write_json, load_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("update_hash")


def update_local_json_files():
    """遍历本地 JSON 文件，重新拉取数据并更新 contentHash"""
    load_env()
    client = WeReadClient()
    data_dir = get_data_dir()

    if not data_dir.exists():
        logger.error("数据目录不存在: %s", data_dir)
        return

    json_files = list(data_dir.rglob("*.json"))
    total = len(json_files)
    updated = 0
    skipped = 0
    failed = 0

    logger.info("发现 %d 个 JSON 文件", total)

    for idx, json_path in enumerate(json_files, 1):
        book_id = json_path.stem
        logger.info("[%d/%d] 处理书籍: %s", idx, total, book_id)

        try:
            book_data = fetch_book_data(client, book_id)
            atomic_write_json(json_path, book_data)
            logger.info("已更新: %s (contentHash=%s)", json_path, book_data["meta"]["contentHash"])
            updated += 1
        except Exception as e:
            logger.error("处理失败 [%s]: %s", book_id, e)
            failed += 1

    logger.info("处理完成: 更新 %d, 跳过 %d, 失败 %d", updated, skipped, failed)


if __name__ == "__main__":
    update_local_json_files()
