"""主同步脚本 — 微信读书笔记同步至 GitHub 仓库

支持模式：
  full         首次全量同步
  incremental  日常增量同步（默认）
  full-compare 全量比对（兜底）
  rebuild      手动重建目录结构

用法：
  python scripts/sync.py --mode full
  python scripts/sync.py --mode incremental
  python scripts/sync.py --mode full --resume
  python scripts/sync.py --mode rebuild
"""

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# 将 scripts 目录加入 Python 路径
sys.path.insert(0, str(Path(__file__).resolve().parent))

from api import WeReadClient, UpgradeRequiredError
from config import load_env, load_config, get_data_dir, get_index_path
from utils import (
    atomic_write_json,
    atomic_write_text,
    extract_category,
    get_folder_name,
    get_weread_web_url,
    get_weread_web_chapter_url,
    get_weread_web_bookmark_url,
    load_json,
    parse_range,
    save_json,
    seconds_to_reading_time,
    timestamp_to_str,
    timestamp_to_utc_str,
)
from renderer import render_markdown

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sync")


def _compute_content_hash(book_data: dict) -> str:
    """计算书籍内容哈希（仅包含想法/点评内容）

    用于检测单条想法/书评的文字修改（微信读书 API 无法直接检测此类变更）。
    只包含会渲染到 Markdown/Notion 的想法和书评内容，排除划线、章节、进度等。

    Args:
        book_data: 完整的书籍 JSON 数据

    Returns:
        SHA256 哈希值（64 位十六进制字符串）
    """
    # 提取所有想法内容（章节内的想法）
    reviews_data = []
    for chapter_content in book_data.get("content", []):
        for item in chapter_content.get("items", []):
            if item.get("type") == "review":
                reviews_data.append({
                    "content": item.get("content", ""),
                    "abstract": item.get("abstract", ""),
                    "star": item.get("star", -1),
                })

    # 提取本书评论（整本书评）
    book_reviews_data = []
    for rv in book_data.get("bookReviews", []):
        book_reviews_data.append({
            "content": rv.get("content", ""),
            "star": rv.get("star", -1),
            "isFinish": rv.get("isFinish", False),
        })

    # 构建用于哈希的数据结构
    hash_source = {
        "reviews": reviews_data,
        "bookReviews": book_reviews_data,
    }

    # 规范化 JSON 字符串（排序 key，确保一致性）
    json_str = json.dumps(hash_source, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


def get_book_file_path(book_id: str, title: str, category: str) -> Path:
    """获取书籍数据文件目录路径

    Args:
        book_id: 书籍 ID
        title: 书名
        category: 分类

    Returns:
        书籍数据目录路径
    """
    data_dir = get_data_dir()
    cat = extract_category(category)
    folder_name = get_folder_name(title, book_id)
    return data_dir / cat / folder_name


def fetch_book_data(client: WeReadClient, book_id: str) -> dict:
    """获取单本书的完整数据

    依次调用：book/info, chapterinfo, bookmarklist, review/list/mine, getprogress

    Args:
        client: API 客户端
        book_id: 书籍 ID

    Returns:
        完整的书籍 JSON 数据
    """
    logger.info("获取书籍数据: %s", book_id)

    # 1. 书籍基本信息
    info = client.get_book_info(book_id)
    book_info = info.get("book", info)

    # 2. 章节目录
    try:
        chapter_data = client.get_chapter_info(book_id)
        chapters = chapter_data.get("chapters", [])
    except Exception as e:
        logger.warning("获取章节目录失败 [%s]: %s", book_id, e)
        chapters = []

    # 3. 划线列表
    try:
        bookmark_data = client.get_bookmark_list(book_id)
        bookmarks = bookmark_data.get("updated", [])
        bookmark_chapters = bookmark_data.get("chapters", [])
    except Exception as e:
        logger.warning("获取划线列表失败 [%s]: %s", book_id, e)
        bookmarks = []
        bookmark_chapters = []

    # 4. 想法/点评（自动翻页）
    try:
        all_reviews_raw = client.get_all_my_reviews(book_id)
    except Exception as e:
        logger.warning("获取想法/点评失败 [%s]: %s", book_id, e)
        all_reviews_raw = []

    # 5. 阅读进度
    try:
        progress_data = client.get_book_progress(book_id)
        progress = progress_data.get("book", {})
    except Exception as e:
        logger.warning("获取阅读进度失败 [%s]: %s", book_id, e)
        progress = {}

    # 6. 热门划线（可选）
    try:
        hot_data = client.get_best_bookmarks(book_id)
        hot_bookmarks = hot_data.get("items", [])
    except Exception as e:
        logger.warning("获取热门划线失败 [%s]: %s", book_id, e)
        hot_bookmarks = []

    # ── 构建章节数据映射 ──────────────────────────────────
    chapter_map: dict[int, dict] = {}
    for ch in chapters:
        uid = ch.get("chapterUid", 0)
        chapter_map[uid] = ch

    # ── 处理划线数据 ──────────────────────────────────────
    highlights = []
    for bm in bookmarks:
        chapter_uid = bm.get("chapterUid", 0)
        range_str = bm.get("range", "")
        range_start, range_end = parse_range(range_str)

        highlights.append({
            "type": "highlight",
            "bookmarkId": bm.get("bookmarkId", ""),
            "range": range_str,
            "rangeStart": range_start,
            "rangeEnd": range_end,
            "markText": bm.get("markText", ""),
            "colorStyle": bm.get("colorStyle", 1),
            "colorLabel": "",
            "createTime": bm.get("createTime", 0),
            "createTimeFormatted": timestamp_to_str(bm.get("createTime", 0)),
            "chapterUid": chapter_uid,
            "appLink": (
                f"weread://bestbookmark?bookId={book_id}"
                f"&chapterUid={chapter_uid}"
                f"&rangeStart={range_start}&rangeEnd={range_end}"
                if range_start and range_end and chapter_uid else ""
            ),
            "webLink": (
                get_weread_web_bookmark_url(book_id, chapter_uid, range_start, range_end)
                if range_start and range_end and chapter_uid else ""
            ),
        })

    # ── 处理想法/点评数据 ──────────────────────────────────
    reviews = []
    book_reviews = []

    for rv_item in all_reviews_raw:
        rv = rv_item.get("review", rv_item)
        review_id = rv.get("reviewId", "")
        content = rv.get("content", "")
        chapter_name = rv.get("chapterName", "")
        star = rv.get("star", -1)
        create_time = rv.get("createTime", 0)
        is_finish = rv.get("isFinish", False)
        rv_chapter_uid = rv.get("chapterUid", 0)
        rv_range = rv.get("range", "")
        # 纯想法的原文摘要从 abstract 字段获取
        abstract = rv.get("abstract", "")

        review_entry = {
            "type": "review",
            "reviewId": review_id,
            "range": rv_range,
            "chapterName": chapter_name,
            "content": content,
            "abstract": abstract,  # 原文摘要
            "star": star,
            "createTime": create_time,
            "createTimeFormatted": timestamp_to_str(create_time),
            "isFinish": is_finish,
            "chapterUid": rv_chapter_uid,
        }

        # 区分书评和普通想法
        if not chapter_name and not rv_range:
            book_reviews.append(review_entry)
        else:
            reviews.append(review_entry)

    # ── 按章节归类内容 ──────────────────────────────────────
    content_by_chapter: dict[int, dict] = {}

    # 建立 range -> review 的映射（用于检测"既有划线又有想法"的情况）
    review_by_range: dict[str, dict] = {}
    for rv in reviews:
        rv_range = rv.get("range", "")
        if rv_range:
            review_by_range[rv_range] = rv

    # 处理划线：如果该 range 也有想法，则跳过（只保留想法）
    for hl in highlights:
        uid = hl.get("chapterUid", 0)
        hl_range = hl.get("range", "")
        
        # 如果该 range 也有想法，则跳过（只保留想法）
        if hl_range and hl_range in review_by_range:
            continue
        
        if uid not in content_by_chapter:
            ch_info = chapter_map.get(uid, {})
            content_by_chapter[uid] = {
                "chapterUid": uid,
                "chapterTitle": ch_info.get("title", ""),
                "items": [],
            }
        content_by_chapter[uid]["items"].append(hl)

    # 将想法按章节分组
    for rv in reviews:
        uid = rv.get("chapterUid", 0)
        if uid not in content_by_chapter:
            ch_info = chapter_map.get(uid, {})
            content_by_chapter[uid] = {
                "chapterUid": uid,
                "chapterTitle": ch_info.get("title", ""),
                "items": [],
            }
        content_by_chapter[uid]["items"].append(rv)

    # ── 处理热门划线 ──────────────────────────────────────
    hot_bookmarks_clean = []
    for hb in hot_bookmarks:
        hb_chapter_uid = hb.get("chapterUid", 0)
        hb_range = hb.get("range", "")
        hb_range_start, hb_range_end = parse_range(hb_range)
        hot_bookmarks_clean.append({
            "bookmarkId": hb.get("bookmarkId", ""),
            "chapterUid": hb_chapter_uid,
            "range": hb_range,
            "markText": hb.get("markText", ""),
            "totalCount": hb.get("totalCount", 0),
            "appLink": (
                f"weread://bestbookmark?bookId={book_id}"
                f"&chapterUid={hb_chapter_uid}"
                f"&rangeStart={hb_range_start}&rangeEnd={hb_range_end}"
                if hb_range_start and hb_range_end and hb_chapter_uid else ""
            ),
            "webLink": (
                get_weread_web_bookmark_url(book_id, hb_chapter_uid, hb_range_start, hb_range_end)
                if hb_range_start and hb_range_end and hb_chapter_uid else ""
            ),
        })

    # ── 处理进度数据 ──────────────────────────────────────
    progress_val = progress.get("progress", 0)
    reading_progress = f"{progress_val}%"
    record_reading_time = progress.get("recordReadingTime", 0)
    reading_time = seconds_to_reading_time(record_reading_time)
    finish_time = progress.get("finishTime", "")
    finished_date = ""
    if finish_time:
        try:
            # finishTime 可能是时间戳或字符串
            if isinstance(finish_time, (int, float)):
                finished_date = timestamp_to_str(int(finish_time), "%Y-%m-%d")
            else:
                finished_date = str(finish_time)[:10]
        except Exception:
            finished_date = str(finish_time)[:10] if finish_time else ""

    # ── 组装完整数据 ──────────────────────────────────────
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # 计算各类型数量
    # note_count 使用原始 bookmarks 数量，避免"既有划线又有想法"的过滤导致数量不一致
    note_count = len(bookmarks)
    review_count = len(reviews) + len(book_reviews)
    bookmark_count = 0  # 书签内容不可导出，只统计数量

    # 计算内容哈希（仅基于想法/点评内容，用于检测文字修改）
    content_hash = _compute_content_hash({
        "content": list(content_by_chapter.values()),
        "bookReviews": book_reviews,
    })

    book_data = {
        "meta": {
            "bookId": book_id,
            "title": book_info.get("title", ""),
            "author": book_info.get("author", ""),
            "translator": book_info.get("translator", ""),
            "cover": book_info.get("cover", ""),
            "intro": book_info.get("intro", ""),
            "category": book_info.get("category", ""),
            "publisher": book_info.get("publisher", ""),
            "publishTime": book_info.get("publishTime", ""),
            "isbn": book_info.get("isbn", ""),
            "wordCount": book_info.get("wordCount", 0),
            "newRating": book_info.get("newRating", 0),
            "newRatingCount": book_info.get("newRatingCount", 0),
            "newRatingDetail": book_info.get("newRatingDetail", {}),
            "appLink": f"weread://reading?bId={book_id}",
            "webLink": get_weread_web_url(book_id),
            "lastSync": now_utc,
            "readingProgress": reading_progress,
            "readingTime": reading_time,
            "finishedDate": finished_date,
            "noteCount": note_count,
            "reviewCount": review_count,
            "bookmarkCount": bookmark_count,
            "contentHash": content_hash,
        },
        "chapters": [
            {
                "chapterUid": ch.get("chapterUid", 0),
                "chapterIdx": ch.get("chapterIdx", 0),
                "title": ch.get("title", ""),
                "level": ch.get("level", 1),
                "wordCount": ch.get("wordCount", 0),
                "price": ch.get("price", 0),
                "paid": ch.get("paid", 0),
                "isMPChapter": ch.get("isMPChapter", 0),
                "anchors": ch.get("anchors", []),
                "appLink": (
                    f"weread://reading?bId={book_id}&chapterUid={ch.get('chapterUid', 0)}"
                ),
                "webLink": get_weread_web_chapter_url(book_id, ch.get("chapterUid", 0)),
            }
            for ch in chapters
        ],
        "content": list(content_by_chapter.values()),
        "bookReviews": book_reviews,
        "hotBookmarks": hot_bookmarks_clean,
        "readProgress": {
            "chapterUid": progress.get("chapterUid", 0),
            "chapterOffset": progress.get("chapterOffset", 0),
            "progress": progress_val,
            "updateTime": progress.get("updateTime", ""),
            "recordReadingTime": record_reading_time,
            "finishTime": finish_time,
        },
    }

    return book_data


def save_book_data(book_data: dict, title: str, category: str) -> Path:
    """保存书籍 JSON 和 Markdown 文件

    Args:
        book_data: 完整的书籍 JSON 数据
        title: 书名
        category: 分类

    Returns:
        JSON 文件路径
    """
    book_id = book_data["meta"]["bookId"]
    book_dir = get_book_file_path(book_id, title, category)

    # 保存 JSON（原子写入）
    json_path = book_dir / f"{book_id}.json"
    atomic_write_json(json_path, book_data)
    logger.info("JSON 已保存: %s", json_path)

    # 渲染并保存 Markdown（原子写入）
    md_content = render_markdown(book_data)
    md_path = book_dir / f"{book_id}.md"
    atomic_write_text(md_path, md_content)
    logger.info("Markdown 已保存: %s", md_path)

    return json_path


def load_index() -> dict:
    """加载全局索引"""
    index_path = get_index_path()
    index = load_json(index_path)
    if index is None:
        return {"lastGlobalSync": "", "books": {}}
    return index


def save_index(index: dict):
    """保存全局索引"""
    index_path = get_index_path()
    save_json(index_path, index)


def need_sync(book_id: str, remote_book: dict, local_index: dict) -> bool:
    """判断书籍是否需要同步（多重校验）

    Args:
        book_id: 书籍 ID
        remote_book: 远程笔记本条目
        local_index: 本地全局索引

    Returns:
        是否需要同步
    """
    local = local_index.get("books", {}).get(book_id)
    if not local:
        return True

    # 校验1：sort 值变化（最后操作时间戳）
    sort_changed = remote_book.get("sort", 0) != local.get("sort", 0)

    # 校验2：数量变化
    count_changed = (
        remote_book.get("noteCount", 0) != local.get("noteCount", 0)
        or remote_book.get("reviewCount", 0) != local.get("reviewCount", 0)
        or remote_book.get("bookmarkCount", 0) != local.get("bookmarkCount", 0)
    )

    # 校验3：本地 JSON 的 lastSync 与远程 sort 时间不一致
    # sort 是秒级时间戳，lastSync 是 ISO 格式，比较日期部分
    local_last_sync = local.get("lastSync", "")
    remote_sort = remote_book.get("sort", 0)
    sync_mismatch = False
    if local_last_sync and remote_sort:
        try:
            from datetime import datetime, timezone
            local_dt = datetime.fromisoformat(local_last_sync.replace("Z", "+00:00"))
            remote_dt = datetime.fromtimestamp(remote_sort, tz=timezone.utc)
            # 如果远程 sort 时间比本地 lastSync 新超过 60 秒，说明有变更
            if (remote_dt - local_dt).total_seconds() > 60:
                sync_mismatch = True
        except (ValueError, OSError):
            pass

    return sort_changed or count_changed or sync_mismatch


def sync_full(client: WeReadClient, resume: bool = False, force: bool = True):
    """全量同步

    强制重新拉取所有书籍数据，覆盖本地已有数据。
    用于数据完整性校验和修复不一致问题。

    Args:
        client: API 客户端
        resume: 是否断点续传（跳过已同步的）
        force: 是否强制覆盖（True=重新拉取所有书，False=同增量逻辑）
    """
    logger.info("开始全量同步 (force=%s)", force)
    index = load_index()

    # 获取所有有笔记的书籍
    notebooks = client.get_all_notebooks()
    logger.info("共 %d 本有笔记的书籍", len(notebooks))

    # 初始化 Notion 客户端
    from notion_push import push_single_book, _get_notion_client
    try:
        notion_client = _get_notion_client()
        logger.info("Notion 客户端已初始化")
    except Exception as e:
        logger.warning("Notion 初始化失败，将跳过 Notion 推送: %s", e)
        notion_client = None

    synced = 0
    failed = 0
    skipped = 0
    notion_ok = 0
    notion_fail = 0

    for nb in notebooks:
        book_id = nb.get("bookId", "")
        book_info = nb.get("book", {})
        title = book_info.get("title", f"未知书名_{book_id}")
        # 从 categories 数组获取分类标题
        categories = book_info.get("categories", [])
        if categories and len(categories) > 0:
            category = categories[0].get("title", "")
        else:
            category = ""

        # 断点续传：跳过已同步的
        if resume and book_id in index.get("books", {}):
            skipped += 1
            logger.info("跳过已同步: %s (%s)", title, book_id)
            continue

        # 非强制模式：检查是否需要同步（同增量逻辑）
        if not force and not resume:
            if not need_sync(book_id, nb, index):
                skipped += 1
                logger.debug("跳过未变更: %s", title)
                continue

        try:
            book_data = fetch_book_data(client, book_id)
            new_hash = book_data["meta"]["contentHash"]

            # 全量模式：比对内容哈希，避免无意义的重建
            if force:
                json_path = get_book_file_path(book_id, title, category) / f"{book_id}.json"
                old_book_data = load_json(json_path)
                if old_book_data:
                    old_hash = old_book_data.get("meta", {}).get("contentHash", "")
                    if old_hash and old_hash == new_hash:
                        # 内容未变，跳过保存和 Notion 推送，只更新索引中的 sort
                        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                        book_dir_rel = Path(extract_category(category)) / get_folder_name(title, book_id)
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

            json_path = save_book_data(book_data, title, category)

            # 推送 Notion（直接覆盖）
            if notion_client and json_path:
                if push_single_book(notion_client, book_data, json_path):
                    notion_ok += 1
                else:
                    notion_fail += 1

            # 更新索引（使用相对路径，避免不同环境路径不一致）
            now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            book_dir_rel = Path(extract_category(category)) / get_folder_name(title, book_id)
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

            synced += 1
            logger.info("同步成功 [%d/%d]: %s", synced, len(notebooks), title)

        except Exception as e:
            failed += 1
            logger.error("同步失败: %s (%s) - %s", title, book_id, e)
            # 单书错误不中断，保留本地已有备份
            continue

    logger.info(
        "全量同步完成: 成功 %d, 失败 %d, 跳过 %d | Notion: 成功 %d, 失败 %d",
        synced, failed, skipped, notion_ok, notion_fail,
    )


def sync_incremental(client: WeReadClient):
    """增量同步

    双重校验判断需要同步的书籍，仅同步变更书籍。
    每本书同步后立即推送到 Notion。
    """
    logger.info("开始增量同步")
    index = load_index()

    # 获取所有有笔记的书籍
    notebooks = client.get_all_notebooks()
    logger.info("远程共 %d 本有笔记的书籍", len(notebooks))

    # 筛选需要同步的书籍
    to_sync = []
    for nb in notebooks:
        book_id = nb.get("bookId", "")
        if need_sync(book_id, nb, index):
            to_sync.append(nb)

    if not to_sync:
        logger.info("没有需要同步的书籍")
        return

    logger.info("需要同步 %d 本书籍", len(to_sync))

    # 初始化 Notion 客户端
    from notion_push import push_single_book, _get_notion_client
    try:
        notion_client = _get_notion_client()
        logger.info("Notion 客户端已初始化")
    except Exception as e:
        logger.warning("Notion 初始化失败，将跳过 Notion 推送: %s", e)
        notion_client = None

    synced = 0
    failed = 0
    notion_ok = 0
    notion_fail = 0

    for nb in to_sync:
        book_id = nb.get("bookId", "")
        book_info = nb.get("book", {})
        title = book_info.get("title", f"未知书名_{book_id}")
        # 从 categories 数组获取分类标题
        categories = book_info.get("categories", [])
        if categories and len(categories) > 0:
            category = categories[0].get("title", "")
        else:
            category = ""

        try:
            # 1. 拉取完整 API 数据
            book_data = fetch_book_data(client, book_id)

            # 2. 写入 JSON（原子写入）
            json_path = save_book_data(book_data, title, category)

            # 3. 渲染 Markdown（save_book_data 内部已调用）

            # 4. 推送 Notion（直接覆盖）
            if notion_client and json_path:
                if push_single_book(notion_client, book_data, json_path):
                    notion_ok += 1
                else:
                    notion_fail += 1

            # 5. 更新 index.json（使用相对路径，避免不同环境路径不一致）
            now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            book_dir_rel = Path(extract_category(category)) / get_folder_name(title, book_id)
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

            synced += 1
            logger.info("增量同步成功: %s", title)

        except Exception as e:
            failed += 1
            logger.error("增量同步失败: %s (%s) - %s", title, book_id, e)
            continue

    logger.info(
        "增量同步完成: 成功 %d, 失败 %d | Notion: 成功 %d, 失败 %d",
        synced, failed, notion_ok, notion_fail,
    )


def sync_full_compare(client: WeReadClient):
    """全量比对（兜底机制）

    每月第一个周日触发，全量比对所有书籍。
    """
    logger.info("开始全量比对")
    index = load_index()

    notebooks = client.get_all_notebooks()
    logger.info("远程共 %d 本有笔记的书籍", len(notebooks))

    synced = 0
    failed = 0

    for nb in notebooks:
        book_id = nb.get("bookId", "")
        book_info = nb.get("book", {})
        title = book_info.get("title", f"未知书名_{book_id}")
        # 从 categories 数组获取分类标题
        categories = book_info.get("categories", [])
        if categories and len(categories) > 0:
            category = categories[0].get("title", "")
        else:
            category = ""

        try:
            book_data = fetch_book_data(client, book_id)

            # 比对：检查本地 JSON 是否存在且 lastSync 一致
            book_dir = get_book_file_path(book_id, title, category)
            json_path = book_dir / f"{book_id}.json"
            local_data = load_json(json_path)

            local_last_sync = ""
            if local_data:
                local_last_sync = local_data.get("meta", {}).get("lastSync", "")

            # 如果本地数据存在且同步时间相同，跳过
            if local_data and local_last_sync:
                # 仅更新索引中的 sort 和数量（使用相对路径）
                book_dir_rel = Path(extract_category(category)) / get_folder_name(title, book_id)
                index["books"][book_id] = {
                    "title": title,
                    "category": extract_category(category),
                    "path": str(book_dir_rel / f"{book_id}.json"),
                    "sort": nb.get("sort", 0),
                    "noteCount": nb.get("noteCount", 0),
                    "reviewCount": nb.get("reviewCount", 0),
                    "bookmarkCount": nb.get("bookmarkCount", 0),
                    "lastSync": local_last_sync,
                }
                continue

            save_book_data(book_data, title, category)

            now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            book_dir_rel = Path(extract_category(category)) / get_folder_name(title, book_id)
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

            synced += 1
            logger.info("全量比对同步: %s", title)

        except Exception as e:
            failed += 1
            logger.error("全量比对失败: %s (%s) - %s", title, book_id, e)
            continue

    logger.info("全量比对完成: 同步 %d, 跳过 %d, 失败 %d",
                synced, len(notebooks) - synced - failed, failed)


def rebuild_markdown():
    """手动重建：从所有 JSON 重新渲染 Markdown

    遍历 data/ 下所有 JSON 文件，重新生成对应的 Markdown。
    """
    logger.info("开始重建 Markdown")
    data_dir = get_data_dir()
    index = load_index()

    count = 0
    for json_path in sorted(data_dir.rglob("*.json")):
        # 跳过 index.json
        if json_path.name == "index.json":
            continue

        try:
            book_data = load_json(json_path)
            if not book_data or "meta" not in book_data:
                continue

            book_id = book_data["meta"]["bookId"]
            title = book_data["meta"]["title"]
            md_content = render_markdown(book_data)
            md_path = json_path.with_suffix(".md")
            atomic_write_text(md_path, md_content)
            count += 1
            logger.info("重建: %s (%s)", title, book_id)

        except Exception as e:
            logger.error("重建失败: %s - %s", json_path, e)

    logger.info("重建完成: 共 %d 个文件", count)


def main():
    parser = argparse.ArgumentParser(description="微信读书笔记同步工具")
    parser.add_argument(
        "--mode",
        choices=["full", "incremental", "full-compare", "rebuild"],
        default="incremental",
        help="同步模式 (默认: incremental)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点续传（仅 full 模式有效）",
    )
    args = parser.parse_args()

    # 加载环境变量
    load_env()

    # rebuild 模式不需要 API
    if args.mode == "rebuild":
        rebuild_markdown()
        return

    try:
        client = WeReadClient()
    except ValueError as e:
        logger.error("%s", e)
        sys.exit(1)

    try:
        if args.mode == "full":
            sync_full(client, resume=args.resume)
        elif args.mode == "incremental":
            sync_incremental(client)
        elif args.mode == "full-compare":
            sync_full_compare(client)
    except UpgradeRequiredError as e:
        logger.error("API 需要升级: %s", e.message)
        sys.exit(2)


if __name__ == "__main__":
    main()
