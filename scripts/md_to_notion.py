"""Markdown 转 Notion Block

将 Markdown 内容转换为 Notion API 的 Block 格式。
"""

import re
from typing import Optional


def md_to_blocks(md_content: str) -> list[dict]:
    """将 Markdown 内容转换为 Notion Blocks

    Args:
        md_content: Markdown 字符串

    Returns:
        Notion Block 列表
    """
    blocks: list[dict] = []

    # 跳过 YAML Front Matter
    lines = md_content.split("\n")
    if lines and lines[0].strip() == "---":
        end_idx = 1
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                end_idx = i + 1
                break
        lines = lines[end_idx:]

    # 重新组合为文本
    text = "\n".join(lines)

    # 按行处理
    i = 0
    all_lines = text.split("\n")

    while i < len(all_lines):
        line = all_lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # 分隔线
        if stripped == "---":
            blocks.append(_divider_block())
            i += 1
            continue

        # 标题
        heading_match = re.match(r"^(#{1,3})\s+(.+)$", stripped)
        if heading_match:
            level = len(heading_match.group(1))
            title_text = heading_match.group(2)
            blocks.append(_heading_block(title_text, level))
            i += 1
            continue

        # 图片（独立行）
        image_match = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", stripped)
        if image_match:
            alt_text = image_match.group(1)
            image_url = image_match.group(2)
            if image_url.startswith(("http://", "https://")):
                blocks.append(_image_block(image_url, alt_text))
                i += 1
                continue

        # 表格
        if stripped.startswith("|"):
            table_lines = []
            while i < len(all_lines) and all_lines[i].strip().startswith("|"):
                table_lines.append(all_lines[i].strip())
                i += 1
            table_block = _parse_table(table_lines)
            if table_block:
                blocks.append(table_block)
            continue

        # 引用块（可能多行）
        if stripped.startswith(">"):
            quote_lines = []
            while i < len(all_lines):
                current = all_lines[i].strip()
                if not current:
                    i += 1
                    continue
                if current.startswith(">"):
                    quote_text = re.sub(r"^>\s?", "", all_lines[i])
                    quote_lines.append(quote_text)
                    i += 1
                else:
                    break
            if quote_lines:
                blocks.append(_parse_quote_block(quote_lines))
            continue

        # 无序列表
        if re.match(r"^[-*+]\s+", stripped):
            list_items = []
            while i < len(all_lines):
                current = all_lines[i].strip()
                if not current:
                    i += 1
                    continue
                list_match = re.match(r"^[-*+]\s+(.+)$", all_lines[i])
                if list_match:
                    list_items.append(list_match.group(1))
                    i += 1
                else:
                    break
            if list_items:
                blocks.append(_bulleted_list_block(list_items))
            continue

        # 有序列表
        ordered_match = re.match(r"^(\d+)\.\s+(.+)$", stripped)
        if ordered_match:
            list_items = []
            while i < len(all_lines):
                current = all_lines[i].strip()
                if not current:
                    i += 1
                    continue
                om = re.match(r"^(\d+)\.\s+(.+)$", all_lines[i])
                if om:
                    list_items.append(om.group(2))
                    i += 1
                else:
                    break
            if list_items:
                blocks.append(_numbered_list_block(list_items))
            continue

        # 普通段落（收集连续的非空行）
        para_lines = []
        while i < len(all_lines):
            current = all_lines[i].strip()
            if not current:
                i += 1
                break
            # 如果遇到其他 block 类型，停止收集
            if current.startswith(("#", ">", "-", "*", "+", "|", "!")):
                if re.match(r"^(#{1,3})\s+", current):
                    break
                if current.startswith(">"):
                    break
                if re.match(r"^[-*+]\s+", current):
                    break
                if current.startswith("|"):
                    break
                if re.match(r"^!\[", current):
                    break
            para_lines.append(all_lines[i])
            i += 1

        if para_lines:
            para_text = "\n".join(para_lines)
            blocks.append(_parse_paragraph(para_text))

    return blocks


def _parse_quote_block(lines: list[str]) -> dict:
    """解析引用块，根据内容类型生成不同 block"""
    text = "\n".join(lines)

    # 过滤掉 "在 App 中打开" 行
    if "在 App 中打开" in text:
        filtered_lines = [line for line in lines if "在 App 中打开" not in line]
        if not filtered_lines:
            return _quote_block("")
        text = "\n".join(filtered_lines)

    stripped_text = text.strip()
    if not stripped_text:
        return _quote_block("")

    # 检查是否是划线内容（以 📌 开头）
    if stripped_text.startswith("📌"):
        content = stripped_text[1:].strip()
        # 如果是时间戳行，用灰色段落
        if content.startswith("⏱"):
            return _paragraph_block(content, color="gray_background")
        # 划线内容用引用块
        return _quote_block(content)

    # 检查是否是时间戳
    if stripped_text.startswith("⏱"):
        return _paragraph_block(stripped_text, color="gray_background")

    # 默认作为引用块
    return _quote_block(text)


def _parse_paragraph(text: str) -> dict:
    """解析段落，根据前缀生成不同样式的 block"""
    stripped = text.strip()

    if not stripped:
        return _paragraph_block("")

    # 检查是否以特殊标记开头
    if stripped.startswith("💭"):
        content = stripped[1:].strip()
        return _callout_block(content, icon="💭")
    elif stripped.startswith("⏱"):
        return _paragraph_block(stripped, color="gray_background")
    elif stripped.startswith("（") and stripped.endswith("）"):
        return _paragraph_block(stripped, color="gray")

    return _paragraph_block(text)


def _rich_text(text: str) -> list[dict]:
    """将纯文本转换为 Notion rich_text 格式

    支持：
    - **粗体**
    - *斜体*
    - ~~删除线~~
    - `行内代码`
    - [链接](url)

    Notion 单个 rich_text part 的 content 上限为 2000 字符，
    超过时自动分割为多个 part。
    """
    if not text:
        return []

    NOTION_TEXT_LIMIT = 2000

    # 第一步：解析所有内联格式，生成带格式的 part 列表
    parts = _parse_inline_formats(text)

    # 第二步：分割超过 2000 字符限制的 part
    final_parts = []
    for part in parts:
        content = part["text"]["content"]
        if len(content) <= NOTION_TEXT_LIMIT:
            final_parts.append(part)
        else:
            # 按字符限制分割
            for i in range(0, len(content), NOTION_TEXT_LIMIT):
                chunk = content[i:i + NOTION_TEXT_LIMIT]
                chunk_part = {"type": "text", "text": {"content": chunk}}
                # 保留链接（仅附加到第一个 chunk）
                if "link" in part.get("text", {}) and i == 0:
                    chunk_part["text"]["link"] = part["text"]["link"]
                # 保留 annotations（仅附加到第一个 chunk）
                if "annotations" in part and i == 0:
                    chunk_part["annotations"] = part["annotations"]
                final_parts.append(chunk_part)

    return final_parts


def _parse_inline_formats(text: str) -> list[dict]:
    """解析 Markdown 内联格式，生成 rich_text parts"""
    # 定义格式模式（按优先级排序）
    patterns = [
        (r'\*\*\*(.+?)\*\*\*', 'bold_italic'),  # ***粗斜体***
        (r'\*\*(.+?)\*\*', 'bold'),             # **粗体**
        (r'\*(.+?)\*', 'italic'),               # *斜体*
        (r'__(.+?)__', 'bold'),                 # __粗体__
        (r'_(.+?)_', 'italic'),                 # _斜体_
        (r'~~(.+?)~~', 'strikethrough'),        # ~~删除线~~
        (r'`(.+?)`', 'code'),                   # `行内代码`
        (r'\[([^\]]+)\]\(([^)]+)\)', 'link'),   # [文本](链接)
    ]

    # 使用递归下降解析
    return _parse_text_segment(text, patterns)


def _parse_text_segment(text: str, patterns: list) -> list[dict]:
    """递归解析文本段中的内联格式"""
    if not text:
        return []

    # 查找最早出现的格式标记
    earliest_match = None
    earliest_pattern = None
    earliest_pos = len(text)

    for pattern, fmt_type in patterns:
        match = re.search(pattern, text)
        if match and match.start() < earliest_pos:
            earliest_pos = match.start()
            earliest_match = match
            earliest_pattern = (pattern, fmt_type)

    if earliest_match is None:
        # 没有格式标记，返回纯文本
        return [{"type": "text", "text": {"content": text}}]

    parts = []

    # 匹配前的普通文本
    if earliest_match.start() > 0:
        before_text = text[:earliest_match.start()]
        parts.append({"type": "text", "text": {"content": before_text}})

    # 处理匹配到的格式
    fmt_type = earliest_pattern[1]

    if fmt_type == 'link':
        link_text = earliest_match.group(1)
        link_url = earliest_match.group(2)
        if link_url.startswith(("http://", "https://")):
            parts.append({
                "type": "text",
                "text": {"content": link_text, "link": {"url": link_url}},
            })
        else:
            parts.append({"type": "text", "text": {"content": link_text}})
    elif fmt_type == 'bold_italic':
        inner_text = earliest_match.group(1)
        inner_parts = _parse_text_segment(inner_text, [])
        for part in inner_parts:
            part.setdefault("annotations", {})
            part["annotations"]["bold"] = True
            part["annotations"]["italic"] = True
            parts.append(part)
    elif fmt_type == 'bold':
        inner_text = earliest_match.group(1)
        inner_parts = _parse_text_segment(inner_text, [])
        for part in inner_parts:
            part.setdefault("annotations", {})
            part["annotations"]["bold"] = True
            parts.append(part)
    elif fmt_type == 'italic':
        inner_text = earliest_match.group(1)
        inner_parts = _parse_text_segment(inner_text, [])
        for part in inner_parts:
            part.setdefault("annotations", {})
            part["annotations"]["italic"] = True
            parts.append(part)
    elif fmt_type == 'strikethrough':
        inner_text = earliest_match.group(1)
        inner_parts = _parse_text_segment(inner_text, [])
        for part in inner_parts:
            part.setdefault("annotations", {})
            part["annotations"]["strikethrough"] = True
            parts.append(part)
    elif fmt_type == 'code':
        inner_text = earliest_match.group(1)
        parts.append({
            "type": "text",
            "text": {"content": inner_text},
            "annotations": {"code": True},
        })

    # 匹配后的文本（递归解析）
    after_text = text[earliest_match.end():]
    if after_text:
        parts.extend(_parse_text_segment(after_text, patterns))

    return parts


def _parse_table(lines: list[str]) -> dict | None:
    """解析 Markdown 表格为 Notion table block"""
    if len(lines) < 2:
        return None

    # 第一行是表头
    header_line = lines[0]
    headers = [cell.strip() for cell in header_line.split("|")[1:-1]]

    # 第二行是分隔符，跳过
    # 剩余是数据行
    rows = []
    for line in lines[2:]:
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        if cells:
            rows.append(cells)

    if not headers and not rows:
        return None

    # Notion table block
    table_children = []

    # 表头作为第一行
    header_cells = []
    for h in headers:
        header_cells.append([{
            "type": "table_cell",
            "table_cell": {
                "rich_text": _rich_text(h)
            }
        }])

    if header_cells:
        table_children.append({
            "type": "table_row",
            "table_row": {
                "cells": header_cells
            }
        })

    # 数据行
    for row in rows:
        row_cells = []
        for cell in row:
            row_cells.append([{
                "type": "table_cell",
                "table_cell": {
                    "rich_text": _rich_text(cell)
                }
            }])
        # 补齐列数
        while len(row_cells) < len(headers):
            row_cells.append([{
                "type": "table_cell",
                "table_cell": {
                    "rich_text": []
                }
            }])

        table_children.append({
            "type": "table_row",
            "table_row": {
                "cells": row_cells
            }
        })

    return {
        "object": "block",
        "type": "table",
        "table": {
            "table_width": len(headers) if headers else 2,
            "has_column_header": True,
            "has_row_header": False,
            "children": table_children,
        }
    }


def _heading_block(text: str, level: int = 1) -> dict:
    """创建标题 Block"""
    heading_type = f"heading_{level}"
    return {
        "object": "block",
        "type": heading_type,
        heading_type: {"rich_text": _rich_text(text)},
    }


def _paragraph_block(text: str, color: str = "default") -> dict:
    """创建段落 Block"""
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {
            "rich_text": _rich_text(text),
            "color": color,
        },
    }


def _quote_block(text: str) -> dict:
    """创建引用 Block"""
    return {
        "object": "block",
        "type": "quote",
        "quote": {"rich_text": _rich_text(text)},
    }


def _callout_block(text: str, icon: str = "💡") -> dict:
    """创建 Callout Block（用于想法/评论）"""
    return {
        "object": "block",
        "type": "callout",
        "callout": {
            "rich_text": _rich_text(text),
            "icon": {"type": "emoji", "emoji": icon},
        },
    }


def _divider_block() -> dict:
    """创建分隔线 Block"""
    return {
        "object": "block",
        "type": "divider",
        "divider": {},
    }


def _image_block(url: str, caption: str = "") -> dict:
    """创建图片 Block"""
    block: dict = {
        "object": "block",
        "type": "image",
        "image": {
            "type": "external",
            "external": {"url": url},
        },
    }
    if caption:
        block["image"]["caption"] = _rich_text(caption)
    return block


def _bulleted_list_block(items: list[str]) -> dict:
    """创建无序列表 Block（使用第一个 item，其余需要单独 block）

    注意：Notion API 中列表项需要单独创建，这里返回第一个，
    调用方需要处理剩余项。
    """
    # 由于 Notion API 限制，我们返回一个包含所有项的段落列表
    # 实际使用时由调用方展开
    if not items:
        return _paragraph_block("")

    # 返回第一个列表项，调用方需要循环处理
    return {
        "object": "block",
        "type": "bulleted_list_item",
        "bulleted_list_item": {
            "rich_text": _rich_text(items[0]),
        },
    }


def _numbered_list_block(items: list[str]) -> dict:
    """创建有序列表 Block"""
    if not items:
        return _paragraph_block("")

    return {
        "object": "block",
        "type": "numbered_list_item",
        "numbered_list_item": {
            "rich_text": _rich_text(items[0]),
        },
    }


def md_to_blocks_enhanced(md_content: str) -> list[dict]:
    """增强版 Markdown 转 Notion Blocks

    支持更多 Markdown 特性，生成更丰富的 Notion 页面结构。
    """
    blocks: list[dict] = []

    # 跳过 YAML Front Matter
    lines = md_content.split("\n")
    if lines and lines[0].strip() == "---":
        end_idx = 1
        for j in range(1, len(lines)):
            if lines[j].strip() == "---":
                end_idx = j + 1
                break
        lines = lines[end_idx:]

    text = "\n".join(lines)
    all_lines = text.split("\n")

    i = 0
    while i < len(all_lines):
        line = all_lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # 分隔线
        if stripped == "---":
            blocks.append(_divider_block())
            i += 1
            continue

        # 标题
        heading_match = re.match(r"^(#{1,3})\s+(.+)$", stripped)
        if heading_match:
            level = len(heading_match.group(1))
            title_text = heading_match.group(2)
            blocks.append(_heading_block(title_text, level))
            i += 1
            continue

        # 图片（独立行）
        image_match = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", stripped)
        if image_match:
            alt_text = image_match.group(1)
            image_url = image_match.group(2)
            if image_url.startswith(("http://", "https://")):
                blocks.append(_image_block(image_url, alt_text))
                i += 1
                continue

        # 表格
        if stripped.startswith("|"):
            table_lines = []
            while i < len(all_lines) and all_lines[i].strip().startswith("|"):
                table_lines.append(all_lines[i].strip())
                i += 1
            table_block = _parse_table(table_lines)
            if table_block:
                blocks.append(table_block)
            continue

        # 引用块
        if stripped.startswith(">"):
            quote_lines = []
            while i < len(all_lines):
                current = all_lines[i].strip()
                if not current:
                    i += 1
                    continue
                if current.startswith(">"):
                    quote_text = re.sub(r"^>\s?", "", all_lines[i])
                    quote_lines.append(quote_text)
                    i += 1
                else:
                    break
            if quote_lines:
                blocks.append(_parse_quote_block(quote_lines))
            continue

        # 无序列表
        if re.match(r"^[-*+]\s+", stripped):
            list_items = []
            while i < len(all_lines):
                current = all_lines[i].strip()
                if not current:
                    i += 1
                    continue
                list_match = re.match(r"^[-*+]\s+(.+)$", all_lines[i])
                if list_match:
                    list_items.append(list_match.group(1))
                    i += 1
                else:
                    break
            for item in list_items:
                blocks.append({
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": _rich_text(item),
                    },
                })
            continue

        # 有序列表
        ordered_match = re.match(r"^(\d+)\.\s+(.+)$", stripped)
        if ordered_match:
            list_items = []
            while i < len(all_lines):
                current = all_lines[i].strip()
                if not current:
                    i += 1
                    continue
                om = re.match(r"^(\d+)\.\s+(.+)$", all_lines[i])
                if om:
                    list_items.append(om.group(2))
                    i += 1
                else:
                    break
            for item in list_items:
                blocks.append({
                    "object": "block",
                    "type": "numbered_list_item",
                    "numbered_list_item": {
                        "rich_text": _rich_text(item),
                    },
                })
            continue

        # 普通段落
        para_lines = []
        while i < len(all_lines):
            current = all_lines[i].strip()
            if not current:
                i += 1
                break
            # 如果遇到其他 block 类型，停止收集
            if current.startswith(("#", ">", "-", "*", "+", "|", "!")):
                if re.match(r"^(#{1,3})\s+", current):
                    break
                if current.startswith(">"):
                    break
                if re.match(r"^[-*+]\s+", current):
                    break
                if current.startswith("|"):
                    break
                if re.match(r"^!\[", current):
                    break
            para_lines.append(all_lines[i])
            i += 1

        if para_lines:
            para_text = "\n".join(para_lines)
            blocks.append(_parse_paragraph(para_text))

    return blocks
