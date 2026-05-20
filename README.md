# 微信读书笔记同步工具

自动同步微信读书笔记到 GitHub 仓库和 Notion 数据库，支持增量同步、全量同步和网页版链接跳转。

## 功能特性

- **增量同步**：只同步有变化的书籍，高效快速
- **全量同步**：强制重新拉取所有书籍，用于数据修复
- **双平台推送**：同步到 GitHub 仓库和 Notion 数据库
- **自动定时**：每日自动同步，也可手动触发
- **本地备份**：JSON + Markdown 双格式保存
- **网页链接**：生成微信读书网页版直达链接，支持跳转到书籍、章节和划线位置
- **App 链接**：保留微信读书 App 深层链接

## 项目结构

```
.
├── .github/workflows/      # GitHub Actions 工作流
│   ├── daily-sync.yml      # 每日增量同步（北京时间 8:00）
│   └── manual-full-sync.yml # 手动全量同步
├── scripts/                # 核心脚本
│   ├── api.py             # 微信读书 API 封装
│   ├── config.py          # 配置管理
│   ├── md_to_notion.py    # Markdown 转 Notion blocks
│   ├── notion_client.py   # Notion API 封装
│   ├── notion_push.py     # Notion 推送逻辑
│   ├── renderer.py        # Markdown 渲染
│   ├── sync.py            # 同步主逻辑
│   └── utils.py           # 工具函数（含网页链接生成）
├── data/                   # 书籍数据（自动创建）
├── index.json             # 书籍索引（自动创建）
├── .env                   # 环境变量（需手动创建）
└── requirements.txt       # Python 依赖
```

## 快速开始

### 1. 配置环境变量

创建 `.env` 文件：

```bash
WEREAD_API_KEY=your_weread_api_key
NOTION_API_KEY=your_notion_integration_token
NOTION_DATABASE_ID=your_notion_database_id
```

### 2. 配置 Notion 数据库

在 Notion 中创建一个数据库，添加以下属性：

| 属性名 | 类型 | 说明 |
|--------|------|------|
| 书名 | Title | 书籍标题 |
| bookId | Rich Text | 微信读书书籍 ID |
| 作者 | Rich Text | 作者名（可选） |
| 译者 | Rich Text | 译者名（可选） |
| 出版社 | Rich Text | 出版社（可选） |
| 分类 | Select | 书籍分类（可选） |
| 阅读进度 | Select | 已读完/在读/未读 |
| 笔记数 | Number | 笔记总数 |
| 封面 | Files | 书籍封面图（可选） |
| App链接 | URL | 微信读书 App 链接（可选） |
| 网页链接 | URL | 微信读书网页版链接（可选） |

### 3. 本地运行

```bash
# 安装依赖
pip install -r requirements.txt

# 增量同步（推荐日常使用）
python scripts/sync.py --mode incremental

# 全量同步（强制重新拉取所有书籍）
python scripts/sync.py --mode full

# 断点续传
python scripts/sync.py --mode full --resume

# 重建 Markdown
python scripts/sync.py --mode rebuild
```

### 4. GitHub Actions 自动同步

1. 将项目推送到 GitHub 仓库
2. 在仓库 Settings → Secrets → Actions 中添加以下 secrets：
   - `WEREAD_API_KEY`
   - `NOTION_API_KEY`
   - `NOTION_DATABASE_ID`
3. 每日北京时间 8:00 自动运行增量同步
4. 也可在 Actions 页面手动触发全量同步

## 同步模式

### 增量同步 (incremental)

自动检测变更的书籍并同步：
- 比对 sort 值变化（最后操作时间戳）
- 比对笔记、想法、书签数量变化
- 比对本地同步时间与远程 sort 时间

### 全量同步 (full)

强制重新拉取所有书籍数据，覆盖本地已有数据。用于：
- 首次同步
- 数据完整性校验
- 修复数据不一致
- 同步微信读书端的"修改笔记"操作

### 全量比对 (full-compare)

每月第一个周日自动触发，全量比对所有书籍并补充缺失数据。

### 重建 Markdown (rebuild)

从所有 JSON 文件重新渲染 Markdown，适用于模板更新后的批量重渲染。

## 链接类型

同步数据中包含多种链接：

| 链接类型 | 格式 | 说明 |
|---------|------|------|
| App 链接 | `weread://reading?...` | 微信读书 App 深层链接 |
| 网页链接 | `https://weread.qq.com/web/reader/...` | 微信读书网页版链接 |

### 网页链接功能

工具会自动生成微信读书网页版直达链接：

- **书籍链接**：`get_weread_web_url(book_id)` → 书籍阅读页面
- **章节链接**：`get_weread_web_chapter_url(book_id, chapter_uid)` → 指定章节
- **划线链接**：`get_weread_web_bookmark_url(book_id, chapter_uid, range_start, range_end)` → 指定划线位置

这些链接会在以下位置自动填充：
- JSON 元数据中的 `webLink` 字段
- Markdown 文件元信息表格中的书名超链接
- Notion 数据库的"网页链接"属性

## 数据格式

### JSON 格式

```json
{
  "meta": {
    "bookId": "123456",
    "title": "书名",
    "author": "作者",
    "category": "分类",
    "noteCount": 10,
    "reviewCount": 5,
    "bookmarkCount": 3,
    "appLink": "weread://reading?bId=123456",
    "webLink": "https://weread.qq.com/web/reader/...",
    "lastSync": "2024-01-01T12:00:00Z"
  },
  "chapters": [
    {
      "chapterUid": 1,
      "title": "第一章",
      "appLink": "weread://reading?bId=123456&chapterUid=1",
      "webLink": "https://weread.qq.com/web/reader/...?chapterUid=1"
    }
  ],
  "content": [
    {
      "chapterTitle": "第一章",
      "items": [
        {
          "type": "highlight",
          "markText": "划线内容",
          "appLink": "weread://bestbookmark?bookId=123456&chapterUid=1&rangeStart=100&rangeEnd=150",
          "webLink": "https://weread.qq.com/web/reader/...?chapterUid=1&rangeStart=100&rangeEnd=150"
        },
        {
          "type": "review",
          "content": "想法内容"
        }
      ]
    }
  ],
  "bookReviews": [],
  "hotBookmarks": []
}
```

### Markdown 格式

```markdown
---
doc_type: weread-highlights-reviews
bookId: "123456"
title: "书名"
webLink: "https://weread.qq.com/web/reader/..."
---

# 元数据

![书名](cover_url)

| 项目 | 内容 |
|------|------|
| 书名 | [书名](https://weread.qq.com/web/reader/...) |
| 作者 | 作者名 |
| 分类 | 分类名 |

---

# 第一章

> 📌 划线内容
> ⏱ 2024-01-01 12:00:00

💭 想法内容
⏱ 2024-01-01 12:30:00

---
```

## 注意事项

1. **API 密钥安全**：`.env` 文件不要提交到 GitHub，使用 GitHub Secrets
2. **Notion 速率限制**：每秒最多 3 次请求，已内置限速
3. **删除的书籍**：本地会保留备份，不会自动删除
4. **修改的笔记**：增量同步检测不到，需要全量同步覆盖
5. **网页链接**：需要 Notion 数据库包含"网页链接"属性才会自动推送

## 依赖

- requests >= 2.28.0
- python-dotenv >= 1.0.0
- notion-client >= 2.0.0

## 许可证

MIT License
