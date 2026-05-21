# 微信读书笔记同步工具

自动同步微信读书笔记到 GitHub 和 Notion，支持增量同步和全量同步。

## 功能特性

- **增量同步**：只同步有变化的书籍，高效快速
- **全量同步**：智能内容比对，避免无意义的数据重建
- **双平台推送**：同步到 GitHub 仓库和 Notion 数据库
- **自动定时**：每周自动全量同步 + 每日增量同步
- **本地备份**：JSON + Markdown 双格式保存
- **内容哈希**：通过 SHA256 检测想法/书评的文字修改

## 项目结构

```
.
├── .github/workflows/      # GitHub Actions 工作流
│   ├── daily-sync.yml      # 每日增量同步（北京时间 8:00）
│   └── manual-full-sync.yml # 每周全量同步（北京时间 2:00）
├── scripts/                # 核心脚本
│   ├── api.py             # 微信读书 API 封装
│   ├── config.py          # 配置管理
│   ├── md_to_notion.py    # Markdown 转 Notion blocks
│   ├── notion_client.py   # Notion API 封装
│   ├── notion_push.py     # Notion 推送逻辑
│   ├── renderer.py        # Markdown 渲染
│   ├── sync.py            # 同步主逻辑
│   └── utils.py           # 工具函数
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

### 3. 本地运行

```bash
# 安装依赖
pip install -r requirements.txt

# 增量同步（推荐日常使用）
python scripts/sync.py --mode incremental

# 全量同步（智能跳过，内容未变则不重建）
python scripts/sync.py --mode full

# 全量同步（强制重建所有书籍，禁用智能跳过）
python scripts/sync.py --mode full --no-skip

# 断点续传（跳过已同步的书籍）
python scripts/sync.py --mode full --resume
```

### 4. GitHub Actions 自动同步

1. 将项目推送到 GitHub 仓库
2. 在仓库 Settings → Secrets → Actions 中添加以下 secrets：
   - `WEREAD_API_KEY`
   - `NOTION_API_KEY`
   - `NOTION_DATABASE_ID`
3. 每周一北京时间 2:00 自动运行全量同步
4. 每日北京时间 8:00 自动运行增量同步
5. 可在 Actions 页面手动触发同步任务

## 同步逻辑

### 增量同步

```
1. 拉取微信读书 /user/notebooks
2. 比对本地 index.json（sort + 笔记数量）
3. 筛选出变更书籍
4. 对每本变更书籍：
   - 拉取完整 API 数据
   - 写入 JSON（原子写入）
   - 渲染 Markdown
   - 推送 Notion（全量覆盖）
   - 更新 index.json
```

### 全量同步

智能全量同步，通过内容哈希比对避免无意义重建：

- 拉取所有书籍数据
- 计算每本书籍的内容哈希（基于想法/书评）
- 比对本地数据：
  - **内容未变**：仅更新索引中的 sort，跳过文件和 Notion 推送
  - **内容已变**：重新写入 JSON、渲染 Markdown、推送 Notion

使用 `--no-skip` 参数可强制重建所有书籍。

### 内容哈希机制

内容哈希（SHA256）仅基于以下数据计算：

- 章节内的想法/点评（review）
- 整本书的书评（bookReviews）

排除的数据：划线、章节信息、阅读进度等。

用于检测微信读书端的"修改想法/书评"操作（增量同步无法检测此类变更）。

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
    "contentHash": "sha256哈希值",
    "lastSync": "2024-01-01T12:00:00Z"
  },
  "content": [
    {
      "chapterTitle": "第一章",
      "items": [
        {"type": "highlight", "markText": "划线内容", ...},
        {"type": "review", "content": "想法内容", ...}
      ]
    }
  ]
}
```

### Markdown 格式

```markdown
# 书名

**作者：** 作者名  
**分类：** 分类名  
**阅读进度：** 已读完

---

## 第一章

📌 划线内容 ⏱ 2024-01-01 12:00:00

💭 想法内容 ⏱ 2024-01-01 12:30:00
```

## 注意事项

1. **API 密钥安全**：`.env` 文件不要提交到 GitHub，使用 GitHub Secrets
2. **Notion 速率限制**：每秒最多 3 次请求，已内置限速
3. **删除的书籍**：本地会保留备份，不会自动删除
4. **修改的笔记**：增量同步检测不到想法/书评的文字修改，由每周全量同步自动检测

## 许可证

MIT License
