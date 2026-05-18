"""配置管理模块 — 加载 config.json 和 .env 环境变量"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv


def _find_project_root() -> Path:
    """从当前脚本位置向上查找项目根目录（含 config.json 的目录）"""
    current = Path(__file__).resolve().parent
    for _ in range(5):
        if (current / "config.json").exists():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    # 兜底：假设 scripts/ 的上级就是项目根
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = _find_project_root()


def load_env():
    """加载 .env 文件到 os.environ"""
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def load_config() -> dict:
    """加载 config.json"""
    config_path = PROJECT_ROOT / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_env(key: str, default: str = "") -> str:
    """获取环境变量"""
    return os.environ.get(key, default)


def get_data_dir() -> Path:
    """获取数据目录路径"""
    config = load_config()
    data_dir = config.get("output", {}).get("data_dir", "data")
    return PROJECT_ROOT / data_dir


def get_index_path() -> Path:
    """获取全局索引文件路径"""
    config = load_config()
    index_file = config.get("output", {}).get("index_file", "index.json")
    return PROJECT_ROOT / index_file


def validate_required_env() -> list[str]:
    """验证必需的环境变量，返回缺失的变量列表"""
    missing = []
    
    if not get_env("WEREAD_API_KEY"):
        missing.append("WEREAD_API_KEY")
    
    if not get_env("NOTION_API_KEY"):
        missing.append("NOTION_API_KEY")
    
    if not get_env("NOTION_DATABASE_ID"):
        config = load_config()
        if not config.get("notion", {}).get("database_id"):
            missing.append("NOTION_DATABASE_ID")
    
    return missing
