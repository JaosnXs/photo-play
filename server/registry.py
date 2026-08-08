"""registry.json 读写封装：photo_id -> {video_file, title}。

注册表为人工可读/可手改的 JSON 文件，与特征文件解耦。
读取线程安全；文件不存在或格式错误时兜底为空注册表。
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class Registry:
    """照片-视频关联注册表。"""

    def __init__(self, registry_path: Path | str):
        self._path = Path(registry_path)
        self._lock = threading.Lock()
        self._entries: dict[str, dict] = {}
        self.reload()

    def reload(self) -> None:
        """从磁盘重新加载注册表。"""
        with self._lock:
            if not self._path.exists():
                logger.warning("注册表不存在: %s，视为空注册表", self._path)
                self._entries = {}
                return
            try:
                data = json.loads(self._path.read_text(encoding="utf-8-sig"))
                if not isinstance(data, dict):
                    raise ValueError("注册表顶层必须是 JSON 对象")
                self._entries = data
                logger.info("注册表已加载，共 %d 条记录", len(self._entries))
            except (json.JSONDecodeError, ValueError) as exc:
                logger.error("注册表解析失败: %s (%s)，视为空注册表", self._path, exc)
                self._entries = {}

    def save(self) -> None:
        """将当前注册表写回磁盘。"""
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._entries, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def get(self, photo_id: str) -> Optional[dict]:
        """按 photo_id 查询条目，不存在返回 None。"""
        with self._lock:
            return self._entries.get(photo_id)

    def set(self, photo_id: str, video_file: str, title: str = "") -> None:
        """登记/更新一条关联记录（不落盘，需显式调用 save）。"""
        with self._lock:
            self._entries[photo_id] = {"video_file": video_file, "title": title}

    def all_ids(self) -> list[str]:
        with self._lock:
            return list(self._entries.keys())

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
