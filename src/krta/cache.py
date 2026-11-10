"""JSON file cache under the data volume, keyed by relative path (e.g. `krx/ep/20261110.json`)."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


class FileCache:
    def __init__(self, root: Path):
        self.root = Path(root)

    def get_json(self, key: str) -> Any | None:
        path = self._path(key)
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None  # corrupt entry: treat as a miss, it will be rewritten

    def put_json(self, key: str, value: Any) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename so a crash never leaves a half-written entry.
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(value, f, ensure_ascii=False)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def _path(self, key: str) -> Path:
        if not key or "\\" in key or "\0" in key:
            raise ValueError(f"잘못된 캐시 키: {key!r}")
        rel = PurePosixPath(key)
        parts = [p for p in rel.parts if p != "."]
        if rel.is_absolute() or not parts or ".." in parts:
            raise ValueError(f"잘못된 캐시 키: {key!r}")
        path = self.root.joinpath(*parts)
        root = self.root.resolve()
        if not path.resolve().is_relative_to(root) or path.resolve() == root:
            raise ValueError(f"잘못된 캐시 키: {key!r}")
        return path
