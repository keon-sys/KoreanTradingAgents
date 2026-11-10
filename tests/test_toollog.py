import re

import pytest

from krta.cache import FileCache
from krta.redact import Redactor
from krta.toollog import ToolLog


def test_toollog_redacts(tmp_path):
    path = tmp_path / "message_tool.log"
    ToolLog(path, Redactor(["ABCD9999"])).write("data", "x", "token=ABCD9999")
    text = path.read_text(encoding="utf-8")
    assert "ABCD9999" not in text and "token=***" in text


def test_toollog_entry_format_and_no_truncation(tmp_path):
    path = tmp_path / "sub" / "message_tool.log"
    tl = ToolLog(path, Redactor([]))
    big = "x" * 200_000
    tl.write("agent", "market", big)
    tl.write("data", "dart", "second")
    text = path.read_text(encoding="utf-8")
    header = r"=== \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+09:00 \[agent\] market ===\n"
    assert re.match(header + big + "\n", text)
    assert text.endswith("[data] dart ===\nsecond\n")


def test_toollog_rejects_unknown_kind(tmp_path):
    with pytest.raises(ValueError):
        ToolLog(tmp_path / "l", Redactor([])).write("other", "x", "y")


def test_cache_roundtrip(tmp_path):
    cache = FileCache(tmp_path)
    assert cache.get_json("krx/ep/20261110.json") is None
    cache.put_json("krx/ep/20261110.json", {"a": [1, "한글"]})
    assert cache.get_json("krx/ep/20261110.json") == {"a": [1, "한글"]}
    assert (tmp_path / "krx/ep/20261110.json").is_file()


def test_cache_rejects_traversal(tmp_path):
    with pytest.raises(ValueError):
        FileCache(tmp_path).put_json("../x", {})


@pytest.mark.parametrize("key", ["", "/abs.json", "a/../../x", "a\\..\\x", "./"])
def test_cache_rejects_bad_keys(tmp_path, key):
    with pytest.raises(ValueError):
        FileCache(tmp_path / "c").get_json(key)
