"""Decision memory: record each final decision, score it once its horizon has
passed, and feed scored results of the same stock back as short lessons.

Storage: one JSON object per line in `<data_dir>/memory-kr/decisions.jsonl` on the data
volume, shared by every run. Only decision fields are stored (no keys, no prompts).
Writers take an exclusive `flock` on a sibling lock file; rewrites go through
write-then-rename, and readers skip any line that does not parse."""

from __future__ import annotations

import fcntl
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import pandas as pd

from krta.agents.schemas import RATING_KO, Decision, Rating
from krta.collect import MARKET_INDEX, DataBundle
from krta.inputs import TradingCalendar
from krta.net import FetchError
from krta.report import write_atomic

History = Callable[[str, date, int], pd.DataFrame]
LESSONS_LIMIT = 5
MAX_ATTEMPTS = 5  # failed scoring tries before an entry (e.g. delisted) is given up
# Extra rows requested beyond the window: Yahoo can carry stale rows on KRX holidays
# (e.g. election days), which would otherwise push the decision-date row out of the tail.
ROW_MARGIN = 10
_KEY = ("code", "market", "date", "horizon")


def memory_path(data_dir: Path) -> Path:
    return Path(data_dir) / "memory-kr" / "decisions.jsonl"


def _pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def _price(v: int | None) -> str:
    return "없음" if v is None else f"{v:,}원"


def _hit(flag: bool | None) -> str:
    return "해당 없음" if flag is None else ("도달" if flag else "미도달")


def _crossed(level: int | None, base: float, high: float, low: float) -> bool | None:
    """Whether the window's range reached `level`, approached from the base close."""
    if level is None:
        return None
    return bool(high >= level) if level >= base else bool(low <= level)


def _window_end(row: dict, today: date, cal: TradingCalendar) -> date | None:
    """Last session of the entry's window, or None while it has not closed before `today`."""
    try:
        end = cal.next_sessions(date.fromisoformat(row["date"]), int(row["trading_days"]))[-1]
    except ValueError:  # window runs past the calendar's range: cannot close yet
        return None
    return end if end < today else None


def _closes(df: pd.DataFrame, start: date, end: date) -> tuple[float, float] | None:
    """Close of the last row on/before `start` and of the row exactly at `end`."""
    before = df[df.index <= pd.Timestamp(start)]
    if before.empty or pd.Timestamp(end) not in df.index:
        return None
    return float(before["close"].iloc[-1]), float(df.at[pd.Timestamp(end), "close"])


def _score(row: dict, end: date, today: date, history: History) -> dict | None:
    """Scored result, or None when the prices cannot be fetched or lack the needed rows."""
    start = date.fromisoformat(row["date"])
    rows = int(row["trading_days"]) + 1 + ROW_MARGIN
    try:
        stock = history(f"{row['code']}.{row['market']}", end, rows)
        bench = history(MARKET_INDEX[row["market"]], end, rows)
    except FetchError:
        return None
    s, b = _closes(stock, start, end), _closes(bench, start, end)
    if s is None or b is None:
        return None
    (base, last), (b_base, b_last) = s, b
    span = stock[(stock.index > pd.Timestamp(start)) & (stock.index <= pd.Timestamp(end))]
    high, low = float(span["high"].max()), float(span["low"].min())
    ret, b_ret = last / base - 1, b_last / b_base - 1
    return {
        "return": ret,
        "bench_return": b_ret,
        "excess": ret - b_ret,
        "hit_target": _crossed(row.get("target_price"), base, high, low),
        "hit_stop": _crossed(row.get("stop_loss"), base, high, low),
        "window_end": end.isoformat(),
        "settled_on": today.isoformat(),
    }


class Memory:
    def __init__(self, path: Path):
        self.path = Path(path)

    # ---------------------------------------------------------------- storage

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path.with_suffix(".lock"), "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def entries(self) -> list[dict]:
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        out = []
        for line in text.splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # partial line from an interrupted append
            if isinstance(row, dict) and all(k in row for k in _KEY):
                out.append(row)
        return out

    def _rewrite(self, rows: list[dict]) -> None:
        write_atomic(self.path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

    # ---------------------------------------------------------------- API

    def record(self, b: DataBundle, d: Decision) -> None:
        """Store the decision. A rerun of the same code/date/horizon replaces the old entry.

        Unconfirmed decisions are stored with the flag set and are never settled."""
        row = {
            "code": b.target.code,
            "market": b.target.market,
            "date": b.date.isoformat(),
            "horizon": b.horizon.raw,
            "trading_days": b.horizon.trading_days,
            "rating": d.rating.value,
            "target_price": d.target_price,
            "stop_loss": d.stop_loss,
            "unconfirmed": b.unconfirmed,
            "settled": None,
        }
        key = tuple(row[k] for k in _KEY)
        with self._locked():
            rows = [r for r in self.entries() if tuple(r[k] for k in _KEY) != key]
            self._rewrite([*rows, row])

    def settle(self, today: date, history: History, cal: TradingCalendar) -> int:
        """Score every pending confirmed entry whose window closed before `today`.

        Window: the `trading_days` sessions after the decision date. An entry whose prices
        cannot be fetched or lack the start/end rows stays pending and counts one failed
        attempt; after `MAX_ATTEMPTS` it is no longer retried. Returns how many were scored."""
        with self._locked():
            rows = self.entries()
            count = 0
            changed = False
            for row in rows:
                if row.get("settled") is not None or row.get("unconfirmed", True):
                    continue
                if row.get("attempts", 0) >= MAX_ATTEMPTS:
                    continue
                end = _window_end(row, today, cal)
                if end is None:
                    continue
                changed = True
                result = _score(row, end, today, history)
                if result is None:
                    row["attempts"] = row.get("attempts", 0) + 1
                else:
                    row["settled"] = result
                    count += 1
            if changed:
                self._rewrite(rows)
            return count

    def lessons(self, code: str, as_of: date, limit: int = LESSONS_LIMIT) -> str:
        """Scored decisions of `code` made and scored before `as_of`, newest first.

        Both bounds keep a past-date analysis from seeing results it could not have known."""
        day = as_of.isoformat()
        rows = [
            r for r in self.entries()
            if r["code"] == code and r["date"] < day
            and r.get("settled") and r["settled"]["settled_on"] < day
        ]
        rows.sort(key=lambda r: (r["date"], r["settled"]["settled_on"]), reverse=True)
        out = []
        for r in rows[:limit]:
            s = r["settled"]
            rating = Rating(r["rating"])
            out.append(
                f"- {r['date']} {r['horizon']} {RATING_KO[rating]}({rating.value}): "
                f"수익률 {_pct(s['return'])}, 벤치마크 {_pct(s['bench_return'])}, "
                f"초과 {_pct(s['excess'])}; "
                f"목표가 {_price(r['target_price'])} {_hit(s['hit_target'])}, "
                f"손절가 {_price(r['stop_loss'])} {_hit(s['hit_stop'])} (~{s['window_end']})"
            )
        return "\n".join(out)
