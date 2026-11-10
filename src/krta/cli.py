"""Entry point: `krta --ticker <code> --horizon <Nd|Nw|Nm|Ny> [--date YYYY-MM-DD]`.

stdout carries exactly one line, `REPORT_DIR=<run_dir>/reports`, printed last on success
(the deployment wrapper parses it). Everything else goes to stderr, redacted.
Exit codes: 0 ok, 2 input error, 3 key problem, 1 anything else.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from collections.abc import Callable, Mapping
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path
from typing import TextIO

from krta.cache import FileCache
from krta.collect import DataBundle, Sources, collect
from krta.config import Settings
from krta.data import yahoo
from krta.data.dart import Dart, DartKeyError
from krta.data.krx_openapi import KrxAuthError, KrxOpenApi, Market
from krta.data import toss
from krta.data.news import GoogleNews
from krta.graph import build_graph, initial_state
from krta.inputs import (
    KST,
    InputError,
    OutOfScopeError,
    ResolvedDate,
    TradingCalendar,
    normalize_code,
    parse_horizon,
    resolve_date,
)
from krta.llm import LLM, AnthropicLLM
from krta.memory import MAX_ATTEMPTS, Memory, memory_path
from krta.net import FetchError, HttpClient
from krta.redact import Redactor
from krta.report import run_dir, write_reports
from krta.toollog import ToolLog

REPORT_DIR_PREFIX = "REPORT_DIR="
LOG_NAME = "message_tool.log"
CACHE_DIR = "cache-kr"
# Runs log here until the target's market (part of the run dir) is known. Same volume as
# logs-kr/, so finalizing is a rename. A failed run's log stays here for diagnosis.
TMP_DIR = "tmp-kr"
MAX_ERROR_CHARS = 500
# A dateless run steps back at most this many sessions to find published KRX quotes.
MAX_STEP_BACK = 5

EXIT_OK, EXIT_ERROR, EXIT_INPUT, EXIT_KEY = 0, 1, 2, 3

SourcesFactory = Callable[[Settings, Redactor, ToolLog, ResolvedDate, TradingCalendar], Sources]
LLMFactory = Callable[[Settings, ToolLog], LLM]


class MissingKeyError(Exception):
    """A required key env var is empty. Names the variable, never a value."""


class NotPublishedError(Exception):
    """No KRX daily quotes within MAX_STEP_BACK sessions of the default date (exit 1)."""


# =========================================================================== factories


def _key(env: Mapping[str, str], name: str) -> str:
    value = env.get(name, "").strip()
    if not value:
        raise MissingKeyError(f"환경변수 {name}가 비어 있음(Secret에 직접 추가 필요)")
    return value


def default_sources(
    settings: Settings,
    redactor: Redactor,
    toollog: ToolLog,
    resolved: ResolvedDate,
    cal: TradingCalendar,
    env: Mapping[str, str] = os.environ,
) -> Sources:
    cache = FileCache(settings.data_dir / CACHE_DIR)
    # Open API, DART and news share one client; collection is sequential (not thread-safe).
    http = HttpClient(redactor, toollog)
    krx = KrxOpenApi(
        http, cache, _key(env, "KRX_API_KEY"),
        # The still-changing day must not be cached.
        unconfirmed_date=resolved.date if resolved.unconfirmed else None,
    )
    dart = Dart(http, cache, _key(env, "DART_API_KEY"))
    # Own client with Toss's pacing; the shared redactor learns the access token at run
    # time. No credentials -> unavailable -> the flow items become missing lines.
    toss_api = toss.TossApi(
        HttpClient(redactor, toollog, min_interval=toss.MIN_INTERVAL), redactor,
        env.get("TOSS_CLIENT_ID"), env.get("TOSS_CLIENT_SECRET"),
    )
    # KRX portal: plaintext login is refused (spec §17), so it is not built.
    return Sources(
        krx=krx, dart=dart, news=GoogleNews(http), history=yahoo.history, cal=cal, toss=toss_api,
    )


def default_llm(settings: Settings, toollog: ToolLog, env: Mapping[str, str] = os.environ) -> LLM:
    _key(env, "ANTHROPIC_API_KEY")  # langchain-anthropic reads it itself; only checked here
    return AnthropicLLM(settings, toollog)


# =========================================================================== stderr guard


class _RedactingLines:
    """Line-buffered text sink: redacts each line and defuses result-line lookalikes.

    Stands in for stdout and stderr while the run executes, so a library print can neither
    leak a secret nor fake the `REPORT_DIR=` line (Job logs merge both streams)."""

    def __init__(self, target: TextIO, redactor: Redactor):
        self._target = target
        self._redactor = redactor
        self._buf = ""

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return False

    @property
    def encoding(self) -> str:
        return getattr(self._target, "encoding", None) or "utf-8"

    def write(self, s: str) -> int:
        self._buf += s
        *lines, self._buf = self._buf.split("\n")
        for line in lines:
            self._emit(line)
        return len(s)

    def flush(self) -> None:
        # Partial lines stay buffered: a secret or the prefix may continue in the next write.
        self._target.flush()

    def drain(self) -> None:
        if self._buf:
            self._emit(self._buf)
            self._buf = ""
        self._target.flush()

    def _emit(self, line: str) -> None:
        line = self._redactor.redact(line)
        if REPORT_DIR_PREFIX in line:
            line = line.replace(REPORT_DIR_PREFIX, "REPORT_DIR(무시)=")
        self._target.write(line + "\n")


# =========================================================================== run


class _InputError(Exception):
    """Wraps an error that means the request itself is invalid (exit 2)."""


class _ArgParser(argparse.ArgumentParser):
    def error(self, message: str):  # type: ignore[override]
        raise _InputError(message)


def _parser() -> argparse.ArgumentParser:
    p = _ArgParser(prog="krta", description="한국 주식(KOSPI·KOSDAQ) LLM 멀티 에이전트 분석")
    p.add_argument("--ticker", required=True, help="종목코드: 005930, 005930.KS, 035720.KQ")
    p.add_argument("--horizon", required=True, help="기간: 5d, 4w, 6m, 1y")
    p.add_argument("--date", default=None, help="분석일 YYYY-MM-DD(KST). 생략 시 최근 확정 거래일")
    return p


def _requested_date(raw: str | None) -> date | None:
    if raw is None or not raw.strip():
        return None
    try:
        return date.fromisoformat(raw.strip())
    except ValueError:
        raise ValueError(f"날짜 형식 오류: {raw!r} (예: 2026-11-09)") from None


def _redact_bundle(b: DataBundle, redactor: Redactor) -> DataBundle:
    """Last line of defence before any prompt: sources should already have redacted
    their errors, but a value that slipped through must not reach the model."""
    r = redactor.redact
    return replace(
        b,
        calendar_md=r(b.calendar_md),
        date_note=r(b.date_note) if b.date_note else b.date_note,
        sections={
            k: replace(s, text=r(s.text), missing=[r(m) for m in s.missing])
            for k, s in b.sections.items()
        },
    )


# =========================================================================== published date


MAX_PROBE_REASON = 80


def _with_note(resolved: ResolvedDate, note: str) -> str:
    return f"{resolved.note}; {note}" if resolved.note else note


def published_date(resolved: ResolvedDate, *, explicit: bool, krx: KrxOpenApi,
                   markets: tuple[Market, ...], cal: TradingCalendar,
                   toollog: ToolLog) -> ResolvedDate:
    """The analysis date adjusted to what KRX has actually published.

    KRX posts day T's quotes later than 18:00 T, when T already is the default date
    (sometimes only the next morning), so an empty `daily_trades` in any of `markets`
    means "not published yet". A dateless run then steps back session by session (at
    most MAX_STEP_BACK); an explicit date is kept but marked unconfirmed. A run that is
    already unconfirmed (explicit date, today, intraday) is not probed: nothing is
    published yet by definition. A failed probe (not auth) proves nothing: that date is
    kept with a note and collection reports the outage as it would have."""
    if resolved.unconfirmed:
        return resolved

    def probe(d: date) -> bool | str:
        """True published, False empty, or the (short, redacted) failure reason."""
        failed: str | None = None
        for m in markets:
            label = f"krx_openapi.daily_trades {m} {d.isoformat()} (공개 확인)"
            try:
                n = len(krx.daily_trades(m, d))
            except FetchError as exc:
                toollog.write("data", label, f"실패: {exc}")
                failed = failed or " ".join(exc.message.split())[:MAX_PROBE_REASON]
                continue
            toollog.write("data", label, f"rows {n}")
            if n == 0:
                return False
        return failed if failed is not None else True

    def failed_note(d: date, reason: str) -> str:
        return f"분석일 {d.isoformat()} 시세 공개 여부 확인 실패({reason})"

    d = resolved.date
    result = probe(d)
    if result is True:
        return resolved
    if isinstance(result, str):
        return replace(resolved, note=_with_note(resolved, failed_note(d, result)))
    if explicit:
        return replace(resolved, unconfirmed=True,
                       note=_with_note(resolved, f"분석일 {d.isoformat()} 시세 미공개(빈 응답)"))
    p = d
    for _ in range(MAX_STEP_BACK):
        p = cal.previous_session(p)
        result = probe(p)
        if result is False:
            continue
        note = f"분석일 {d.isoformat()} 시세 미공개로 직전 거래일 {p.isoformat()} 기준"
        if isinstance(result, str):
            note += f"; {failed_note(p, result)}"
        return replace(resolved, date=p, note=_with_note(resolved, note))
    raise NotPublishedError(
        f"KRX 일별 시세 미공개: {d.isoformat()}와 직전 {MAX_STEP_BACK}거래일"
        f"({p.isoformat()}까지) 모두 빈 응답. 나중에 다시 실행하거나 --date로 지정")


class _Run:
    def __init__(self, now: Callable[[], datetime], sources_factory: SourcesFactory,
                 llm_factory: LLMFactory):
        self.now = now
        self.sources_factory = sources_factory
        self.llm_factory = llm_factory
        self.temp_log: Path | None = None

    def __call__(self, argv: list[str] | None, redactor: Redactor) -> Path:
        # ---- inputs: any ValueError here is a bad request
        try:
            args = _parser().parse_args(argv)
            settings = Settings.from_env()
            cal = TradingCalendar()
            instant = self.now()
            requested = _requested_date(args.date)
            resolved = resolve_date(requested, instant, cal)
            default_date = resolve_date(None, instant, cal).date
            horizon = parse_horizon(args.horizon)
            code, suffix = normalize_code(args.ticker)
        except ValueError as exc:
            raise _InputError(str(exc)) from None

        tmp_root = settings.data_dir / TMP_DIR
        tmp_root.mkdir(parents=True, exist_ok=True)
        self.temp_log = Path(tempfile.mkdtemp(prefix="run-", dir=tmp_root)) / LOG_NAME
        toollog = ToolLog(self.temp_log, redactor)

        src = self.sources_factory(settings, redactor, toollog, resolved, cal)
        llm = self.llm_factory(settings, toollog)

        # Before anything uses the date (lessons, collection, run dir). Without a suffix the
        # market is not known before the listing, so both must be published. Non-empty
        # results are cached and reused by collection.
        markets: tuple[Market, ...] = (suffix,) if suffix else ("KS", "KQ")  # type: ignore[assignment]
        resolved = published_date(resolved, explicit=requested is not None, krx=src.krx,
                                  markets=markets, cal=cal, toollog=toollog)
        if requested is None:
            default_date = resolved.date  # still the latest analysable day: no past-date note

        memory = Memory(memory_path(settings.data_dir))
        self._settle(memory, instant, src, cal, toollog)
        lessons = memory.lessons(code, resolved.date)

        # ---- collection: listing lookup failures (out of scope, wrong suffix, before
        # listing) are input errors; any other ValueError (e.g. a malformed number in source
        # data) is a run failure (exit 1).
        try:
            bundle = collect(args.ticker, resolved, horizon, src, llm, toollog)
        except (InputError, OutOfScopeError) as exc:
            raise _InputError(str(exc)) from None
        bundle = _redact_bundle(bundle, redactor)

        state = build_graph(llm, settings).invoke(initial_state(bundle, lessons))

        out = run_dir(settings.data_dir, bundle)
        out.mkdir(parents=True, exist_ok=True)
        final_log = out / LOG_NAME
        if self.temp_log.exists():
            os.replace(self.temp_log, final_log)
        toollog.path = final_log
        shutil.rmtree(self.temp_log.parent, ignore_errors=True)
        self.temp_log = None

        write_reports(out, state, default_date=default_date)
        memory.record(bundle, state["decision"])
        return out.absolute() / "reports"

    @staticmethod
    def _settle(memory: Memory, instant: datetime, src: Sources, cal: TradingCalendar,
                toollog: ToolLog) -> None:
        try:
            settled = memory.settle(instant.astimezone(KST).date(), src.history, cal)
            toollog.write("data", "memory.settle", f"채점 {settled}건")
        except Exception as exc:  # noqa: BLE001 - scoring old decisions must not stop a run
            toollog.write("data", "memory.settle", f"실패: {type(exc).__name__}")
            return
        given_up = sum(
            1 for r in memory.entries()
            if r.get("settled") is None and not r.get("unconfirmed", True)
            and r.get("attempts", 0) >= MAX_ATTEMPTS
        )
        if given_up:
            # Goes through the stderr guard (sys.stderr is redirected during the run).
            print(f"krta: 메모리 채점 포기 {given_up}건(가격 조회 {MAX_ATTEMPTS}회 실패, "
                  "더 시도하지 않음)", file=sys.stderr)


def _one_line(text: str, redactor: Redactor) -> str:
    text = " ".join(redactor.redact(text).split())
    if len(text) > MAX_ERROR_CHARS:
        text = text[:MAX_ERROR_CHARS] + "…"
    return text.replace(REPORT_DIR_PREFIX, "REPORT_DIR(무시)=")


def main(
    argv: list[str] | None = None,
    *,
    now: Callable[[], datetime] | None = None,
    sources_factory: SourcesFactory | None = None,
    llm_factory: LLMFactory | None = None,
) -> int:
    # Built once, first, so even a settings error is printed redacted. Shared by the tool
    # log and every HttpClient.
    redactor = Redactor.from_env()
    real_out, real_err = sys.stdout, sys.stderr
    guard = _RedactingLines(real_err, redactor)
    run = _Run(now or (lambda: datetime.now(KST)), sources_factory or default_sources,
               llm_factory or default_llm)
    code, message = EXIT_OK, ""
    try:
        with redirect_stdout(guard), redirect_stderr(guard):
            reports = run(argv, redactor)
    except _InputError as exc:
        code, message = EXIT_INPUT, f"입력 오류: {exc}"
    except (DartKeyError, KrxAuthError, MissingKeyError) as exc:
        code, message = EXIT_KEY, f"키 문제: {exc}"
    except NotPublishedError as exc:
        code, message = EXIT_ERROR, str(exc)
    except Exception as exc:  # noqa: BLE001 - top level: one redacted line, never a traceback
        code, message = EXIT_ERROR, f"실행 실패({type(exc).__name__}): {exc}"
    finally:
        guard.drain()

    if code != EXIT_OK:
        if run.temp_log is not None:
            if run.temp_log.exists():
                message += f" (도구 로그: {run.temp_log})"
            else:  # failed before anything was logged: leave no empty run dir behind
                shutil.rmtree(run.temp_log.parent, ignore_errors=True)
        real_err.write(f"krta: {_one_line(message, redactor)}\n")
        real_err.flush()
        return code
    real_out.write(f"{REPORT_DIR_PREFIX}{reports}\n")
    real_out.flush()
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
