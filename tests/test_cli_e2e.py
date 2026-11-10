"""End-to-end: `krta.cli.main` with fake sources and FakeLLM (sockets disabled)."""

import json
from datetime import date, datetime
from pathlib import Path

import pytest
from fakes import FakeLLM
from test_collect import CAL, FakeDart, FakeKrx, FakeNews, FakeToss, sources
from test_graph import ANALYST_TITLES, Script, role_of

from krta import cli
from krta.agents.pm import PORTFOLIO_MANAGER
from krta.data import toss
from krta.data.dart import DartKeyError
from krta.data.krx_openapi import KrxAuthError
from krta.inputs import KST, ResolvedDate
from krta.memory import MAX_ATTEMPTS, Memory, memory_path
from krta.net import FetchError
from krta.redact import Redactor
from krta.report import REPORT_FILES
from krta.toollog import ToolLog

NOW = datetime(2026, 11, 9, 20, tzinfo=KST)  # Monday session, after 18:00 -> analyse today
DART_SECRET = "DARTSECRET123"
PW_SECRET = "PWSECRET456"
PEER_SYSTEM_START = "너는 한국 주식시장 애널리스트다"


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("KRTA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("KRTA_QUICK_MODEL", "quick-model")
    monkeypatch.setenv("KRTA_DEEP_MODEL", "deep-model")
    monkeypatch.setenv("KRTA_DEBATE_ROUNDS", "1")
    monkeypatch.setenv("KRTA_RISK_ROUNDS", "1")
    return tmp_path


class Responder:
    """Peer finder gets one valid peer; every agent answers like test_graph's Script.

    With `echo`, analyst reports repeat their whole prompt (what a leaky model could do)."""

    def __init__(self, echo=False):
        self.script = Script()
        self.echo = echo

    def __call__(self, system, user):
        if system.startswith(PEER_SYSTEM_START):
            return "SK하이닉스"
        answer = self.script(system, user)
        if self.echo and role_of(system) in ANALYST_TITLES:
            return f"{answer}\n\n{user}"
        return answer


def run_main(argv, *, src=None, llm=None, now=NOW):
    seen = {}

    def sources_factory(settings, redactor, toollog, resolved, cal):
        seen["redactor"], seen["resolved"] = redactor, resolved
        return src or sources()

    fake = llm or FakeLLM(Responder())
    code = cli.main(
        argv, now=lambda: now, sources_factory=sources_factory,
        llm_factory=lambda settings, toollog: fake,
    )
    return code, fake, seen


def all_text_under(root: Path) -> str:
    return "\n".join(
        p.read_text(encoding="utf-8", errors="replace") for p in root.rglob("*") if p.is_file()
    )


# --------------------------------------------------------------------------- success


def test_e2e_writes_reports_and_prints_dir(env, capsys):
    code, llm, seen = run_main(["--ticker", "005930", "--horizon", "5d"])
    out, err = capsys.readouterr()
    assert code == 0, err
    run_dir = env / "logs-kr" / "005930.KS" / "2026-11-09" / "5d"
    assert out.splitlines() == [f"REPORT_DIR={run_dir}/reports"]
    assert out.endswith("\n")
    assert sorted(p.name for p in (run_dir / "reports").iterdir()) == sorted(REPORT_FILES)
    assert len(REPORT_FILES) == 10
    decision = json.loads((run_dir / "decision.json").read_text(encoding="utf-8"))
    assert (decision["code"], decision["date"], decision["horizon"]) == ("005930", "2026-11-09", "5d")
    assert decision["rating"] == "OVERWEIGHT" and decision["unconfirmed"] is False
    # The tool log was finalized into the run dir; no temp run dir is left behind.
    log = (run_dir / "message_tool.log").read_text(encoding="utf-8")
    assert "krx_openapi.base_info" in log
    assert not [p for p in (env / cli.TMP_DIR).iterdir()]
    # Dateless run: the resolved date is the default one, so no past-date note.
    final = (run_dir / "reports" / "final_trade_decision.md").read_text(encoding="utf-8")
    assert "과거 날짜 분석" not in final
    assert seen["resolved"] == ResolvedDate(NOW.date(), False, None)
    # The decision is recorded in memory.
    rows = Memory(memory_path(env)).entries()
    assert [(r["code"], r["date"], r["horizon"]) for r in rows] == [("005930", "2026-11-09", "5d")]


def test_e2e_explicit_past_date_gets_note(env, capsys):
    code, _, _ = run_main(["--ticker=005930.KS", "--date=2026-11-06", "--horizon=3m"])
    out, err = capsys.readouterr()
    assert code == 0, err
    run_dir = env / "logs-kr" / "005930.KS" / "2026-11-06" / "3m"
    assert out.splitlines() == [f"REPORT_DIR={run_dir}/reports"]
    final = (run_dir / "reports" / "final_trade_decision.md").read_text(encoding="utf-8")
    assert "과거 날짜 분석" in final


def test_unconfirmed_run_passes_resolved_date_to_factory(env, capsys):
    intraday = datetime(2026, 11, 9, 11, tzinfo=KST)
    code, _, seen = run_main(["--ticker", "005930", "--date", "2026-11-09", "--horizon", "5d"],
                             now=intraday)
    assert code == 0, capsys.readouterr().err
    assert seen["resolved"] == ResolvedDate(intraday.date(), True, None)
    rows = Memory(memory_path(env)).entries()
    assert rows[0]["unconfirmed"] is True


LATE_NIGHT = datetime(2026, 11, 10, 1, 42, tzinfo=KST)  # default date: Monday 2026-11-09


class UnpublishedKrx(FakeKrx):
    """daily_trades is empty (not yet published) for the given days."""

    def __init__(self, *empty_days):
        super().__init__()
        self.empty = set(empty_days)

    def daily_trades(self, market, d):
        if d in self.empty:
            self.calls.append(("daily_trades", market, d))
            return []
        return super().daily_trades(market, d)


def test_default_date_steps_back_to_latest_published_session(env, capsys):
    krx = UnpublishedKrx(date(2026, 11, 9))
    code, _, _ = run_main(["--ticker", "005930", "--horizon", "5d"], src=sources(krx=krx),
                          now=LATE_NIGHT)
    out, err = capsys.readouterr()
    assert code == 0, err
    run_dir = env / "logs-kr" / "005930.KS" / "2026-11-06" / "5d"
    assert out.splitlines() == [f"REPORT_DIR={run_dir}/reports"]
    decision = json.loads((run_dir / "decision.json").read_text(encoding="utf-8"))
    assert decision["date"] == "2026-11-06" and decision["unconfirmed"] is False
    final = (run_dir / "reports" / "final_trade_decision.md").read_text(encoding="utf-8")
    assert "분석일 2026-11-09 시세 미공개로 직전 거래일 2026-11-06 기준" in final
    # Still the latest analysable day, so not flagged as a past-date analysis.
    assert "과거 날짜 분석" not in final
    assert [r["date"] for r in Memory(memory_path(env)).entries()] == ["2026-11-06"]
    # The probe asks only the code's market; nothing is collected for the unpublished day.
    assert ("base_info", "KS", date(2026, 11, 9)) not in krx.calls
    assert not (env / "logs-kr" / "005930.KS" / "2026-11-09").exists()


def test_default_date_probe_uses_suffix_market(env, capsys):
    krx = UnpublishedKrx(date(2026, 11, 9))
    code, _, _ = run_main(["--ticker", "035720.KQ", "--horizon", "5d"], src=sources(krx=krx),
                          now=LATE_NIGHT)
    assert code == 0, capsys.readouterr().err
    probes = [c for c in krx.calls if c[0] == "daily_trades" and c[2] == date(2026, 11, 9)]
    assert probes[0] == ("daily_trades", "KQ", date(2026, 11, 9))
    assert (env / "logs-kr" / "035720.KQ" / "2026-11-06").is_dir()


def test_default_date_gives_up_after_five_steps_exit_1(env, capsys):
    # 11-09 plus the five sessions before it are all unpublished.
    empty = [date(2026, 11, 9), *CAL.sessions_between(date(2026, 11, 2), date(2026, 11, 6))]
    assert len(empty) == 6
    krx = UnpublishedKrx(*empty)
    code, _, _ = run_main(["--ticker", "005930", "--horizon", "5d"], src=sources(krx=krx),
                          now=LATE_NIGHT)
    out, err = capsys.readouterr()
    assert (code, out) == (1, "")
    assert len(err.splitlines()) == 1 and "시세 미공개" in err and "2026-11-02" in err
    probed = sorted({c[2] for c in krx.calls if c[0] == "daily_trades"})
    assert probed == sorted(empty)  # exactly D and five previous sessions
    assert not (env / "logs-kr").exists()


def test_default_date_five_steps_back_is_still_accepted(env, capsys):
    # D and four previous sessions unpublished: the fifth step back (11-02) is used.
    krx = UnpublishedKrx(date(2026, 11, 9), *CAL.sessions_between(date(2026, 11, 3), date(2026, 11, 6)))
    code, _, _ = run_main(["--ticker", "005930", "--horizon", "5d"], src=sources(krx=krx),
                          now=LATE_NIGHT)
    assert code == 0, capsys.readouterr().err
    assert (env / "logs-kr" / "005930.KS" / "2026-11-02").is_dir()


def test_explicit_date_unpublished_is_kept_but_unconfirmed(env, capsys):
    krx = UnpublishedKrx(date(2026, 11, 9))
    code, _, _ = run_main(["--ticker", "005930", "--date", "2026-11-09", "--horizon", "5d"],
                          src=sources(krx=krx), now=LATE_NIGHT)
    out, err = capsys.readouterr()
    assert code == 0, err
    run_dir = env / "logs-kr" / "005930.KS" / "2026-11-09" / "5d"
    assert out.splitlines() == [f"REPORT_DIR={run_dir}/reports"]
    decision = json.loads((run_dir / "decision.json").read_text(encoding="utf-8"))
    assert decision["date"] == "2026-11-09" and decision["unconfirmed"] is True
    final = (run_dir / "reports" / "final_trade_decision.md").read_text(encoding="utf-8")
    assert "분석일 2026-11-09 시세 미공개(빈 응답)" in final
    assert Memory(memory_path(env)).entries()[0]["unconfirmed"] is True


def test_explicit_holiday_note_is_combined_with_unpublished_note(env, capsys):
    krx = UnpublishedKrx(date(2026, 11, 6))
    code, _, _ = run_main(["--ticker", "005930", "--date", "2026-11-08", "--horizon", "5d"],
                          src=sources(krx=krx), now=LATE_NIGHT)
    assert code == 0, capsys.readouterr().err
    final = (env / "logs-kr" / "005930.KS" / "2026-11-06" / "5d" / "reports"
             / "final_trade_decision.md").read_text(encoding="utf-8")
    assert "요청일 2026-11-08은 휴장일" in final and "분석일 2026-11-06 시세 미공개(빈 응답)" in final


def final_md(env, day, ticker="005930.KS"):
    return (env / "logs-kr" / ticker / day / "5d" / "reports"
            / "final_trade_decision.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("date_args", [[], ["--date", "2026-11-09"]])
def test_probe_failure_keeps_date_with_note(env, capsys, date_args):
    krx = FakeKrx(fail_trades=True)
    code, _, _ = run_main(["--ticker", "005930", "--horizon", "5d", *date_args],
                          src=sources(krx=krx), now=LATE_NIGHT)
    assert code == 0, capsys.readouterr().err
    final = final_md(env, "2026-11-09")
    assert "분석일 2026-11-09 시세 공개 여부 확인 실패(HTTP 503: sto/stk_bydd_trd)" in final
    assert "시세 미공개" not in final


def test_probe_failure_while_stepping_back_keeps_both_notes(env, capsys):
    class Flaky(UnpublishedKrx):
        def daily_trades(self, market, d):
            if d == date(2026, 11, 6):
                raise FetchError("krx_openapi", "HTTP 503: sto/stk_bydd_trd", status=503)
            return super().daily_trades(market, d)

    code, _, _ = run_main(["--ticker", "005930", "--horizon", "5d"],
                          src=sources(krx=Flaky(date(2026, 11, 9))), now=LATE_NIGHT)
    assert code == 0, capsys.readouterr().err
    final = final_md(env, "2026-11-06")
    assert "분석일 2026-11-09 시세 미공개로 직전 거래일 2026-11-06 기준" in final
    assert "분석일 2026-11-06 시세 공개 여부 확인 실패(HTTP 503" in final


def test_suffixless_ticker_needs_both_markets_published(env, capsys):
    class KqLate(FakeKrx):
        def daily_trades(self, market, d):
            if market == "KQ" and d == date(2026, 11, 9):
                self.calls.append(("daily_trades", market, d))
                return []
            return super().daily_trades(market, d)

    code, _, _ = run_main(["--ticker", "005930", "--horizon", "5d"], src=sources(krx=KqLate()),
                          now=LATE_NIGHT)
    assert code == 0, capsys.readouterr().err
    assert "분석일 2026-11-09 시세 미공개로 직전 거래일 2026-11-06 기준" in final_md(env, "2026-11-06")


def test_suffixed_ticker_probes_only_its_market(env, capsys):
    class KqLate(FakeKrx):
        def daily_trades(self, market, d):
            if market == "KQ" and d == date(2026, 11, 9):
                return []
            return super().daily_trades(market, d)

    code, _, _ = run_main(["--ticker", "005930.KS", "--horizon", "5d"],
                          src=sources(krx=KqLate()), now=LATE_NIGHT)
    assert code == 0, capsys.readouterr().err
    assert (env / "logs-kr" / "005930.KS" / "2026-11-09").is_dir()


def test_explicit_intraday_run_gets_no_unpublished_note(env, capsys):
    intraday = datetime(2026, 11, 9, 11, tzinfo=KST)
    krx = UnpublishedKrx(date(2026, 11, 9))
    code, _, _ = run_main(["--ticker", "005930", "--date", "2026-11-09", "--horizon", "5d"],
                          src=sources(krx=krx), now=intraday)
    assert code == 0, capsys.readouterr().err
    final = final_md(env, "2026-11-09")
    assert "시세 미공개" not in final and "확인 실패" not in final
    assert Memory(memory_path(env)).entries()[0]["unconfirmed"] is True


def test_memory_is_settled_and_lessons_reach_pm(env, capsys):
    mem = Memory(memory_path(env))
    mem.path.parent.mkdir(parents=True)
    pending = {
        "code": "005930", "market": "KS", "date": "2026-10-20", "horizon": "5d",
        "trading_days": 5, "rating": "BUY", "target_price": 200, "stop_loss": 50,
        "unconfirmed": False, "settled": None,
    }
    scored = {
        **pending, "date": "2026-10-15", "rating": "SELL",
        "settled": {"return": -0.02, "bench_return": 0.01, "excess": -0.03, "hit_target": False,
                    "hit_stop": None, "window_end": "2026-10-22", "settled_on": "2026-10-23"},
    }
    mem.path.write_text(json.dumps(pending) + "\n" + json.dumps(scored) + "\n", encoding="utf-8")
    code, llm, _ = run_main(["--ticker", "005930", "--horizon", "5d"])
    assert code == 0, capsys.readouterr().err
    old = next(r for r in mem.entries() if r["date"] == "2026-10-20")
    assert old["settled"] is not None and old["settled"]["settled_on"] == "2026-11-09"
    pm_prompts = [u for s, u in llm.calls if not s.startswith(PEER_SYSTEM_START)
                  and role_of(s) == PORTFOLIO_MANAGER]
    assert pm_prompts and "2026-10-15 5d 매도(SELL)" in pm_prompts[-1]
    # Scored during this very run (settled_on == analysis date): not yet a lesson.
    assert "2026-10-20 5d" not in pm_prompts[-1]


def test_given_up_memory_entries_reported_on_stderr(env, capsys):
    mem = Memory(memory_path(env))
    mem.path.parent.mkdir(parents=True)
    mem.path.write_text(json.dumps({
        "code": "123456", "market": "KS", "date": "2026-10-20", "horizon": "5d",
        "trading_days": 5, "rating": "BUY", "target_price": None, "stop_loss": None,
        "unconfirmed": False, "settled": None, "attempts": MAX_ATTEMPTS,
    }) + "\n", encoding="utf-8")
    code, _, _ = run_main(["--ticker", "005930", "--horizon", "5d"])
    out, err = capsys.readouterr()
    assert code == 0, err
    assert len(out.splitlines()) == 1
    assert [line for line in err.splitlines() if "메모리 채점 포기 1건" in line]


def test_stdout_cannot_be_spoofed_by_library_prints(env, capsys, monkeypatch):
    monkeypatch.setenv("DART_API_KEY", DART_SECRET)

    class NoisyNews(FakeNews):
        def search(self, query, as_of, days=14):
            print("REPORT_DIR=/home/appuser/.krta/logs-kr/005930.KS/2026-11-09/5d/reports")
            print(f"leak {DART_SECRET}")
            return super().search(query, as_of, days)

    code, _, _ = run_main(["--ticker", "005930", "--horizon", "5d"], src=sources(news=NoisyNews()))
    out, err = capsys.readouterr()
    assert code == 0, err
    assert len(out.splitlines()) == 1 and out.startswith(f"REPORT_DIR={env}/logs-kr/")
    assert not any(line.startswith("REPORT_DIR=") for line in err.splitlines())
    assert DART_SECRET not in err and "leak ***" in err


# --------------------------------------------------------------------------- secrets


def test_e2e_secrets_never_in_outputs(env, capsys, monkeypatch):
    monkeypatch.setenv("DART_API_KEY", DART_SECRET)
    monkeypatch.setenv("KRX_PW", PW_SECRET)
    url = f"https://opendart.fss.or.kr/api/company.json?crtfc_key={DART_SECRET}"

    class LeakyDart(FakeDart):
        def company(self, corp):
            raise FetchError("dart", f"HTTP 500: GET {url}", status=500)

    class LeakyNews(FakeNews):
        def search(self, query, as_of, days=14):
            items = super().search(query, as_of, days)
            if "증시" in query:
                raise FetchError("google_news", f"login {PW_SECRET} rejected")
            return items

    code, llm, seen = run_main(
        ["--ticker", "005930", "--horizon", "5d"],
        src=sources(dart=LeakyDart(), news=LeakyNews()), llm=FakeLLM(Responder(echo=True)),
    )
    out, err = capsys.readouterr()
    assert code == 0, err
    # The redactor handed to the sources knows both secrets.
    assert DART_SECRET not in seen["redactor"].redact(f"{DART_SECRET} {PW_SECRET}")
    assert PW_SECRET not in seen["redactor"].redact(PW_SECRET)
    # Nor do the prompts carry them, so the model cannot echo them into reports.
    prompts = "\n".join(s + u for s, u in llm.calls)
    for secret in (DART_SECRET, PW_SECRET):
        assert secret not in out and secret not in err
        assert secret not in prompts
        assert secret not in all_text_under(env)
    final_dir = env / "logs-kr" / "005930.KS" / "2026-11-09" / "5d"
    assert "***" in (final_dir / "message_tool.log").read_text(encoding="utf-8")


def test_runtime_token_registered_with_shared_redactor_is_masked(env, capsys):
    token = "TOSSTOKEN-abcdef-987654"

    class TokenToss(FakeToss):
        def __init__(self, redactor):
            self.redactor = redactor

        def investor_flow(self, code, start, end):
            self.redactor.add(token)  # what TossApi does once it is issued a token
            print(f"debug Authorization: Bearer {token}")
            raise FetchError("toss", f"HTTP 500 with Bearer {token}", status=500)

    def factory(settings, redactor, toollog, resolved, cal):
        return sources(toss=TokenToss(redactor))

    code = cli.main(["--ticker", "005930", "--horizon", "5d"], now=lambda: NOW,
                    sources_factory=factory, llm_factory=lambda s, t: FakeLLM(Responder(echo=True)))
    out, err = capsys.readouterr()
    assert code == 0, err
    assert token not in out and token not in err and "Bearer ***" in err
    assert token not in all_text_under(env)


def test_key_error_exit_3_redacted_single_line(env, capsys, monkeypatch):
    monkeypatch.setenv("DART_API_KEY", DART_SECRET)

    class BadKeyDart(FakeDart):
        def corp_code(self, stock_code):
            raise DartKeyError(f"DART 키 문제(상태 010) key={DART_SECRET}")

    code, _, _ = run_main(["--ticker", "005930", "--horizon", "5d"], src=sources(dart=BadKeyDart()))
    out, err = capsys.readouterr()
    assert code == 3
    assert out == ""
    assert len(err.splitlines()) == 1 and "DART 키 문제" in err
    assert DART_SECRET not in err and DART_SECRET not in all_text_under(env)


def test_krx_auth_error_on_listing_exit_3(env, capsys):
    class NoAuthKrx(FakeKrx):
        def base_info(self, market, d):
            raise KrxAuthError("KRX Open API 인증 실패")

    code, _, _ = run_main(["--ticker", "005930", "--horizon", "5d"], src=sources(krx=NoAuthKrx()))
    out, err = capsys.readouterr()
    assert (code, out) == (3, "")
    assert len(err.splitlines()) == 1 and "KRX Open API 인증 실패" in err


# --------------------------------------------------------------------------- input errors


def test_bad_horizon_exit_2(env, capsys):
    def no_sources(*a):
        raise AssertionError("sources must not be built for bad input")

    code = cli.main(["--ticker", "005930", "--horizon", "5x"], now=lambda: NOW,
                    sources_factory=no_sources, llm_factory=no_sources)
    out, err = capsys.readouterr()
    assert (code, out) == (2, "")
    assert len(err.splitlines()) == 1 and "기간 형식 오류" in err


def test_etf_exit_2_out_of_scope(env, capsys):
    code, _, _ = run_main(["--ticker", "069500", "--horizon", "5d"])
    out, err = capsys.readouterr()
    assert (code, out) == (2, "")
    assert len(err.splitlines()) == 1 and "범위 밖" in err
    assert not (env / "logs-kr").exists()


@pytest.mark.parametrize("argv, needle", [
    (["--horizon", "5d"], "--ticker"),
    (["--ticker", "005930"], "--horizon"),
    (["--ticker", "5930", "--horizon", "5d"], "종목코드 형식 오류"),
    (["--ticker", "005930", "--horizon", "5d", "--date", "2026/11/09"], "날짜 형식 오류"),
    (["--ticker", "005930", "--horizon", "5d", "--date", "2026-11-10"], "미래 날짜"),
])
def test_input_errors_exit_2(env, capsys, argv, needle):
    code, _, _ = run_main(argv)
    out, err = capsys.readouterr()
    assert (code, out) == (2, "")
    assert len(err.splitlines()) == 1 and needle in err


def test_suffix_mismatch_exit_2(env, capsys):
    code, _, _ = run_main(["--ticker", "005930.KQ", "--horizon", "5d"])
    out, err = capsys.readouterr()
    assert (code, out) == (2, "")
    assert len(err.splitlines()) == 1 and "입력 오류" in err and "접미사" in err


def test_data_format_error_inside_collect_exit_1(env, capsys):
    class BadIndex(FakeKrx):
        def index_daily(self, market, d):
            return [{"IDX_NM": "전기전자", "CLSPRC_IDX": "N/A?"}]

    code, _, _ = run_main(["--ticker", "005930", "--horizon", "5d"], src=sources(krx=BadIndex()))
    out, err = capsys.readouterr()
    assert (code, out) == (1, "")
    assert len(err.splitlines()) == 1 and "ValueError" in err and "입력 오류" not in err


def test_missing_settings_exit_2(env, capsys, monkeypatch):
    monkeypatch.delenv("KRTA_DEEP_MODEL")
    code, _, _ = run_main(["--ticker", "005930", "--horizon", "5d"])
    assert code == 2 and "KRTA_DEEP_MODEL" in capsys.readouterr().err


def test_unexpected_error_exit_1_after_collect(env, capsys):
    def broken(system, user):
        if system.startswith(PEER_SYSTEM_START):
            return "SK하이닉스"
        raise ValueError("model output did not parse")  # not an input error once data is in

    code, _, _ = run_main(["--ticker", "005930", "--horizon", "5d"], llm=FakeLLM(broken))
    out, err = capsys.readouterr()
    assert (code, out) == (1, "")
    assert len(err.splitlines()) == 1 and "ValueError" in err
    assert "message_tool.log" in err  # points at the kept temp log for diagnosis


def test_pm_without_structured_output_exit_1(env, capsys):
    responder = Responder()
    responder.script = Script(["null"])
    code, _, _ = run_main(["--ticker", "005930", "--horizon", "5d"], llm=FakeLLM(responder))
    out, err = capsys.readouterr()
    assert (code, out) == (1, "")
    assert len(err.splitlines()) == 1 and "RuntimeError" in err and "PM 구조화 출력 없음" in err


# --------------------------------------------------------------------------- default factories


def test_default_sources_wiring(env, monkeypatch):
    for name, value in (("KRX_API_KEY", "krx-dummy"), ("DART_API_KEY", "dart-dummy"),
                        ("TOSS_CLIENT_ID", "toss-id-dummy"),
                        ("TOSS_CLIENT_SECRET", "toss-secret-dummy")):
        monkeypatch.setenv(name, value)
    settings = cli.Settings.from_env()
    red = Redactor.from_env()
    log = ToolLog(env / "x.log", red)
    src = cli.default_sources(settings, red, log, ResolvedDate(NOW.date(), True, None), CAL)
    assert src.krx._http is src.dart._http
    assert src.toss._http is not src.krx._http  # own client and pacing
    assert src.toss._http._min_interval == toss.MIN_INTERVAL
    assert src.toss._redactor is red and src.toss._http._redactor is red
    assert src.toss._http._toollog is log
    assert src.toss.available()
    assert src.portal is None
    assert src.krx._unconfirmed_date == NOW.date()
    assert src.cal is CAL
    monkeypatch.delenv("TOSS_CLIENT_SECRET")
    confirmed = cli.default_sources(settings, red, log, ResolvedDate(NOW.date(), False, None), CAL)
    assert confirmed.krx._unconfirmed_date is None
    assert not confirmed.toss.available()  # no keys: flow items become missing lines


@pytest.mark.parametrize("missing", ["KRX_API_KEY", "DART_API_KEY"])
def test_default_sources_missing_key_exit_3(env, capsys, monkeypatch, missing):
    monkeypatch.setenv("KRX_API_KEY", "krx-dummy")
    monkeypatch.setenv("DART_API_KEY", "dart-dummy")
    monkeypatch.delenv(missing)
    code = cli.main(["--ticker", "005930", "--horizon", "5d"], now=lambda: NOW,
                    llm_factory=lambda s, t: FakeLLM(Responder()))
    out, err = capsys.readouterr()
    assert (code, out) == (3, "")
    assert missing in err and len(err.splitlines()) == 1
    assert not list((env / cli.TMP_DIR).iterdir())  # nothing logged yet: no empty run dir


def test_default_llm_requires_anthropic_key(env, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(cli.MissingKeyError, match="ANTHROPIC_API_KEY"):
        cli.default_llm(cli.Settings.from_env(), ToolLog(env / "x.log", Redactor([])))


def test_anthropic_chat_has_request_timeout(monkeypatch):
    from krta.llm import REQUEST_TIMEOUT, _chat_anthropic

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy")
    chat = _chat_anthropic("claude-x", 2)
    assert chat.default_request_timeout == REQUEST_TIMEOUT
    assert chat.max_retries == 2
