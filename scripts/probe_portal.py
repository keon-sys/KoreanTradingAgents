"""Cluster check for krta.data.krx_portal: logs in with KRX_ID/KRX_PW from env and fetches
삼성전자(005930) for the last 5 trading days.

Prints ONLY row counts, key lists, booleans and redacted status/reason strings: never
credentials, cookies, headers or market values. Always exits 0 so the Pod reaches Succeeded.
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime, timedelta, timezone

from krta.data import krx_portal
from krta.data.krx_portal import KrxPortal, PortalUnavailable
from krta.net import HttpClient
from krta.redact import Redactor

CODE = "005930"
DAYS = 5


def main() -> int:
    redactor = Redactor.from_env()
    http = HttpClient(redactor, None, min_interval=1.0)
    portal = KrxPortal(http, os.environ.get("KRX_ID"), os.environ.get("KRX_PW"))
    print(f"available={portal.available()}")

    end = datetime.now(timezone(timedelta(hours=9))).date()
    start = end - timedelta(days=14)
    calls = [
        ("investor_flow", lambda: portal.investor_flow(CODE, start, end)),
        ("shorting", lambda: portal.shorting(CODE, start, end)),
        ("foreign_holding", lambda: portal.foreign_holding(CODE, start, end)),
        ("investor_flow_market", lambda: portal.investor_flow_market("KS", start, end)),
    ]
    flow: list[dict] = []
    for name, call in calls:
        try:
            rows = call()[-DAYS:]
        except PortalUnavailable as exc:
            print(f"{name}: UNAVAILABLE reason={redactor.redact(exc.reason)}")
            continue
        except Exception as exc:  # noqa: BLE001 - report type only
            print(f"{name}: ERROR type={type(exc).__name__}")
            continue
        keys = sorted(rows[0]) if rows else []
        nones = sum(v is None for r in rows for v in r.values())
        dates_ok = all(isinstance(r["date"], date) and start <= r["date"] <= end for r in rows)
        print(f"{name}: rows={len(rows)} keys={keys} none_values={nones} dates_in_range={dates_ok}")
        if name == "investor_flow":
            flow = rows
            zero = all(r["외국인"] + r["기관합계"] + r["기타법인"] + r["개인"] == 0 for r in rows)
            ints = all(type(r[k]) is int for r in rows for k in r if k != "date")
            print(f"{name}: net_sum_zero={zero} all_int={ints}")
    if flow:
        cross_check(portal, start, end, flow)
    return 0


def cross_check(portal: KrxPortal, start: date, end: date, flow: list[dict]) -> None:
    """Compare the detailed view's derived totals with the summary view (MDCSTAT02302):
    TRDVAL1 기관합계, TRDVAL2 기타법인, TRDVAL3 개인, TRDVAL4 외국인합계."""
    try:
        isin = portal._isin_of(CODE)
        rows = portal._query(
            "dbms/MDC/STAT/standard/MDCSTAT02302", "output",
            strtDd=start.strftime("%Y%m%d"), endDd=end.strftime("%Y%m%d"), isuCd=isin,
            inqTpCd=2, trdVolVal=2, askBid=3,
        )
    except PortalUnavailable as exc:
        print(f"cross_check: UNAVAILABLE reason={exc.reason}")
        return
    summary = {}
    for r in rows:
        d = datetime.strptime(r["TRD_DD"], "%Y/%m/%d").date()
        summary[d] = [krx_portal.to_int(r[f"TRDVAL{i}"]) or 0 for i in (1, 2, 3, 4)]
    names = ["기관합계", "기타법인", "개인", "외국인"]
    matched = {n: all(summary.get(r["date"], [None] * 4)[i] == r[n] for r in flow)
               for i, n in enumerate(names)}
    print(f"cross_check(summary vs detail): days={len(flow)} match={matched}")


if __name__ == "__main__":
    sys.exit(main())
