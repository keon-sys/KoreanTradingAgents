"""Connectivity check for krta.data.toss, run in the environment that will run krta (its
public IP must be on the Toss allow list): issues one token with TOSS_CLIENT_ID /
TOSS_CLIENT_SECRET from env and fetches 삼성전자(005930) and the KOSPI/KOSDAQ market flows
for the last 5 trading days of every method.

Prints ONLY row counts, key lists (normalized keys and the raw field names Toss sent),
None counts per key, the latest row date and redacted reasons: never credentials, the
token, headers or market values. Stops at the first auth/IP failure (401/403) instead of
retrying. Always exits 0; read the output, not the exit code.

Note: issuing a token revokes any other token for the same client (one valid token per
client), so do not run it while krta is running with the same keys.

Usage: `uv run python scripts/probe_toss.py`
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime, timedelta, timezone

from krta.data import toss
from krta.data.toss import TossApi, TossUnavailable
from krta.net import HttpClient
from krta.redact import Redactor

CODE = "005930"
DAYS = 5
_STOP = ("401", "403")


def raw_keys(api: TossApi, path: str, start: date, end: date, extra: dict) -> list[str]:
    """Field names (top level and one level down) of the raw records; no values."""
    names: set[str] = set()
    for r in api._records(path, start, end, extra):
        if not isinstance(r, dict):
            continue
        for k, v in r.items():
            names.add(k)
            if isinstance(v, dict):
                names.update(f"{k}.{kk}" for kk in v)
    return sorted(names)


def main() -> int:
    redactor = Redactor.from_env()
    http = HttpClient(redactor, None, min_interval=toss.MIN_INTERVAL, max_retries=1)
    api = TossApi(http, redactor, os.environ.get("TOSS_CLIENT_ID"),
                  os.environ.get("TOSS_CLIENT_SECRET"))
    print(f"available={api.available()}")
    if not api.available():
        return 0

    end = datetime.now(timezone(timedelta(hours=9))).date()
    start = end - timedelta(days=14)
    stock = f"/api/v1/stocks/{CODE}"
    calls = [
        ("investor_flow", lambda: api.investor_flow(CODE, start, end), f"{stock}/investor-trading", {}),
        ("foreign_holding", lambda: api.foreign_holding(CODE, start, end), None, {}),
        ("shorting", lambda: api.shorting(CODE, start, end), f"{stock}/short-selling", {}),
        ("program_trading", lambda: api.program_trading(CODE, start, end), f"{stock}/program-trades", {}),
        ("credit_balance", lambda: api.credit_balance(CODE, start, end), f"{stock}/credit-trades", {}),
        ("securities_lending", lambda: api.securities_lending(CODE, start, end),
         f"{stock}/securities-lending", {}),
        ("investor_flow_market KS", lambda: api.investor_flow_market("KS", start, end),
         "/api/v1/market-indicators/KOSPI/investor-trading", {"interval": "1d"}),
        ("investor_flow_market KQ", lambda: api.investor_flow_market("KQ", start, end),
         "/api/v1/market-indicators/KOSDAQ/investor-trading", {"interval": "1d"}),
    ]
    for name, call, path, extra in calls:
        try:
            rows = call()[-DAYS:]
        except TossUnavailable as exc:
            reason = redactor.redact(exc.reason)
            print(f"{name}: UNAVAILABLE reason={reason}")
            if any(code in reason for code in _STOP):
                print("stop: auth/IP failure, not retrying")
                return 0
            continue
        except Exception as exc:  # noqa: BLE001 - report type only
            print(f"{name}: ERROR type={type(exc).__name__}")
            continue
        keys = list(rows[0]) if rows else []
        nones = {k: sum(r[k] is None for r in rows) for k in keys if k != "date"}
        dates_ok = all(isinstance(r["date"], date) and start <= r["date"] <= end for r in rows)
        latest = rows[-1]["date"].isoformat() if rows else None
        print(f"{name}: rows={len(rows)} latest={latest} dates_in_range={dates_ok}")
        print(f"{name}: keys={keys}")
        print(f"{name}: none_per_key={nones}")
        if path is not None:
            print(f"{name}: raw_fields={raw_keys(api, path, start, end, extra)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
