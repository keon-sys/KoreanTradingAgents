# KoreanTradingAgents (krta)

한국 주식(KOSPI `.KS`, KOSDAQ `.KQ`) 한 종목을 여러 LLM 에이전트가 나눠 분석하고, 한국어 리포트와 최종 판단(5단계 rating, 목표가, 손절가)을 내는 CLI.

> 투자 조언이 아니다. LLM 출력은 실행마다 달라질 수 있고, 투자 판단과 책임은 사용자에게 있다. 리포트의 매수 비중·손절가는 보유 현황을 모르고 쓴 일반 제안이다.

## 무엇을 하나

```
데이터 수집(코드) → 분석가 6명(병렬) → 강세/약세 토론 → 리서치 매니저 → 트레이더 → 리스크 토론(공격·보수·중립) → 포트폴리오 매니저
```

숫자는 모두 코드가 미리 계산해 섹션으로 넘기고, 에이전트는 그 섹션만 읽는다. 가져오지 못한 항목은 `데이터 없음: ...` 줄로 표시되고 실행은 계속된다. 분석가는 빠른 모델(`KRTA_QUICK_MODEL`), 토론부터 최종 판단까지는 깊은 모델(`KRTA_DEEP_MODEL`)을 쓴다.

| 분석가 | 보는 것 | 데이터 |
|---|---|---|
| 시장·지수 | 지수 추세, 시장 전체 외국인·기관 순매수, 환율, 전날 미국장, 상해종합, 반도체 지수, 시장 정책 뉴스, 지수 대비 상대 강도 | Yahoo Finance, 토스증권, Google News |
| 수급 | 투자자별 순매수(일별·누적), 주체별 추정 평균 단가, 외국인 보유·한도 소진율, 공매도 거래, 대차·신용잔고, 프로그램 매매 | 토스증권, Yahoo Finance |
| 테마·재료 | 거래대금 순위, 테마 자금 쏠림, 테마 내 위치(대장주/후발주), 업종 지수 등락, 뉴스 재료 | KRX Open API, Google News, LLM 추정 동료 종목 |
| 차트·거래량 | 이동평균, 거래량 급증, 지지·저항, 변동성, 상·하한가 근접 이력 | Yahoo Finance 일봉 |
| 공시 | 최근 공시, 주요사항보고서(증자·CB/BW·자사주·합병·분할), 지분 변동, 잠정실적, 정정공시 | OpenDART |
| 펀더멘털·밸류 | 재무 추이, PER·PBR·배당, 동료 대비 밸류, 지배구조·주주환원 | OpenDART, KRX Open API(시가총액) |

이벤트 캘린더(선물·옵션 만기, KOSPI200·MSCI 정기변경, FOMC·금통위, 휴장일)도 코드가 계산한다. FOMC·금통위 날짜는 `src/krta/data/schedules.toml`에 공식 출처와 확인일을 함께 적어 두며, 연도가 바뀌면 직접 갱신해야 한다.

같은 종목의 과거 판단은 기간이 지나면 실제 주가로 채점되고, 다음 실행에 짧은 교훈으로 들어간다.

## 데이터 출처와 한계

- **공매도 잔고 없음.** 토스증권 API는 공매도 거래량·거래대금만 주고 잔고는 주지 않는다.
- **종목별 수급은 수량(주) 기준.** 종목별 투자자 순매수·프로그램 매매·신용·대차는 주식 수다. 원 금액은 시장 전체 투자자별 순매수에만 있다. 외국인은 등록외국인 기준이다. 당일 장중에는 외국인·기관합계 잠정치만 있다.
- **뉴스는 헤드라인 위주.** Google News RSS(한국어판)의 제목과 언론사만 쓴다. 기사 본문은 읽지 않는다.
- **동료 종목은 추정.** LLM이 고른 이름을 KRX 상장 목록과 대조해 통과한 것만 쓴다. 업종 분류 데이터에서 나온 목록이 아니다.
- **주가는 수정주가가 아님.** Yahoo 일봉을 그대로 쓰므로 액면분할·병합은 급등락처럼 보인다.
- **과거 날짜 분석의 한계.** 재무제표 정정 전 원본이나 당시 상장 목록은 재현하지 않는다.

## 기간(horizon) 규칙

`--horizon`은 `N` + 단위다(N은 1~999).

| 단위 | 거래일 환산 | 예 |
|---|---|---|
| `d` | N | `5d` = 5거래일 |
| `w` | N × 5 | `4w` = 20거래일 |
| `m` | N × 21 | `6m` = 126거래일 |
| `y` | N × 252 | `1y` = 252거래일 |

거래일 수로 구간이 정해지고, 구간에 따라 데이터 조회 기간과 근거 가중치가 달라진다.

| 구간 | 거래일 | 주 근거 | 보조 근거 |
|---|---|---|---|
| 단기 | ≤ 20 | 수급, 테마·재료, 차트·거래량, 시장 | 공시(악재 필터), 밸류(참고) |
| 중기 | ≤ 126 | 6명 고르게 | — |
| 장기 | > 126 | 펀더멘털·밸류, 공시 | 수급·차트(진입 타이밍) |

## 설치·테스트

Python 3.13과 [uv](https://docs.astral.sh/uv/)가 필요하다.

```sh
uv sync
uv run pytest   # 네트워크 차단(--disable-socket) 상태로 실행
```

## 실행

```sh
uv run krta --ticker <코드> --horizon <Nd|Nw|Nm|Ny> [--date YYYY-MM-DD]
uv run krta --ticker 005930 --horizon 1m
uv run krta --ticker 035720.KQ --horizon 5d --date 2026-11-06
```

- `--ticker`: 6자리 종목코드. `.KS`/`.KQ`를 생략하면 상장 목록에서 시장을 찾는다.
- `--date`: 분석일(KST). 휴장일이면 직전 거래일로 바꾼다. 오늘 날짜를 18:00 KST 전에 지정하거나 지정한 날의 KRX 시세가 아직 비어 있으면 "미확정" 분석이 되어 리포트에 경고가 붙고 메모리 채점에서 빠진다.
- `--date`를 생략하면 KRX 시세가 공개된 가장 최근 거래일을 쓴다. 18:00 KST 이후면 오늘, 아니면 직전 거래일에서 시작해 시세가 비어 있으면 최대 5거래일까지 거슬러 올라간다.

## 출력

```
<KRTA_DATA_DIR>/
├── logs-kr/<코드>.<KS|KQ>/<분석일>/<기간>/
│   ├── reports/
│   │   ├── event_calendar.md
│   │   ├── market_report.md
│   │   ├── flow_report.md
│   │   ├── theme_report.md
│   │   ├── technical_report.md
│   │   ├── disclosure_report.md
│   │   ├── fundamentals_report.md
│   │   ├── investment_plan.md          # 리서치 매니저
│   │   ├── trader_investment_plan.md   # 트레이더
│   │   └── final_trade_decision.md     # 포트폴리오 매니저 최종 판단
│   ├── decision.json                   # 최종 판단(구조화)
│   └── message_tool.log                # 원자료·요청·프롬프트 로그(마스킹됨)
├── cache-kr/                           # KRX 응답·DART 고유번호 캐시
├── memory-kr/decisions.jsonl           # 과거 판단과 채점 결과
└── tmp-kr/                             # 실행 중 로그. 실패한 실행의 로그는 여기 남는다
```

- 성공하면 stdout에 `REPORT_DIR=<...>/reports` 한 줄만 마지막에 출력한다. 진행 메시지와 오류는 stderr로 간다(마스킹됨).
- 종료 코드: `0` 성공, `2` 입력 오류(인자, 종목코드, 기간, 날짜, `KRTA_*` 설정값), `3` 키 문제(키 없음, KRX·DART 인증 실패), `1` 그 밖의 실패(시세 미공개 포함).

## 환경변수

| 이름 | 필수 | 기본값 | 용도 |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | 필수 | | LLM 호출 |
| `DART_API_KEY` | 필수 | | OpenDART 공시·재무 |
| `KRX_API_KEY` | 필수 | | KRX Open API 시세·상장 정보·지수 |
| `TOSS_CLIENT_ID`, `TOSS_CLIENT_SECRET` | 선택 | | 토스증권 Open API 수급 데이터. 없으면 수급 항목이 `데이터 없음`으로 남는다 |
| `KRTA_QUICK_MODEL` | 필수 | | 분석가·동료 종목 추정에 쓰는 Anthropic 모델 ID |
| `KRTA_DEEP_MODEL` | 필수 | | 토론·트레이더·리스크·최종 판단에 쓰는 Anthropic 모델 ID |
| `KRTA_DATA_DIR` | 선택 | `~/.krta` | 출력·캐시·메모리 루트 |
| `KRTA_DEBATE_ROUNDS` | 선택 | `1` | 강세/약세 토론 라운드(1 이상) |
| `KRTA_RISK_ROUNDS` | 선택 | `1` | 리스크 토론 라운드(1 이상) |
| `KRTA_LLM_MAX_RETRIES` | 선택 | `6` | LLM 요청 재시도 횟수(0 이상) |

토스증권 Open API를 쓰려면 토스증권 WTS의 설정 > Open API > 허용 IP 관리에서 실행 환경의 공인 IP를 허용 IP로 등록해야 한다(미등록이면 403). 연결 확인은 `uv run python scripts/probe_toss.py`로 할 수 있다. 행 수·필드 이름 같은 요약만 출력하고 키·토큰·시세 값은 출력하지 않는다. 토큰을 새로 발급하므로 같은 키로 krta가 도는 중에는 실행하지 않는다.

## 컨테이너 실행

`Dockerfile`은 `python:3.13-slim`(digest 고정) 멀티스테이지 빌드다. 빌드 컨텍스트는 저장소 루트다. 런타임 의존성은 `uv.lock`에서 내보낸 해시로, 빌드 도구(uv, hatchling)는 `build-requirements.txt`의 해시로 검증 설치한다(`--require-hashes`, `--only-binary :all:`). 엔트리포인트는 `krta`다.

```sh
docker build -t krta .
docker run --rm --read-only --tmpfs /tmp --tmpfs /home/appuser/.cache \
  -v krta-data:/home/appuser/.krta --env-file <키 파일> \
  -e LANGSMITH_TRACING=false -e LANGCHAIN_TRACING_V2=false -e LANGSMITH_TRACING_V2=false \
  krta --ticker 005930 --horizon 1m
```

- uid/gid 1000(`appuser`)으로 돈다. 읽기 전용 루트 파일시스템에서 동작하며, 쓰기가 필요한 곳은 `/tmp`, `/home/appuser/.cache`, `KRTA_DATA_DIR`(이미지 기본 `/home/appuser/.krta`)뿐이다. 데이터 경로는 uid 1000이 쓸 수 있는 볼륨으로 마운트한다.
- LangSmith 추적은 위 세 환경변수로 끈다.
- 키는 이미지에 넣지 않는다. `.dockerignore`가 `.env*`, `*.secret`, `secrets/`, `*.pem`, `*.key`를 빌드 컨텍스트에서 뺀다.

## 보안

- **토스증권 키는 주문 권한도 갖는다.** `krta.data.toss`는 허용 목록에 있는 요청(토큰 발급, 종목별 매매동향 5종, 시장 투자자별 매매대금 조회)만 보낸다. 그 밖의 경로와 계좌 헤더(`X-Tossinvest-Account`)는 보내기 전에 거부한다. 테스트로 이를 확인한다.
- **마스킹.** 키 이름 패턴(`*KEY`, `*TOKEN`, `*SECRET`, `*PASSWORD`, `*_PW`, `*_ID`)에 맞는 환경변수 값과 실행 중 발급된 access token은 로그, 리포트, 예외 메시지, 프롬프트에서 `***`로 가려진다. 토큰은 메모리에만 둔다.
- **키당 토큰 1개.** 토스증권은 클라이언트당 access token 하나만 유효하다. krta가 토큰을 발급하면 같은 키로 다른 프로그램이 받아 둔 토큰은 즉시 무효가 된다.

## 라이선스

Apache License 2.0. `LICENSE` 참고.
