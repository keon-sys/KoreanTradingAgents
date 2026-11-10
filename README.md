# KoreanTradingAgents (krta)

한국 주식(KOSPI `.KS`, KOSDAQ `.KQ`)을 위한 LLM 멀티 에이전트 분석기. 애널리스트 6명 → 강세/약세 토론 → 리서치 매니저 → 트레이더 → 리스크 토론 → 포트폴리오 매니저 순으로 한국어 리포트를 만든다. 구조는 TradingAgents를 참고했지만 코드는 새로 작성했다.

투자 조언이 아니다. LLM 출력은 실행마다 달라질 수 있고, 투자 판단과 책임은 사용자에게 있다.

## 개발

```sh
uv sync
uv run pytest   # 네트워크 차단(--disable-socket) 상태로 실행
```

## 환경변수

키는 환경변수로만 읽는다. 값은 로그·리포트·예외 메시지에 남지 않도록 마스킹된다. 저장소에 값을 넣지 않는다.

| 이름 | 용도 |
|---|---|
| `ANTHROPIC_API_KEY` | LLM 호출 |
| `DART_API_KEY` | OpenDART 공시·재무 |
| `KRX_API_KEY` | KRX Open API |
| `TOSS_CLIENT_ID`, `TOSS_CLIENT_SECRET` | 토스증권 Open API: 투자자별 매매, 외국인 보유, 공매도 거래, 프로그램 매매, 신용·대차 잔고, 시장 전체 투자자별 매매대금. 토스증권 WTS 설정 > Open API > 허용 IP 관리에 집 공인 IP를 등록해야 한다(미등록 시 403) |
| `KRX_ID`, `KRX_PW` | KRX 정보데이터시스템 로그인(평문 로그인이 거부되어 수집 경로에서 쓰지 않음) |
| `KRTA_DATA_DIR` | 출력·캐시 루트 (기본 `~/.krta`) |
| `KRTA_QUICK_MODEL`, `KRTA_DEEP_MODEL` | 사용할 모델 (필수) |
| `KRTA_DEBATE_ROUNDS`, `KRTA_RISK_ROUNDS` | 토론 라운드 수 (기본 1) |
| `KRTA_LLM_MAX_RETRIES` | LLM 재시도 횟수 (기본 6) |

## 배포

배포 설정(이미지 빌드 스크립트, Job 매니페스트, 실행 스크립트, 키 주입 방식)은 이 저장소 밖에서 관리한다. 이 저장소가 제공하는 것은 `Dockerfile`과 아래 실행 계약이다.

- 이미지: `python:3.13-slim`(digest 고정) 멀티스테이지. 빌드 컨텍스트는 저장소 루트. 런타임 의존성은 `uv.lock`에서 내보낸 해시로, 빌드 도구(uv, hatchling)는 `build-requirements.txt`의 해시로 검증 설치한다(`--require-hashes`, `--only-binary :all:`). 패키지 자신은 `--no-build-isolation`으로 만든 wheel을 `--no-deps`로 설치한다.
- 실행: `krta --ticker <코드[.KS|.KQ]> --horizon <Nd|Nw|Nm|Ny> [--date YYYY-MM-DD]`. 날짜를 빼면 최근 확정 거래일을 고른다.
- 출력: 성공하면 stdout 마지막 줄에 `REPORT_DIR=<KRTA_DATA_DIR>/logs-kr/<CODE.KS|KQ>/<날짜>/<기간>/reports` 한 줄만 낸다. 나머지는 stderr(마스킹됨). 종료 코드는 0 성공, 2 입력 오류, 3 키 문제, 1 기타.
- 환경: 위 "환경변수" 표의 키와 설정을 주입하고, `KRTA_DATA_DIR`(이미지 기본 `/home/appuser/.krta`)을 쓰기 가능한 볼륨으로 마운트한다.

### 보안 설정

- 토스증권 키에는 주문 권한도 있다. `krta.data.toss`는 조회 경로 허용 목록(토큰 발급, 종목별 매매동향 5종, 시장 지표 투자자별 매매대금)만 보내고, 그 밖의 경로와 `X-Tossinvest-Account` 헤더는 보내기 전에 거부한다(테스트로 강제). 발급받은 access token은 메모리에만 두고 로그·예외에서 마스킹한다. 토큰은 클라이언트당 하나만 유효해서, 분석을 돌리면 같은 키로 발급해 둔 다른 프로그램의 토큰은 무효가 된다.
- 이미지는 uid/gid 1000(`appuser`)으로 돈다. 권장 실행 설정: non-root, 읽기 전용 루트 파일시스템, capabilities 전부 drop, `allowPrivilegeEscalation: false`, seccomp `RuntimeDefault`. 쓰기가 필요한 곳은 `/tmp`, `~/.cache`, `KRTA_DATA_DIR`뿐이다.
- LangSmith 추적은 실행 환경에서 끈다(`LANGSMITH_TRACING=false`, `LANGCHAIN_TRACING_V2=false`, `LANGSMITH_TRACING_V2=false`).
- 키는 이미지에 넣지 않는다. `.dockerignore`가 `.env*`·`*.secret`·`secrets/`·`*.pem`·`*.key`를 빌드 컨텍스트에서 뺀다.

## 라이선스

Apache License 2.0. `LICENSE` 참고.
