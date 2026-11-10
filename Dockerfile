# KoreanTradingAgents(krta) 실행 이미지. 빌드 컨텍스트는 저장소 루트다.
# 공급망: 베이스 이미지는 digest로, 파이썬 패키지는 전부 해시로 고정한다.
#   - 런타임 의존성: uv.lock → `uv export`(해시 포함) → `pip install --require-hashes`
#   - 빌드 도구(uv, hatchling): build-requirements.txt(해시 포함). hatchling은 빌드 격리로
#     해시 없이 받아지지 않도록 미리 해시 검증 설치하고 `--no-build-isolation`으로 wheel을 만든다.
#   - `--only-binary :all:`로 sdist 빌드(해시 없는 빌드 의존성 설치)를 막는다.
# 베이스 갱신: python:3.13-slim의 새 index digest로 두 FROM을 함께 바꾼다.
ARG PYTHON_IMAGE=python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

FROM ${PYTHON_IMAGE} AS builder
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_NO_CACHE=1 \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /build

# 빌드 도구(런타임 이미지에는 복사하지 않는다)
COPY build-requirements.txt ./
RUN python -m venv /opt/build \
 && /opt/build/bin/pip install --require-hashes --only-binary :all: -r build-requirements.txt

# 런타임 의존성: 프로젝트 자신(-e .)은 해시가 없으므로 빼고 내보낸다.
COPY pyproject.toml uv.lock ./
RUN /opt/build/bin/uv export --frozen --no-dev --no-emit-project --format requirements-txt > req.txt \
 && python -m venv /opt/venv \
 && /opt/venv/bin/pip install --require-hashes --only-binary :all: -r req.txt

# 패키지 자신: 해시 검증된 hatchling으로 wheel을 만들고 의존성 없이 설치한다.
COPY README.md LICENSE ./
COPY src ./src
RUN /opt/build/bin/pip wheel --no-deps --no-build-isolation --no-index --wheel-dir /build/dist . \
 && /opt/venv/bin/pip install --no-deps --no-index /build/dist/krta-*.whl

FROM ${PYTHON_IMAGE}
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    HOME=/home/appuser \
    KRTA_DATA_DIR=/home/appuser/.krta

# uid/gid 1000 = 실행 환경의 runAsUser/runAsGroup. 데이터 볼륨도 이 uid가 쓸 수 있어야 한다.
# 루트 파일시스템을 읽기 전용으로 돌릴 수 있다. 쓰기는 /tmp, ~/.cache, ~/.krta(데이터 볼륨)만.
RUN groupadd --gid 1000 appuser \
 && useradd --uid 1000 --gid 1000 --create-home --shell /usr/sbin/nologin appuser \
 && install -d -m 0755 -o appuser -g appuser /home/appuser/.krta /home/appuser/.cache

COPY --from=builder /opt/venv /opt/venv
USER 1000:1000
WORKDIR /home/appuser
ENTRYPOINT ["krta"]
