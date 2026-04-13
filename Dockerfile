# ---------- base ----------
FROM python:3.12-slim AS base

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgmsh-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir .

# ---------- dev ----------
FROM base AS dev

RUN pip install --no-cache-dir -e ".[dev,viz,ml]"

COPY tests/ tests/
COPY examples/ examples/

# ---------- prod ----------
FROM base AS prod

CMD ["feaweld", "--help"]
