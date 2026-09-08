FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/data

WORKDIR /app

RUN groupadd --system app && useradd --system --gid app --home-dir /app app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

RUN mkdir -p /data && chown -R app:app /data /app

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz').status==200 else 1)"

# --proxy-headers is on, but uvicorn only honours X-Forwarded-* from peers in
# FORWARDED_ALLOW_IPS (default: 127.0.0.1). Do NOT set that to "*" - it would
# let any client spoof X-Forwarded-For and bypass the per-IP login lockout.
# When running behind a trusted reverse proxy, set FORWARDED_ALLOW_IPS to that
# proxy's address (and TRUST_PROXY=true) in the environment.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers"]
