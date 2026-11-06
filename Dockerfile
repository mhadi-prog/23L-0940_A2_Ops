FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 TF_CPP_MIN_LOG_LEVEL=2
WORKDIR /srv

# 1. Dependencies first: this layer is cached until requirements change
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

# 2. Source code last: edits here don't invalidate the slow layer above
COPY app/ app/

RUN useradd --create-home --uid 1001 appuser
USER appuser

ENV HOST=0.0.0.0 PORT=5001 MODEL_PATH=/models/champion.keras
EXPOSE 5001
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5001/health').status==200 else 1)"

CMD ["python", "-m", "app.app"]