FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2 PIP_NO_CACHE_DIR=1
RUN useradd -m -u 1000 user
WORKDIR /home/user/app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY --chown=user main.py champion.keras ./
USER user
ENV MODEL_PATH=/home/user/app/champion.keras
EXPOSE 7860
CMD ["gunicorn", "-b", "0.0.0.0:7860", "-w", "1", "--threads", "2", "--timeout", "120", "main:app"]