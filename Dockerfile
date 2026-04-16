FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    openbabel \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip && \
    pip install gunicorn && \
    pip install -r requirements.txt

COPY . /app

RUN mkdir -p \
    /app/runtime/uploads \
    /app/runtime/generated_files \
    /app/runtime/docking_results \
    /app/runtime/blind_jobs \
    /app/runtime/admet_jobs \
    /app/runtime/qsar_jobs \
    /app/runtime/hit_to_lead_jobs \
    /app/runtime/plip_jobs \
    /app/models/qsar_saved_models

CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT:-10000} --workers 1 --timeout 300"]
