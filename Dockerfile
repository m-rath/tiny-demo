# syntax=docker/dockerfile:1

FROM python:3.10.11-slim
WORKDIR /app
COPY . /app
RUN cd /app && pip install --no-cache-dir -r requirements.txt
RUN python3 get_roberta.py
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000", "-w", "3", "-t", "60"]