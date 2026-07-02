FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# All service modules (app, training, real-data feed, forecasting, churn,
# registry, …) — a glob so adding a module never breaks the image again.
COPY *.py ./

# Hugging Face Spaces (Docker SDK) expects the container to listen on 7860.
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
