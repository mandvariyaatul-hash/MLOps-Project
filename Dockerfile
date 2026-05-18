FROM python:3.9-slim

ENV OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# THE FIX: 
# 1. Clean up multipart mess
# 2. Install requirements
# 3. VERIFY evidently.report exists before finishing the build
RUN pip install --no-cache-dir --upgrade pip && \
    pip uninstall -y multipart python-multipart evidently && \
    pip install --no-cache-dir -r requirements.txt && \
    python -c "from evidently.report import Report; print('✅ Evidently Verified')"

COPY . .
RUN mkdir -p drift_reports

EXPOSE 8000
EXPOSE 5000

CMD sh -c "mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root /app/mlartifacts --host 0.0.0.0 --port 5000 & sleep 2 && uvicorn main:app --host 0.0.0.0 --port 8000"