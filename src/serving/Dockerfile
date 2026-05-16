FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements_serving.txt .
RUN pip install --no-cache-dir -r requirements_serving.txt

# Copy app and model artifacts
COPY src/serving/app.py .
COPY mlruns/ ./mlruns/
COPY .env .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]