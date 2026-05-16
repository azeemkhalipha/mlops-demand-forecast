
## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Check if API and model are running |
| `/predict` | POST | Single demand prediction |
| `/batch_predict` | POST | Bulk predictions |
| `/docs` | GET | Interactive Swagger UI |

## Running the API

**Locally:**
```bash
conda activate mlops
cd src/serving
uvicorn app:app --reload --port 8000
```

**With Docker:**
```bash
docker build -t demand-forecast-api:v1 .
docker run -p 8001:8000 demand-forecast-api:v1
```

**Test a prediction:**
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "qty_lag_1": 10.0,
    "qty_lag_7": 8.0,
    "qty_lag_30": 9.0,
    "qty_rolling_avg_7": 9.5,
    "qty_rolling_avg_30": 9.0,
    "qty_rolling_std_7": 1.2,
    "daily_revenue": 45.0
  }'
```

## Running Tests
```bash
pytest tests/ -v
```
