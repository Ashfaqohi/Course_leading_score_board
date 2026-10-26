# Lead conversion prediction API

## Setup

From this directory, use `uv` to resolve dependencies and create the project environment:

```bash
cd 4_classification
uv sync
```

## Start the local server

```bash
uv run uvicorn predict:app --host 0.0.0.0 --port 9696
```

The API will be available at `http://127.0.0.1:9696`. Interactive API documentation is at `http://127.0.0.1:9696/docs`.

## Test it

Health check:

```bash
curl http://127.0.0.1:9696/health
```

Prediction request:

```bash
curl -X POST http://127.0.0.1:9696/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "lead_source": "paid_ads",
    "industry": "technology",
    "employment_status": "employed",
    "location": "north_america",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0,
    "interaction_count": 4,
    "lead_score": 0.41
  }'
```

The response contains `conversion_probability`, `predicted_class`, and a boolean `converted` value.

## Run with Docker

Build the image:

```bash
docker build -t course-lead-api .
```

Start the container:

```bash
docker run --rm -p 9696:9696 course-lead-api
```

The API is then available at `http://127.0.0.1:9696` on your computer.
