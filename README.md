# MLOps Movie Recommendation System

## Overview
End-to-end MLOps pipeline for MovieLens recommendation system.

## Pipeline
- Data Ingestion (Lab 4)
- Feature Engineering (Lab 5)
- Model Training (Lab 6)
- Evaluation (Lab 7)
- Experiment Tracking (MLflow)
- API Deployment (FastAPI)

## API Endpoints
- /health → check service
- /recommend → get movie recommendations

## Run API
uvicorn src.app:app --reload

## Test API
pytest tests -v