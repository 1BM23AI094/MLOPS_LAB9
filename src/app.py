import joblib
from fastapi import FastAPI

app = FastAPI()

# Load model and features
model = joblib.load("models/model.pkl")
features = joblib.load("models/user_similarity.pkl")


@app.get("/")
def home():
    return {"message": "API running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/recommend")
def recommend(user_id: int, n: int = 5):
    recommendations = []

    for movie_id in range(1, 20):
        rating = model.predict_rating(user_id, movie_id)
        recommendations.append({
            "movie_id": movie_id,
            "predicted_rating": float(rating)
        })

    top_n = sorted(recommendations, key=lambda x: x["predicted_rating"], reverse=True)[:n]

    return {
        "user_id": user_id,
        "recommendations": top_n
    }