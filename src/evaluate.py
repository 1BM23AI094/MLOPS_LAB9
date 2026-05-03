import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ===============================
# LOAD MODEL
# ===============================
def load_model(path):
    return joblib.load(path)


# ===============================
# RATING METRICS
# ===============================
def evaluate_rating(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    errors = np.abs(y_true - y_pred)

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "mean_error": float(errors.mean()),
        "max_error": float(errors.max()),
        "min_error": float(errors.min())
    }


# ===============================
# COVERAGE
# ===============================
def compute_coverage(model, df):
    movies = df["movie_id"].unique()
    recommended = set()

    for movie in movies:
        users = df[df["movie_id"] == movie]["user_id"].unique()

        for u in users:
            pred = model.predict_rating(int(u), int(movie))
            if pred >= 3.0:
                recommended.add(movie)
                break

    coverage = len(recommended) / len(movies)

    return {
        "coverage_ratio": float(coverage),
        "recommended_movies": int(len(recommended)),
        "total_movies": int(len(movies))
    }


# ===============================
# SPARSITY
# ===============================
def analyze_sparsity(df):
    n_users = df["user_id"].nunique()
    n_movies = df["movie_id"].nunique()
    n_ratings = len(df)

    max_possible = n_users * n_movies
    density = n_ratings / max_possible
    sparsity = 1 - density

    return {
        "n_users": int(n_users),
        "n_movies": int(n_movies),
        "n_ratings": int(n_ratings),
        "density": float(density),
        "sparsity": float(sparsity)
    }


# ===============================
# BASELINE
# ===============================
def baseline_rmse(y_true):
    mean_pred = np.full_like(y_true, np.mean(y_true))
    rmse = np.sqrt(mean_squared_error(y_true, mean_pred))
    return float(rmse)