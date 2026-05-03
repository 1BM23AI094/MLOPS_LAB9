import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from pathlib import Path

def main():
    df = pd.read_csv("data/processed/ratings_clean.csv")

    print("Loaded data:", df.shape)

    matrix = df.pivot_table(
        index='user_id',
        columns='movie_id',
        values='rating',
        fill_value=0
    )

    print("Matrix shape:", matrix.shape)

    similarity = cosine_similarity(matrix)

    print("Similarity shape:", similarity.shape)

    feature_store = {
        "similarity": similarity,
        "user_ids": matrix.index.values,
        "movie_ids": matrix.columns.values
    }

    Path("models").mkdir(exist_ok=True)
    joblib.dump(feature_store, "models/user_similarity.pkl")

    print("Feature store saved!")

if __name__ == "__main__":
    main()
    