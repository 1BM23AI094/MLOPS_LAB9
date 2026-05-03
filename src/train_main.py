import joblib
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.train import KNNRecommendationModel


def main():
    print("Loading features...")
    features = joblib.load("models/user_similarity.pkl")

    print("Loading data...")
    df = pd.read_csv("data/processed/ratings_clean.csv")

    # split
    n = len(df)
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)

    print("Training model...")
    model = KNNRecommendationModel(k=5)
    model.fit(features, train_df)

    print("Evaluating...")
    y_true = val_df['rating'].values
    y_pred = model.predict_batch(val_df)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    print("RMSE:", rmse)
    print("MAE:", mae)

    Path("models").mkdir(exist_ok=True)

    model.save("models/model.pkl")

    metadata = {
        "k": 5,
        "rmse": float(rmse),
        "mae": float(mae)
    }

    with open("models/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Model saved!")


if __name__ == "__main__":
    main()