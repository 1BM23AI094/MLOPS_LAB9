import mlflow
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from src.train import KNNRecommendationModel


def run_experiment(k):
    mlflow.start_run()

    df = pd.read_csv("data/processed/ratings_clean.csv")
    features = joblib.load("models/user_similarity.pkl")

    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)

    model = KNNRecommendationModel(k=k)
    model.fit(features, train_df)

    y_true = test_df["rating"].values
    y_pred = model.predict_batch(test_df)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    mlflow.log_param("k", k)
    mlflow.log_metric("rmse", rmse)

    model_path = f"models/model_k{k}.pkl"
    joblib.dump(model, model_path)

    mlflow.log_artifact(model_path)

    print(f"K={k}, RMSE={rmse}")

    mlflow.end_run()


if __name__ == "__main__":
    for k in [3, 5, 10]:
        run_experiment(k)