import json
import pandas as pd
from pathlib import Path

from src.evaluate import (
    load_model,
    evaluate_rating,
    compute_coverage,
    analyze_sparsity,
    baseline_rmse
)


def main():
    print("Loading model...")
    model = load_model("models/model.pkl")

    print("Loading data...")
    df = pd.read_csv("data/processed/ratings_clean.csv")

    # Split (same as training)
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)

    print("Generating predictions...")
    y_true = test_df["rating"].values
    y_pred = model.predict_batch(test_df)

    print("Evaluating...")
    rating_metrics = evaluate_rating(y_true, y_pred)
    coverage_metrics = compute_coverage(model, test_df)
    sparsity_metrics = analyze_sparsity(df)
    baseline = baseline_rmse(y_true)

    report = {
        "rating": rating_metrics,
        "coverage": coverage_metrics,
        "sparsity": sparsity_metrics,
        "baseline_rmse": baseline
    }

    Path("evaluations").mkdir(exist_ok=True)

    with open("evaluations/evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n===== RESULTS =====")
    print("RMSE:", rating_metrics["rmse"])
    print("MAE:", rating_metrics["mae"])
    print("Coverage:", coverage_metrics["coverage_ratio"])
    print("Sparsity:", sparsity_metrics["sparsity"])
    print("Baseline RMSE:", baseline)
    print("\nEvaluation saved!")


if __name__ == "__main__":
    main()