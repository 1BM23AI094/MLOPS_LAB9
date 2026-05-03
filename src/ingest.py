import pandas as pd
import json
import logging
from pathlib import Path
from src.config import RATINGS_SCHEMA, DATA_PATHS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate(df):
    before = len(df)

    for col, rules in RATINGS_SCHEMA.items():
        df = df[(df[col] >= rules['min']) & (df[col] <= rules['max'])]

    df = df.dropna()

    after = len(df)

    report = {
        "rows_before": before,
        "rows_after": after,
        "rows_removed": before - after
    }

    return df, report

def main():
    df = pd.read_csv(DATA_PATHS['raw'], sep='\t')

    logger.info(f"Loaded {len(df)} rows")

    df = df.drop_duplicates(subset=['user_id', 'movie_id'])

    df, report = validate(df)

    Path('data/processed').mkdir(parents=True, exist_ok=True)
    Path('evaluations').mkdir(parents=True, exist_ok=True)

    df.to_csv(DATA_PATHS['processed'], index=False)

    with open(DATA_PATHS['validation_report'], 'w') as f:
        json.dump(report, f, indent=2)

    logger.info("Pipeline complete")

if __name__ == "__main__":
    main()
    