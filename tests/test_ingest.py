import pandas as pd
from src.ingest import validate

def test_valid_data():
    df = pd.DataFrame({
        'user_id': [1, 2],
        'movie_id': [10, 20],
        'rating': [4.0, 5.0],
        'timestamp': [1200000000, 1300000000]
    })

    clean_df, report = validate(df)

    assert len(clean_df) == 2
    assert report['rows_removed'] == 0


def test_invalid_rating():
    df = pd.DataFrame({
        'user_id': [1],
        'movie_id': [10],
        'rating': [10.0],  # invalid
        'timestamp': [1200000000]
    })

    clean_df, report = validate(df)

    assert len(clean_df) == 0
    assert report['rows_removed'] == 1


def test_missing_values():
    df = pd.DataFrame({
        'user_id': [1],
        'movie_id': [None],
        'rating': [4.0],
        'timestamp': [1200000000]
    })

    clean_df, report = validate(df)

    assert len(clean_df) == 0