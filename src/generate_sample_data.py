import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

n = 2000

df = pd.DataFrame({
    'user_id': np.random.randint(1, 200, n),
    'movie_id': np.random.randint(1, 100, n),
    'rating': np.random.uniform(1, 5, n),
    'timestamp': np.random.randint(1000000000, 1500000000, n)
})

Path('data/raw').mkdir(parents=True, exist_ok=True)

df.to_csv('data/raw/ratings.csv', sep='\t', index=False)

print("data generated")