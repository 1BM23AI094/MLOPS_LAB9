import joblib
import numpy as np
import pandas as pd

class KNNRecommendationModel:
    def __init__(self, k=5):
        self.k = k
        self.features = None
        self.ratings_df = None
        self.default_rating = 3.0

    def fit(self, features, ratings_df):
        self.features = features
        self.ratings_df = ratings_df
        return self

    def predict_rating(self, user_id, movie_id):
        user_ids = self.features["user_ids"]
        similarity = self.features["similarity"]

        # find index of user
        try:
            idx = list(user_ids).index(user_id)
        except ValueError:
            return self.default_rating

        sim_scores = similarity[idx]

        # top K similar users
        similar_idx = np.argsort(sim_scores)[-self.k:][::-1]
        similar_users = user_ids[similar_idx]

        ratings = self.ratings_df[
            (self.ratings_df['user_id'].isin(similar_users)) &
            (self.ratings_df['movie_id'] == movie_id)
        ]['rating'].values

        if len(ratings) == 0:
            return self.default_rating

        return float(np.mean(ratings))

    def predict_batch(self, df):
        return np.array([
            self.predict_rating(row['user_id'], row['movie_id'])
            for _, row in df.iterrows()
        ])

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)