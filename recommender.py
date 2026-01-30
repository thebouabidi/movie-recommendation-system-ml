import io
import zipfile
import requests

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error


MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def load_data(url: str = MOVIELENS_URL) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Downloads MovieLens 'latest-small' zip and loads ratings + movies.
    Kept simple on purpose (TP style).
    """
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(
            "Failed to download MovieLens dataset. Check your internet connection."
        ) from e

    try:
        z = zipfile.ZipFile(io.BytesIO(r.content))
        ratings = pd.read_csv(z.open("ml-latest-small/ratings.csv"))
        movies = pd.read_csv(z.open("ml-latest-small/movies.csv"))
    except Exception as e:
        raise RuntimeError("Dataset zip exists but files could not be read.") from e

    return ratings, movies


def preprocess_data(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    min_user_ratings: int = 20,
    min_movie_ratings: int = 20,
) -> pd.DataFrame:
    ratings = ratings.drop_duplicates()

    data = pd.merge(ratings, movies, on="movieId", how="inner")

    user_counts = data["userId"].value_counts()
    movie_counts = data["movieId"].value_counts()

    data = data[
        data["userId"].isin(user_counts[user_counts >= min_user_ratings].index)
        & data["movieId"].isin(movie_counts[movie_counts >= min_movie_ratings].index)
    ].copy()

    return data


def build_user_movie_matrix(data: pd.DataFrame) -> pd.DataFrame:
    matrix = (
        data.pivot_table(index="userId", columns="movieId", values="rating")
        .fillna(0)
        .astype(float)
    )
    return matrix


def compute_similarity(user_movie_matrix: pd.DataFrame) -> pd.DataFrame:
    sim = cosine_similarity(user_movie_matrix)
    sim_df = pd.DataFrame(sim, index=user_movie_matrix.index, columns=user_movie_matrix.index)
    return sim_df


def get_top_popular_movies(data: pd.DataFrame, n: int = 10) -> list[dict]:
    top = (
        data.groupby(["movieId", "title", "genres"])["rating"]
        .mean()
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )

    results = []
    for _, row in top.iterrows():
        results.append(
            {
                "movieId": int(row["movieId"]),
                "title": str(row["title"]),
                "genres": str(row["genres"]),
                "score": float(row["rating"]),
                "type": "popular",
            }
        )
    return results


def recommend_movies(
    user_id: int,
    user_movie_matrix: pd.DataFrame,
    user_similarity_df: pd.DataFrame,
    movies: pd.DataFrame,
    n: int = 10,
    k_neighbors: int = 5,
) -> list[dict]:
    if user_id not in user_movie_matrix.index:
        raise ValueError(f"userId {user_id} not found in filtered dataset.")

    similar_users = user_similarity_df[user_id].sort_values(ascending=False)

    similar_users = similar_users.drop(index=user_id, errors="ignore")

    neighbors = similar_users.head(k_neighbors)

    weighted_ratings = user_movie_matrix.loc[neighbors.index].T.dot(neighbors)
    denom = neighbors.sum()

    if denom == 0:
        return []

    scores = weighted_ratings / denom

    seen = user_movie_matrix.loc[user_id]
    scores = scores[seen == 0]

    top = scores.sort_values(ascending=False).head(n)

    movies_small = movies[["movieId", "title", "genres"]].copy()
    movies_small["movieId"] = movies_small["movieId"].astype(int)
    meta = movies_small.set_index("movieId")

    results = []
    for movie_id, score in top.items():
        if int(movie_id) in meta.index:
            results.append(
                {
                    "movieId": int(movie_id),
                    "title": str(meta.loc[int(movie_id), "title"]),
                    "genres": str(meta.loc[int(movie_id), "genres"]),
                    "score": float(score),
                    "type": "user_based",
                }
            )

    return results


def compute_rmse(user_movie_matrix: pd.DataFrame, user_similarity_df: pd.DataFrame) -> float:
    R = user_movie_matrix.values
    S = user_similarity_df.values

    denom = np.abs(S).sum(axis=1).reshape(-1, 1)
    denom[denom == 0] = 1.0  

    predicted = S.dot(R) / denom

    mask = R > 0
    rmse = np.sqrt(mean_squared_error(R[mask], predicted[mask]))
    return float(rmse)
