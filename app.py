from flask import Flask, render_template, request

import recommender


app = Flask(__name__)

CACHE = {
    "loaded": False,
    "ratings": None,
    "movies": None,
    "data": None,
    "matrix": None,
    "sim": None,
    "rmse": None,
}


def init_model_if_needed():
    if CACHE["loaded"]:
        return

    ratings, movies = recommender.load_data()
    data = recommender.preprocess_data(ratings, movies, min_user_ratings=20, min_movie_ratings=20)

    matrix = recommender.build_user_movie_matrix(data)
    sim = recommender.compute_similarity(matrix)

    rmse = recommender.compute_rmse(matrix, sim)

    CACHE["ratings"] = ratings
    CACHE["movies"] = movies
    CACHE["data"] = data
    CACHE["matrix"] = matrix
    CACHE["sim"] = sim
    CACHE["rmse"] = rmse
    CACHE["loaded"] = True


@app.route("/")
def index():
    error = request.args.get("error")

    try:
        init_model_if_needed()
        user_count = int(CACHE["matrix"].shape[0])
        movie_count = int(CACHE["matrix"].shape[1])
        popular = recommender.get_top_popular_movies(CACHE["data"], n=8)
        rmse = CACHE["rmse"]
        return render_template(
            "index.html",
            error=error,
            user_count=user_count,
            movie_count=movie_count,
            popular=popular,
            rmse=rmse,
        )
    except Exception as e:
        return render_template(
            "index.html",
            error=str(e),
            user_count=0,
            movie_count=0,
            popular=[],
            rmse=None,
        )


@app.route("/recommend", methods=["POST"])
def recommend():
    init_model_if_needed()

    user_id_raw = request.form.get("user_id", "").strip()
    n_raw = request.form.get("n_reco", "10").strip()

    try:
        user_id = int(user_id_raw)
    except ValueError:
        return render_template(
            "results.html",
            error="userId must be an integer (example: 1, 15, 42).",
            recommendations=[],
            user_id=user_id_raw,
            n=n_raw,
        )

    try:
        n = int(n_raw)
        if n < 1:
            n = 1
        if n > 30:
            n = 30 
    except ValueError:
        n = 10

    try:
        recos = recommender.recommend_movies(
            user_id=user_id,
            user_movie_matrix=CACHE["matrix"],
            user_similarity_df=CACHE["sim"],
            movies=CACHE["movies"],
            n=n,
            k_neighbors=5,
        )

        if not recos:
            msg = (
                "No recommendations found (maybe the user has seen too many movies "
                "in the filtered dataset, or similarity sum was 0)."
            )
            return render_template(
                "results.html",
                error=msg,
                recommendations=[],
                user_id=user_id,
                n=n,
            )

        return render_template(
            "results.html",
            error=None,
            recommendations=recos,
            user_id=user_id,
            n=n,
        )

    except ValueError as e:
        return render_template(
            "results.html",
            error=str(e),
            recommendations=[],
            user_id=user_id,
            n=n,
        )


@app.route("/report")
def report():
    init_model_if_needed()

    data = CACHE["data"]
    stats = {
        "n_ratings_before": int(CACHE["ratings"].shape[0]),
        "n_movies_total": int(CACHE["movies"].shape[0]),
        "n_ratings_after": int(data.shape[0]),
        "n_users_after": int(CACHE["matrix"].shape[0]),
        "n_movies_after": int(CACHE["matrix"].shape[1]),
        "rmse": CACHE["rmse"],
    }

    return render_template("report.html", stats=stats)


if __name__ == "__main__":
    app.run(debug=True)
