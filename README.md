🎬 CineReco — Movie Recommendation System (Machine Learning)

CineReco is a machine learning–based movie recommendation system developed as a university project (TP).
It uses the MovieLens dataset to generate personalized movie recommendations using
**user-based collaborative filtering**.

────────────────────────────────
🧠 How It Works
────────────────────────────────
- Build a User × Movie rating matrix
- Compute cosine similarity between users
- Predict ratings using similar users (weighted average)
- Recommend unseen movies with the highest predicted scores

A popularity-based model (mean rating) is also used as a baseline.

────────────────────────────────
📂 Dataset
────────────────────────────────
MovieLens (latest-small):
- ratings.csv
- movies.csv

Users and movies with fewer than 20 ratings are filtered to improve recommendation quality.

────────────────────────────────
⚙️ Installation & Run
────────────────────────────────
```bash
python -m venv venv
# Activate venv
pip install -r requirements.txt
python app.py
Open: http://127.0.0.1:5000/

────────────────────────────────
🛠 Technologies
────────────────────────────────
Python · Pandas · NumPy · Scikit-learn · Flask · HTML · CSS

────────────────────────────────
👤 Author
────────────────────────────────
thebouabidi(SALMA BOUABIDI) — Cybersecurity & AI Student | Full-Stack Web Developer