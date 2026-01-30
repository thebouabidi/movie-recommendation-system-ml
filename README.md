🎬 CineReco — Movie Recommendation System (Machine Learning)

CineReco is a machine learning–based movie recommendation system developed as a university project (TP).
It uses **collaborative filtering techniques** to generate personalized movie recommendations
based on user ratings and similarity between users.

The project is designed as a **web application using Flask**, with a clean interface built using HTML and CSS.

🧠 How the Recommendation Works

1️⃣ Data Preparation  
The system uses the MovieLens dataset and works with:

- userId  
- movieId  
- rating  
- title  
- genres  

Users and movies with very few ratings are filtered to improve recommendation quality.

2️⃣ User-Based Collaborative Filtering  
The recommendation process is based on:

- Building a User × Movie rating matrix  
- Computing cosine similarity between users  
- Selecting the most similar users (neighbors)  
- Predicting ratings using a weighted average  

Movies not yet rated by the target user are recommended.

3️⃣ Popularity-Based Baseline  
As a comparison baseline, the system also recommends movies
with the highest **average rating**.

📂 Dataset

The project uses the **MovieLens (latest-small)** dataset.

Files:
- ratings.csv  
- movies.csv  

The dataset is downloaded automatically when the application starts.

⚙️ Installation

1. Clone the repository
```bash
git clone https://github.com/your-username/CineReco.git
cd CineReco
