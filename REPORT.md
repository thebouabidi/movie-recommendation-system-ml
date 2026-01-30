# TP 4 — Système de recommandation de films (MovieLens)

## Objectif
Le but de ce TP est de construire un système de recommandation de films à partir de données réelles (MovieLens).
On suit les étapes A1 à A6 demandées dans l’énoncé.

---

## A1 — Exploration des données
On utilise deux fichiers :
- `ratings.csv` : (userId, movieId, rating, timestamp)
- `movies.csv` : (movieId, title, genres)

Avant de modéliser, on observe :
- la taille des données (nombre de notes, nombre de films)
- les statistiques sur `rating` (min, max, moyenne)
- la distribution des notes (souvent concentrée autour de 3–4)

Cette étape permet de comprendre si les données sont “bruitées” ou déséquilibrées.

---

## A2 — Prétraitement
Les opérations faites :
1) Suppression des doublons dans `ratings`  
2) Fusion `ratings + movies` via `movieId` pour récupérer les titres/genres  
3) Filtrage :
   - garder uniquement les utilisateurs avec au moins 20 notes
   - garder uniquement les films avec au moins 20 notes

Pourquoi filtrer ?
- si un utilisateur a 1–2 notes, la similarité avec les autres est peu fiable
- si un film est très rare, il influence peu la recommandation et ajoute du bruit

---

## A3 — Modélisation
### 1) Recommandation par popularité (baseline)
On calcule la moyenne des notes par film, puis on prend le top-N.
Avantage : simple.  
Limite : pas personnalisé.

### 2) Filtrage collaboratif user-based
On construit une matrice utilisateur × film (ratings, valeurs manquantes = 0).
Ensuite on calcule la similarité cosinus entre utilisateurs.
Pour un utilisateur cible :
- on choisit les K utilisateurs les plus similaires
- on prédit une note par film (moyenne pondérée par la similarité)
- on recommande les films non vus avec le meilleur score

---

## A4 — Séparation train / test (concept)
Normalement, on sépare les notes en train/test :
- train : apprentissage des préférences
- test : vérifier la capacité du modèle à prédire des notes non vues

Dans cette version, on calcule une RMSE sur les notes existantes (mask R>0).
C’est une évaluation rapide, même si ce n’est pas le split le plus strict.

---

## A5 — Évaluation (RMSE)
La RMSE mesure l’erreur moyenne de prédiction (plus elle est faible, mieux c’est).

On calcule :
predicted = S × R / sum(|S|)
- S : matrice des similarités utilisateur-utilisateur
- R : matrice des notes

Puis on évalue uniquement là où une note existe (R > 0).

---

## A6 — Recommandations personnalisées
Pour un `userId`, on retourne les top-N films non notés avec :
- titre
- genres
- score prédit

Le résultat dépend des utilisateurs similaires, donc il est personnalisé.

---

## Conclusion
Le projet propose :
- une baseline popularité
- un modèle user-based simple
- une interface web Flask propre (HTML/CSS, sans JS)
- un score RMSE pour l’évaluation
