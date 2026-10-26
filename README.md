# Movie Recommendation System

An interactive movie discovery and recommendation platform built with Python, Scikit-Learn, Plotly, and Streamlit.

---

## 📌 Project Overview

The **Movie Recommendation System** combines Natural Language Processing (TF-IDF vector space modeling) and IMDb Bayesian Weighted Rating formulations to provide instant, high-precision film recommendations across a catalog of 62,000+ movies and over 25,000,000 viewer ratings.

The application allows users to discover movies similar to their favorites, explore top charts filtered by genre and era, generate customized preference-based watchlists, and analyze cinema trends.

---

## ✨ Key Features

- **Content-Based Movie Recommender**: Search or select any movie from thousands of titles to receive instant recommendations ranked by Cosine Similarity and Bayesian weighted rating.
- **Top Charts & Genre Explorer**: Browse the top 50 all-time highest-rated movies with interactive multi-genre filters, decade filters, and minimum vote threshold controls.
- **Personalized Watchlist Builder**: Tailor a custom watchlist based on preferred genres, release decades, and target minimum ratings.
- **Cinema Analytics Dashboard**: Empirical analytics on genre distributions, rating frequency distributions, and popularity vs. quality scatter visualizations.
- **Mathematical Transparency**: Detailed explanations and LaTeX formulas for the Bayesian Weighted Rating and TF-IDF similarity metrics.

---

## 🧠 System Architecture & Mathematical Formulations

### 1. IMDb Bayesian Weighted Rating ($WR$)
To ensure movies with only a handful of 5-star reviews do not unfairly outrank acclaimed films with tens of thousands of reviews, we compute:

$$WR = \left(\frac{v}{v + m}\right) \cdot R + \left(\frac{m}{v + m}\right) \cdot C$$

- $v$: Number of ratings for the movie (`vote_count`)
- $m$: Minimum vote threshold for rating stability ($m = 100$)
- $R$: Average community rating of the movie (`vote_average`)
- $C$: Mean rating across the entire catalog ($C \approx 3.12$)

### 2. TF-IDF & Cosine Similarity
Film features (title tokens, double-weighted genre tags, and decade indicators) are converted into vector space embeddings using an 8,000-feature TF-IDF matrix. Similarity between movies $A$ and $B$ is determined via Cosine Similarity:

$$\text{Cosine Similarity}(A, B) = \frac{A \cdot B}{\|A\|_2 \|B\|_2}$$

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9+ (tested on Python 3.11)
- `pip` package manager

### 2. Installation
Clone the repository and install the dependencies:
```bash
git clone <your-repository-url>
cd Chetya
pip install -r requirements.txt
```

### 3. Model Training (Optional)
Precomputed and serialized models are included in `models/`. To retrain the recommendation index:
```bash
python train_model.py
```

### 4. Running the Application
Launch the Streamlit web dashboard:
```bash
streamlit run app.py
```
The app will open automatically in your browser at `http://localhost:8501`.

---

## 🌐 Streamlit Cloud Deployment Guide

1. Push this directory (`Chetya/`) to a GitHub repository.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **"New app"**.
4. Select your repository, set the branch to `main`, and specify **Main file path** as `app.py`.
5. Click **"Deploy!"**.

---

## 📁 Repository Structure

```text
├── app.py                      # Main interactive Streamlit application
├── train_model.py              # Recommendation model training & indexing script
├── requirements.txt            # Python dependencies for deployment
├── README.md                   # Project documentation
├── .gitignore                  # Git ignore rules
├── movies.csv                  # MovieLens movies catalog
├── movies_with_ratings.csv     # Enriched movies dataset with Bayesian ratings
└── models/                     # Saved model artifacts
    ├── movie_recommender.pkl   # Serialized TF-IDF vectorizer & sparse matrix
    ├── movies_pool.csv         # Processed recommendation pool
    └── metadata.json           # Catalog genres, decades, and statistics
```
