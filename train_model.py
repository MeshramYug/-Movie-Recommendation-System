"""
Movie Recommendation System - Model Training Script
Trains:
1. Content-Based TF-IDF & Genre Cosine Similarity Recommender
2. IMDb Bayesian Weighted Rating Engine
"""

import os
import re
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def train_movie_recommender(
    movies_ratings_path='movies_with_ratings.csv',
    movies_raw_path='movies.csv',
    output_dir='models'
):
    print("Preparing movie dataset...")
    os.makedirs(output_dir, exist_ok=True)
    
    if os.path.exists(movies_ratings_path):
        df = pd.read_csv(movies_ratings_path)
    elif os.path.exists(movies_raw_path):
        df = pd.read_csv(movies_raw_path)
        df['vote_count'] = 0
        df['vote_average'] = 0.0
        df['weighted_rating'] = 0.0
    else:
        raise FileNotFoundError("Neither movies_with_ratings.csv nor movies.csv found.")

    print(f"Total Movies: {len(df):,}")

    # ---------------------------------------------------------
    # 1. Feature Engineering
    # ---------------------------------------------------------
    # Extract Year from title (e.g. "Toy Story (1995)" -> 1995)
    def extract_year(title):
        match = re.search(r'\((\d{4})\)', str(title))
        return int(match.group(1)) if match else np.nan

    def clean_title(title):
        # Remove trailing year and parenthesis
        return re.sub(r'\s*\(\d{4}\)\s*$', '', str(title)).strip()

    df['year'] = df['title'].apply(extract_year)
    df['clean_title'] = df['title'].apply(clean_title)
    df['genres'] = df['genres'].fillna('').replace('(no genres listed)', 'Other')

    # Calculate decade
    df['decade'] = (df['year'] // 10 * 10).fillna(2000).astype(int).astype(str) + "s"

    # Ensure Bayesian Weighted Rating is present
    if 'weighted_rating' not in df.columns or df['weighted_rating'].isna().all():
        C = df[df['vote_count'] > 0]['vote_average'].mean() if len(df[df['vote_count'] > 0]) > 0 else 3.0
        m = 50
        df['weighted_rating'] = (
            (df['vote_count'] / (df['vote_count'] + m)) * df['vote_average'] + 
            (m / (df['vote_count'] + m)) * C
        ).round(2)

    # ---------------------------------------------------------
    # 2. Select Recommendation Pool for Low Latency
    # ---------------------------------------------------------
    # Filter for movies with at least 25 ratings, or top 12,000 by vote_count
    # This guarantees high-quality recommendations and sub-second Streamlit response time
    pool_df = df[df['vote_count'] >= 25].copy()
    if len(pool_df) < 5000:
        pool_df = df.sort_values('vote_count', ascending=False).head(10000).copy()

    pool_df = pool_df.reset_index(drop=True)
    print(f"Core Recommendation Pool: {len(pool_df):,} popular/cult titles")

    # Combine genres (repeated for weighting), title keywords, and decade
    pool_df['content_soup'] = (
        pool_df['clean_title'] + " " + 
        pool_df['genres'].str.replace('|', ' ') + " " +
        pool_df['genres'].str.replace('|', ' ') + " " +
        pool_df['decade']
    )

    # ---------------------------------------------------------
    # 3. Fit TF-IDF Vectorizer
    # ---------------------------------------------------------
    print("Fitting TF-IDF Vectorizer...")
    tfidf = TfidfVectorizer(stop_words='english', max_features=8000, ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(pool_df['content_soup'])
    print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")

    # Extract all distinct genres
    all_genres = set()
    for g_str in df['genres'].dropna():
        for g in g_str.split('|'):
            if g.strip() and g.strip() != 'Other':
                all_genres.add(g.strip())

    # ---------------------------------------------------------
    # 4. Save Processed Artifacts
    # ---------------------------------------------------------
    recommender_data = {
        'pool_df': pool_df[['movieId', 'clean_title', 'title', 'year', 'genres', 'vote_count', 'vote_average', 'weighted_rating']],
        'tfidf_matrix': tfidf_matrix,
        'tfidf_vectorizer': tfidf
    }
    joblib.dump(recommender_data, os.path.join(output_dir, 'movie_recommender.pkl'), compress=3)

    # Save compact CSV of movies for fast discovery in app
    pool_df.to_csv(os.path.join(output_dir, 'movies_pool.csv'), index=False)

    metadata = {
        'total_catalog_movies': len(df),
        'pool_movies_count': len(pool_df),
        'genres': sorted(list(all_genres)),
        'decades': sorted(pool_df['decade'].unique().tolist())
    }
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Movie recommendation artifacts saved to '{output_dir}/'")

if __name__ == '__main__':
    train_movie_recommender()
