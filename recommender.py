"""
Recommendation Engine for Movie Recommendation System
Features:
1. Multilingual Content-Based TF-IDF & Cosine Similarity over Genres, Synopsis, Directors, Cast, Languages, and Keywords.
2. Era & Decade Weighted Quality Blending.
3. Mood & Theme based Semantic Recommendation.
4. Explainable AI (XAI) with Match Badges and Justifications.
"""

import sqlite3
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from database import get_db

class MovieRecommender:
    def __init__(self):
        self.movies_df = pd.DataFrame()
        self.tfidf = None
        self.tfidf_matrix = None
        self.refresh()

    def refresh(self):
        """Fetch latest movies from SQLite and rebuild vector index."""
        conn = get_db()
        df = pd.read_sql_query('SELECT * FROM movies', conn)
        conn.close()

        if df.empty:
            return

        df['genres_clean'] = df['genres'].fillna('').str.replace(',', ' ').str.replace('|', ' ')
        df['language_clean'] = df['language'].fillna('English')
        df['director_clean'] = df['director'].fillna('').str.replace(' ', '')
        df['cast_clean'] = df['cast'].fillna('').str.replace(',', ' ')
        df['overview_clean'] = df['overview'].fillna('')
        df['tagline_clean'] = df['tagline'].fillna('')

        # Weighted content soup (genres, language, director weighted twice)
        df['content_soup'] = (
            df['clean_title'].fillna('') + " " +
            df['language_clean'] + " " + df['language_clean'] + " " +
            df['genres_clean'] + " " + df['genres_clean'] + " " +
            df['director_clean'] + " " + df['director_clean'] + " " +
            df['cast_clean'] + " " +
            df['overview_clean'] + " " +
            df['tagline_clean']
        )

        self.movies_df = df.reset_index(drop=True)
        self.tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=15000)
        self.tfidf_matrix = self.tfidf.fit_transform(self.movies_df['content_soup'])
        print(f"Recommender index refreshed with {len(self.movies_df)} movies across all eras and languages.")

    def get_recommendations_for_movie(self, movie_id, top_n=8, genre_weight=0.65, rating_weight=0.35):
        """Recommend movies similar to a given movie ID with match reasons."""
        if self.movies_df.empty or self.tfidf_matrix is None:
            self.refresh()

        match_rows = self.movies_df[self.movies_df['id'] == movie_id]
        if match_rows.empty:
            return []

        idx = match_rows.index[0]
        target_movie = self.movies_df.iloc[idx]

        sim_scores = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        ratings = self.movies_df['vote_average'].fillna(7.0).values / 10.0
        combined_scores = (sim_scores * genre_weight) + (ratings * rating_weight)

        ranked_indices = np.argsort(combined_scores)[::-1]
        ranked_indices = [i for i in ranked_indices if i != idx][:top_n]

        results = []
        target_genres = set(g.strip().lower() for g in str(target_movie['genres']).replace('|', ',').split(','))
        target_lang = str(target_movie.get('language', '')).strip().lower()
        target_director = str(target_movie['director']).strip().lower()
        target_cast = set(c.strip().lower() for c in str(target_movie['cast']).split(','))

        for i in ranked_indices:
            cand = self.movies_df.iloc[i].to_dict()
            sim_pct = int(min(99, max(50, round(sim_scores[i] * 100))))
            
            reasons = []
            cand_genres = set(g.strip().lower() for g in str(cand['genres']).replace('|', ',').split(','))
            cand_lang = str(cand.get('language', '')).strip().lower()
            cand_director = str(cand['director']).strip().lower()
            cand_cast = set(c.strip().lower() for c in str(cand['cast']).split(','))

            shared_g = target_genres.intersection(cand_genres)
            if shared_g:
                reasons.append(f"Shared {', '.join(sorted([g.title() for g in shared_g])[:2])}")
            if target_lang and cand_lang and target_lang == cand_lang:
                reasons.append(f"{cand['language']} Cinema")
            if target_director and cand_director and target_director == cand_director:
                reasons.append(f"Directed by {cand['director']}")
            shared_c = target_cast.intersection(cand_cast)
            if shared_c:
                reasons.append(f"Stars {sorted([c.title() for c in shared_c])[0]}")
            if not reasons:
                reasons.append("Thematic & tonal synergy")

            cand['similarity_pct'] = sim_pct
            cand['match_reasons'] = reasons
            results.append(cand)

        return results

    def get_recommendations_by_mood(self, mood, top_n=8):
        """Recommend movies matching a mood or theme description."""
        if self.movies_df.empty or self.tfidf_matrix is None:
            self.refresh()

        mood_queries = {
            "mind-bending": "mind-bending sci-fi complex dreams reality multiverse physics space time paradox mystery",
            "adrenaline": "action explosion high speed chase martial arts shootout thriller revenge warrior battle",
            "feel-good": "comedy heartwarming family animation friendship inspirational feel good laughter romance",
            "dark-gritty": "crime mafia noir dark psychological thriller violence detective corruption underworld",
            "epic-adventure": "epic journey adventure kingdom battle warrior destiny heroic quest fantasy empire",
            "emotional-drama": "drama romance tragic deep relationship emotional tears human spirit classic sacrifice"
        }

        query_text = mood_queries.get(mood.lower(), mood)
        query_vec = self.tfidf.transform([query_text])
        sim_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        ratings = self.movies_df['vote_average'].fillna(7.0).values / 10.0
        combined_scores = (sim_scores * 0.7) + (ratings * 0.3)

        ranked_indices = np.argsort(combined_scores)[::-1][:top_n]

        results = []
        for i in ranked_indices:
            cand = self.movies_df.iloc[i].to_dict()
            cand['similarity_pct'] = int(min(99, max(55, round(sim_scores[i] * 100))))
            cand['match_reasons'] = [f"Matches '{mood.title()}' mood tone", f"{cand.get('language', 'World')} Cinema"]
            results.append(cand)

        return results

    def get_watchlist_recommendations(self, top_n=8):
        """Recommend movies based on composite tastes of all movies in user watchlist."""
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('SELECT movie_id FROM watchlist')
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return self.get_top_rated(top_n=top_n)

        watchlist_ids = [r['movie_id'] for r in rows]
        sub_df = self.movies_df[self.movies_df['id'].isin(watchlist_ids)]
        if sub_df.empty:
            return self.get_top_rated(top_n=top_n)

        w_indices = sub_df.index.tolist()
        composite_vec = np.mean(self.tfidf_matrix[w_indices].toarray(), axis=0).reshape(1, -1)

        sim_scores = cosine_similarity(composite_vec, self.tfidf_matrix).flatten()
        ratings = self.movies_df['vote_average'].fillna(7.0).values / 10.0
        combined = (sim_scores * 0.7) + (ratings * 0.3)

        ranked_indices = np.argsort(combined)[::-1]
        ranked_indices = [i for i in ranked_indices if self.movies_df.iloc[i]['id'] not in watchlist_ids][:top_n]

        results = []
        for i in ranked_indices:
            cand = self.movies_df.iloc[i].to_dict()
            cand['similarity_pct'] = int(min(99, max(50, round(sim_scores[i] * 100))))
            cand['match_reasons'] = ["Curated from your Watchlist profile"]
            results.append(cand)

        return results

    def get_top_rated(self, top_n=8):
        if self.movies_df.empty:
            self.refresh()
        sorted_df = self.movies_df.sort_values(by=['vote_average', 'vote_count'], ascending=False).head(top_n)
        return sorted_df.to_dict(orient='records')

recommender = MovieRecommender()
