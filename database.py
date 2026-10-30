"""
Database Module for Movie Recommendation System
Manages SQLite database with rich multilingual movie metadata (1900 to 2024),
watchlist, and user ratings across Hindi, English, and World Cinema.
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), 'movies.db')

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()

    # Movies table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS movies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            clean_title TEXT,
            year INTEGER,
            genres TEXT NOT NULL,
            language TEXT DEFAULT 'English',
            director TEXT,
            cast TEXT,
            overview TEXT,
            vote_average REAL DEFAULT 7.0,
            vote_count INTEGER DEFAULT 1000,
            runtime INTEGER DEFAULT 120,
            poster_url TEXT,
            backdrop_url TEXT,
            trailer_url TEXT,
            certification TEXT DEFAULT 'PG-13',
            tagline TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Watchlist table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            movie_id INTEGER UNIQUE,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (movie_id) REFERENCES movies(id) ON DELETE CASCADE
        )
    ''')

    # User Ratings table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_ratings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            movie_id INTEGER,
            rating REAL NOT NULL,
            review TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (movie_id) REFERENCES movies(id) ON DELETE CASCADE
        )
    ''')

    conn.commit()

    cursor.execute('SELECT COUNT(*) FROM movies')
    count = cursor.fetchone()[0]
    if count == 0:
        import seed_data
        seed_data.repopulate_database()

    conn.close()

if __name__ == '__main__':
    init_db()
    print("Database ready at:", DB_PATH)
