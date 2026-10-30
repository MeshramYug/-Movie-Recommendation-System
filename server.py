"""
Movie Recommendation System - Standalone Flask Server
Port: 5001
Features:
- Vast Multilingual Catalog from 1900 to 2024 (English, Hindi, Telugu, Tamil, Japanese, Korean, Italian, French, etc.)
- Multi-dimensional filters (Era/Decade, Language, Genre, IMDb Rating)
- AI Similarity Engine with Cosine & Bayesian Hybrid Ranking
- Watchlist tracking with SQLite
"""

import os
import sys
import re
from flask import Flask, render_template, request, jsonify
from database import get_db, init_db
from recommender import recommender

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

init_db()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/movies', methods=['GET'])
def get_movies():
    query = request.args.get('q', '').strip().lower()
    genre = request.args.get('genre', '').strip()
    language = request.args.get('language', '').strip()
    decade = request.args.get('decade', '').strip()
    min_rating = float(request.args.get('min_rating', 0.0))
    sort_by = request.args.get('sort_by', 'popular')
    limit = int(request.args.get('limit', 100))

    conn = get_db()
    cursor = conn.cursor()

    sql = 'SELECT * FROM movies WHERE vote_average >= ?'
    params = [min_rating]

    if query:
        sql += ' AND (LOWER(title) LIKE ? OR LOWER(director) LIKE ? OR LOWER(cast) LIKE ? OR LOWER(genres) LIKE ? OR LOWER(language) LIKE ?)'
        wildcard = f'%{query}%'
        params.extend([wildcard, wildcard, wildcard, wildcard, wildcard])

    if genre and genre != 'All':
        sql += ' AND genres LIKE ?'
        params.append(f'%{genre}%')

    if language and language != 'All':
        sql += ' AND LOWER(language) = LOWER(?)'
        params.append(language)

    if decade and decade != 'All':
        if decade == 'Classic (Pre-1960)':
            sql += ' AND year < 1960'
        elif decade == '1960s-1970s':
            sql += ' AND year BETWEEN 1960 AND 1979'
        elif decade == '1980s-1990s':
            sql += ' AND year BETWEEN 1980 AND 1999'
        elif decade == '2000s':
            sql += ' AND year BETWEEN 2000 AND 2009'
        elif decade == '2010s':
            sql += ' AND year BETWEEN 2010 AND 2019'
        elif decade == '2020s (Current)':
            sql += ' AND year >= 2020'

    if sort_by == 'rating':
        sql += ' ORDER BY vote_average DESC, vote_count DESC'
    elif sort_by == 'year_desc':
        sql += ' ORDER BY year DESC, vote_average DESC'
    elif sort_by == 'year_asc':
        sql += ' ORDER BY year ASC, vote_average DESC'
    elif sort_by == 'title':
        sql += ' ORDER BY title ASC'
    else: # popular / default
        sql += ' ORDER BY vote_count DESC, vote_average DESC'

    sql += ' LIMIT ?'
    params.append(limit)

    cursor.execute(sql, params)
    movies = [dict(row) for row in cursor.fetchall()]

    cursor.execute('SELECT movie_id FROM watchlist')
    watchlist_set = set(r['movie_id'] for r in cursor.fetchall())
    for m in movies:
        m['in_watchlist'] = (m['id'] in watchlist_set)

    conn.close()
    return jsonify({"count": len(movies), "movies": movies})

@app.route('/api/movie/<int:movie_id>', methods=['GET'])
def get_movie_detail(movie_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM movies WHERE id = ?', (movie_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return jsonify({"error": "Movie not found"}), 404

    movie = dict(row)
    cursor.execute('SELECT 1 FROM watchlist WHERE movie_id = ?', (movie_id,))
    movie['in_watchlist'] = bool(cursor.fetchone())
    conn.close()

    movie['similar_movies'] = recommender.get_recommendations_for_movie(movie_id, top_n=4)
    return jsonify(movie)

@app.route('/api/recommend', methods=['GET'])
def get_recommendations():
    movie_id = request.args.get('movie_id', type=int)
    mood = request.args.get('mood', '').strip()
    from_watchlist = request.args.get('from_watchlist', 'false').lower() == 'true'
    top_n = int(request.args.get('top_n', 8))

    if movie_id:
        results = recommender.get_recommendations_for_movie(movie_id, top_n=top_n)
    elif mood:
        results = recommender.get_recommendations_by_mood(mood, top_n=top_n)
    elif from_watchlist:
        results = recommender.get_watchlist_recommendations(top_n=top_n)
    else:
        results = recommender.get_top_rated(top_n=top_n)

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT movie_id FROM watchlist')
    watchlist_set = set(r['movie_id'] for r in cursor.fetchall())
    conn.close()

    for r in results:
        r['in_watchlist'] = (r['id'] in watchlist_set)

    return jsonify({"count": len(results), "recommendations": results})

@app.route('/api/watchlist/toggle', methods=['POST'])
def toggle_watchlist():
    data = request.get_json() or {}
    movie_id = data.get('movie_id')
    if not movie_id:
        return jsonify({"error": "Movie ID required"}), 400

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT 1 FROM watchlist WHERE movie_id = ?', (movie_id,))
    exists = cursor.fetchone()

    if exists:
        cursor.execute('DELETE FROM watchlist WHERE movie_id = ?', (movie_id,))
        in_watchlist = False
        message = "Removed from Watchlist"
    else:
        cursor.execute('INSERT INTO watchlist (movie_id) VALUES (?)', (movie_id,))
        in_watchlist = True
        message = "Added to Watchlist"

    conn.commit()
    conn.close()

    return jsonify({"success": True, "in_watchlist": in_watchlist, "message": message})

@app.route('/api/watchlist', methods=['GET'])
def get_watchlist():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT m.*, w.added_at 
        FROM movies m
        JOIN watchlist w ON m.id = w.movie_id
        ORDER BY w.added_at DESC
    ''')
    movies = [dict(row) for row in cursor.fetchall()]
    for m in movies:
        m['in_watchlist'] = True
    conn.close()
    return jsonify({"count": len(movies), "watchlist": movies})

@app.route('/api/filters', methods=['GET'])
def get_filters():
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute('SELECT DISTINCT language FROM movies WHERE language IS NOT NULL ORDER BY language ASC')
    languages = [r['language'] for r in cursor.fetchall()]

    cursor.execute('SELECT genres FROM movies')
    rows = cursor.fetchall()
    conn.close()

    genre_counts = {}
    for r in rows:
        raw = r['genres'] or ''
        parts = re.split(r'[,|]', raw)
        for p in parts:
            p = p.strip()
            if p and p != 'Other':
                genre_counts[p] = genre_counts.get(p, 0) + 1

    sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)

    return jsonify({
        "languages": languages,
        "genres": [{"name": g[0], "count": g[1]} for g in sorted_genres]
    })

@app.route('/api/stats', methods=['GET'])
def get_stats():
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) FROM movies')
    total_movies = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(DISTINCT language) FROM movies')
    total_languages = cursor.fetchone()[0]

    cursor.execute('SELECT MIN(year), MAX(year) FROM movies')
    min_year, max_year = cursor.fetchone()

    cursor.execute('SELECT COUNT(*) FROM movies WHERE year >= 2020')
    modern_movies = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM watchlist')
    watchlist_count = cursor.fetchone()[0]

    cursor.execute('SELECT ROUND(AVG(vote_average), 2) FROM movies')
    avg_rating = cursor.fetchone()[0]

    conn.close()

    return jsonify({
        "total_movies": total_movies,
        "total_languages": total_languages,
        "era_span": f"{min_year} - {max_year}",
        "modern_movies": modern_movies,
        "watchlist_count": watchlist_count,
        "avg_rating": avg_rating
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print("=" * 60)
    print(f"🎬 Movie Recommendation System (1900 - 2024)")
    print(f"📡 Running locally at: http://127.0.0.1:{port}")
    print("=" * 60)
    app.run(host='127.0.0.1', port=port, debug=True)
