"""
Movie Recommendation System - Standalone Flask Server
Port: 5001
Features:
- Full REST API for Movie Discovery & Recommendations
- AI Similarity Engine with Cosine & Bayesian Hybrid Ranking
- Live Add-Movie Studio for online & new releases
- Watchlist & Ratings tracking with SQLite
"""

import os
import sys
import re
from flask import Flask, render_template, request, jsonify
from database import get_db, init_db
from recommender import recommender

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Ensure database is initialized at startup
init_db()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/movies', methods=['GET'])
def get_movies():
    query = request.args.get('q', '').strip().lower()
    genre = request.args.get('genre', '').strip()
    decade = request.args.get('decade', '').strip()
    min_rating = float(request.args.get('min_rating', 0.0))
    sort_by = request.args.get('sort_by', 'popular') # popular, rating, year_desc, title
    limit = int(request.args.get('limit', 50))

    conn = get_db()
    cursor = conn.cursor()

    sql = 'SELECT * FROM movies WHERE vote_average >= ?'
    params = [min_rating]

    if query:
        sql += ' AND (LOWER(title) LIKE ? OR LOWER(director) LIKE ? OR LOWER(cast) LIKE ? OR LOWER(genres) LIKE ?)'
        wildcard = f'%{query}%'
        params.extend([wildcard, wildcard, wildcard, wildcard])

    if genre and genre != 'All':
        sql += ' AND genres LIKE ?'
        params.append(f'%{genre}%')

    if decade and decade != 'All':
        try:
            start_year = int(decade.replace('s', ''))
            end_year = start_year + 9
            sql += ' AND year BETWEEN ? AND ?'
            params.extend([start_year, end_year])
        except ValueError:
            pass

    if sort_by == 'rating':
        sql += ' ORDER BY vote_average DESC, vote_count DESC'
    elif sort_by == 'year_desc':
        sql += ' ORDER BY year DESC, vote_average DESC'
    elif sort_by == 'title':
        sql += ' ORDER BY title ASC'
    else: # popular / default
        sql += ' ORDER BY vote_count DESC, vote_average DESC'

    sql += ' LIMIT ?'
    params.append(limit)

    cursor.execute(sql, params)
    movies = [dict(row) for row in cursor.fetchall()]

    # Check watchlist status for each movie
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

    # Fetch top 4 recommendations for this movie
    movie['similar_movies'] = recommender.get_recommendations_for_movie(movie_id, top_n=4)
    return jsonify(movie)

@app.route('/api/recommend', methods=['GET'])
def get_recommendations():
    movie_id = request.args.get('movie_id', type=int)
    mood = request.args.get('mood', '').strip()
    from_watchlist = request.args.get('from_watchlist', 'false').lower() == 'true'
    top_n = int(request.args.get('top_n', 8))
    genre_weight = float(request.args.get('genre_weight', 0.6))
    rating_weight = float(request.args.get('rating_weight', 0.4))

    if movie_id:
        results = recommender.get_recommendations_for_movie(
            movie_id, top_n=top_n, genre_weight=genre_weight, rating_weight=rating_weight
        )
    elif mood:
        results = recommender.get_recommendations_by_mood(mood, top_n=top_n)
    elif from_watchlist:
        results = recommender.get_watchlist_recommendations(top_n=top_n)
    else:
        results = recommender.get_top_rated(top_n=top_n)

    # Attach in_watchlist status
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT movie_id FROM watchlist')
    watchlist_set = set(r['movie_id'] for r in cursor.fetchall())
    conn.close()

    for r in results:
        r['in_watchlist'] = (r['id'] in watchlist_set)

    return jsonify({"count": len(results), "recommendations": results})

@app.route('/api/movies/add', methods=['POST'])
def add_new_movie():
    """Allows user to input new online and current movies directly into the live catalog."""
    data = request.get_json() or request.form.to_dict()
    
    title = data.get('title', '').strip()
    if not title:
        return jsonify({"error": "Movie title is required"}), 400

    # Auto extract year if present in title
    year = data.get('year')
    if not year:
        match = re.search(r'\((\d{4})\)', title)
        year = int(match.group(1)) if match else 2024
    else:
        try:
            year = int(year)
        except ValueError:
            year = 2024

    clean_title = re.sub(r'\s*\(\d{4}\)\s*$', '', title).strip()
    formatted_title = f"{clean_title} ({year})"

    genres = data.get('genres', 'Drama, Thriller').strip()
    director = data.get('director', 'Independent Director').strip()
    cast = data.get('cast', 'Leading Cast').strip()
    overview = data.get('overview', 'A gripping cinematic story.').strip()
    
    try:
        vote_average = float(data.get('vote_average', 7.5))
    except (ValueError, TypeError):
        vote_average = 7.5

    try:
        runtime = int(data.get('runtime', 120))
    except (ValueError, TypeError):
        runtime = 120

    poster_url = data.get('poster_url', '').strip() or 'https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&auto=format&fit=crop'
    backdrop_url = data.get('backdrop_url', '').strip() or 'https://images.unsplash.com/photo-1517604931442-7e0c8ed2963c?w=1280&auto=format&fit=crop'
    trailer_url = data.get('trailer_url', '').strip()
    certification = data.get('certification', 'PG-13').strip()
    tagline = data.get('tagline', '').strip()

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO movies (
            title, clean_title, year, genres, director, cast, overview,
            vote_average, vote_count, runtime, poster_url, backdrop_url,
            trailer_url, certification, tagline, is_custom
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
    ''', (
        formatted_title, clean_title, year, genres, director, cast, overview,
        vote_average, 1500, runtime, poster_url, backdrop_url,
        trailer_url, certification, tagline
    ))
    new_id = cursor.lastrowid
    conn.commit()
    conn.close()

    # Re-index recommender with newly added movie
    recommender.refresh()

    return jsonify({
        "success": True,
        "message": f"Successfully added '{formatted_title}' to live catalog and recommendation engine!",
        "movie_id": new_id
    }), 201

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

@app.route('/api/genres', methods=['GET'])
def get_genres():
    conn = get_db()
    cursor = conn.cursor()
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
    return jsonify({"genres": [{"name": g[0], "count": g[1]} for g in sorted_genres]})

@app.route('/api/stats', methods=['GET'])
def get_stats():
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) FROM movies')
    total_movies = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM movies WHERE year >= 2020')
    modern_movies = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM movies WHERE is_custom = 1')
    custom_movies = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM watchlist')
    watchlist_count = cursor.fetchone()[0]

    cursor.execute('SELECT ROUND(AVG(vote_average), 2) FROM movies')
    avg_rating = cursor.fetchone()[0]

    conn.close()

    return jsonify({
        "total_movies": total_movies,
        "modern_movies": modern_movies,
        "custom_movies": custom_movies,
        "watchlist_count": watchlist_count,
        "avg_rating": avg_rating
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print("=" * 60)
    print(f"🎬 Movie Recommendation System Local Server")
    print(f"📡 Running locally at: http://127.0.0.1:{port}")
    print(f"💡 Press Ctrl+C in terminal to stop server")
    print("=" * 60)
    app.run(host='127.0.0.1', port=port, debug=True)
