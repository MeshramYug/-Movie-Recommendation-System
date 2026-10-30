# 🎬 CineMatch AI — Movie Recommendation System

An intelligent, full-stack Movie Recommendation System featuring interactive content-based filtering, explainable AI (XAI) recommendations, a rich cinematic dark UI, and an online movie input studio.

---

## 🌟 Key Features

1. **Independent Local Server (Flask)**:
   - Built on Flask + SQLite for fast, offline, local operation without Streamlit dependencies.
   - RESTful API endpoints for filtering, searching, bookmarking, and generating recommendations.

2. **Curated Modern & Online Releases**:
   - Seeded with 40+ high-profile current blockbusters and cinema masterpieces (*Dune: Part Two*, *Oppenheimer*, *Spider-Man: Across the Spider-Verse*, *Deadpool & Wolverine*, *Gladiator II*, *Alien: Romulus*, *Interstellar*, *Inception*, *The Dark Knight*, etc.).
   - Full rich movie metadata: HD poster art, backdrop banners, release years, runtimes, certifications, IMDb ratings, directing styles, and top cast ensembles.

3. **Online Movie Input Studio ("Add New Movie")**:
   - Allows users to input new online movies directly via an interactive UI form.
   - Automatically saves to SQLite and updates the TF-IDF recommendation index in real time.
   - Includes 1-click preset templates for instant testing.

4. **Interactive Cinematic Web Interface**:
   - **Discover & Browse**: Filter by 14+ genres, eras (2020s, 2010s, 2000s, 1990s), IMDb rating slider, and live search.
   - **AI Recommender Studio**: Recommend by seed movie or choose by mood (*Mind-Bending Sci-Fi*, *Adrenaline Rush*, *Feel-Good*, *Dark Crime*, *Epic Adventure*, *Heartfelt Drama*).
   - **Movie Detail Modal with Video Trailer**: Backdrop banner, official YouTube trailer preview, full cast, synopsis, and "More Like This" carousel.
   - **My Watchlist**: Save movies to personal favorites with real-time counters.
   - **Platform Analytics**: Dynamic metrics on catalog breakdown, modern titles, and genre distribution.

5. **1-Click Launchers**:
   - `run_movie_recommender.bat` automatically verifies SQLite, starts the server on port `5001`, and opens your web browser at `http://127.0.0.1:5001`.

---

## 🚀 How to Run

### Option 1: 1-Click Batch Launcher (Recommended)
Double click:
```cmd
run_movie_recommender.bat
```

### Option 2: Command Line
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the application server
python server.py
```
Open your browser at: **`http://127.0.0.1:5001`**

---

## 📡 API Endpoints

- `GET /` — Main Interactive Cinematic Dashboard
- `GET /api/movies` — Filtered catalog (search query, genre, era, rating, sort order)
- `GET /api/movie/<id>` — Full movie details with trailer preview and similar titles
- `GET /api/recommend` — Recommendation engine (by `movie_id`, `mood`, or `from_watchlist`)
- `POST /api/movies/add` — Add a new online movie and trigger live re-indexing
- `POST /api/watchlist/toggle` — Add/remove movie from watchlist
- `GET /api/watchlist` — Get all saved movies
- `GET /api/genres` — List of unique genres with title counts
- `GET /api/stats` — Live statistics and metrics

---

## 🛠️ Tech Stack
- **Backend**: Python 3, Flask, SQLite3
- **Machine Learning**: Scikit-Learn (TF-IDF Vectorizer, Cosine Similarity, Bayesian Weighted Rating)
- **Frontend**: HTML5, Vanilla JavaScript (ES6+), Modern Responsive CSS3 (Flexbox/Grid, Dark Cinematic Theme)
- **Icons & Fonts**: FontAwesome 6, Google Fonts (*Plus Jakarta Sans*, *Space Grotesk*)
