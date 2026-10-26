import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import json
import os

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .movie-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 14px;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .movie-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    }
    .genre-tag {
        background-color: #EEF2F6;
        color: #475569;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 4px;
    }
    .star-score {
        color: #EAB308;
        font-weight: 700;
    }
    .match-pill {
        background-color: #DCFCE7;
        color: #166534;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Recommender Artifacts
# ---------------------------------------------------------
@st.cache_resource
def load_movie_recommender():
    base_dir = os.path.dirname(__file__)
    models_dir = os.path.join(base_dir, 'models')
    
    rec_path = os.path.join(models_dir, 'movie_recommender.pkl')
    meta_path = os.path.join(models_dir, 'metadata.json')
    
    recommender_data = joblib.load(rec_path)
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
        
    pool_df = recommender_data['pool_df']
    tfidf_matrix = recommender_data['tfidf_matrix']
    tfidf_vectorizer = recommender_data['tfidf_vectorizer']
    
    return pool_df, tfidf_matrix, tfidf_vectorizer, metadata

try:
    pool_df, tfidf_matrix, tfidf_vectorizer, metadata = load_movie_recommender()
except Exception as e:
    st.error(f"Error loading movie recommender: {e}. Please ensure 'train_model.py' has been executed.")
    st.stop()

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.markdown("## 🎬 Movie System")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    [
        "🎬 Movie Recommender",
        "🏆 Top Charts & Genre Explorer",
        "🎯 Preference Watchlist Builder",
        "📊 Cinema & Ratings Insights",
        "🔬 Model Architecture & Formula"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Recommendation Pool:** `{len(pool_df):,} movies`")
st.sidebar.markdown(f"**Ratings Ingested:** `25,000,000+`")
st.sidebar.markdown(f"**Algorithm:** `TF-IDF + Bayesian WR`")

# ---------------------------------------------------------
# Page 1: Movie Recommender
# ---------------------------------------------------------
if page == "🎬 Movie Recommender":
    st.markdown("<div class='main-header'>Movie Recommendation System</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Content-based similarity engine paired with Bayesian weighted rating scoring.</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        # Default to a famous classic
        default_idx = int(pool_df[pool_df['clean_title'].str.contains('Inception', case=False, na=False)].index[0]) if len(pool_df[pool_df['clean_title'].str.contains('Inception', case=False, na=False)]) > 0 else 0
        selected_movie_title = st.selectbox(
            "Select or Search a Movie:",
            options=pool_df['title'].tolist(),
            index=default_idx
        )
    with col2:
        num_recommendations = st.slider("Recommendations Count", 3, 20, 6)
    with col3:
        min_rating = st.slider("Min Bayesian Rating", 1.0, 4.5, 2.5, 0.1)

    if selected_movie_title:
        selected_idx = pool_df[pool_df['title'] == selected_movie_title].index[0]
        selected_row = pool_df.iloc[selected_idx]

        # Display Selected Movie Highlight Card
        genres_badges = " ".join([f"<span class='genre-tag'>{g.strip()}</span>" for g in str(selected_row['genres']).split('|')])
        year_str = f"({int(selected_row['year'])})" if pd.notna(selected_row['year']) else ""
        
        st.markdown(f"""
        <div style='background:#F8FAFC; border:1px solid #CBD5E1; border-radius:10px; padding:18px 22px; margin-bottom:20px;'>
            <div style='font-size:1.3rem; font-weight:700; color:#0F172A; margin-bottom:6px;'>
                {selected_row['clean_title']} <span style='color:#64748B; font-weight:400;'>{year_str}</span>
            </div>
            <div style='margin-bottom:10px;'>{genres_badges}</div>
            <div style='font-size:0.95rem; color:#334155;'>
                ⭐ <strong>Bayesian Rating:</strong> {selected_row['weighted_rating']} / 5.0 &nbsp;&nbsp;|&nbsp;&nbsp; 
                👥 <strong>Community Ratings:</strong> {int(selected_row['vote_count']):,} votes &nbsp;&nbsp;|&nbsp;&nbsp; 
                📊 <strong>Avg Rating:</strong> {selected_row['vote_average']} / 5.0
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Compute Cosine Similarity
        cosine_sim = cosine_similarity(tfidf_matrix[selected_idx], tfidf_matrix).flatten()
        
        # Sort indices excluding self
        top_indices = cosine_sim.argsort()[::-1]
        recommended_indices = []
        for idx in top_indices:
            if idx == selected_idx:
                continue
            if pool_df.iloc[idx]['weighted_rating'] >= min_rating:
                recommended_indices.append(idx)
            if len(recommended_indices) >= num_recommendations:
                break

        st.subheader("Recommended Similar Movies")

        # Display in 2 or 3 columns
        cols = st.columns(2)
        for i, r_idx in enumerate(recommended_indices):
            r_movie = pool_df.iloc[r_idx]
            match_pct = round(cosine_sim[r_idx] * 100, 1)
            r_year = f"({int(r_movie['year'])})" if pd.notna(r_movie['year']) else ""
            r_genres = " ".join([f"<span class='genre-tag'>{g.strip()}</span>" for g in str(r_movie['genres']).split('|')])

            col_target = cols[i % 2]
            with col_target:
                col_target.markdown(f"""
                <div class='movie-card'>
                    <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;'>
                        <span style='font-weight:700; font-size:1.05rem; color:#1E293B;'>#{i+1}. {r_movie['clean_title']} {r_year}</span>
                        <span class='match-pill'>{match_pct}% Match</span>
                    </div>
                    <div style='margin-bottom:8px;'>{r_genres}</div>
                    <div style='font-size:0.85rem; color:#475569;'>
                        <span class='star-score'>★ {r_movie['weighted_rating']}</span> &nbsp;({int(r_movie['vote_count']):,} votes) &nbsp;|&nbsp; Avg: {r_movie['vote_average']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Page 2: Top Charts & Genre Explorer
# ---------------------------------------------------------
elif page == "🏆 Top Charts & Genre Explorer":
    st.markdown("<div class='main-header'>Top Charts & Genre Explorer</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Top-rated films ranked by IMDb Bayesian Weighted Rating formula ($m=100$).</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        genre_filter = st.selectbox("Filter by Genre", ["All Genres"] + metadata['genres'])
    with col2:
        decade_filter = st.selectbox("Filter by Decade", ["All Decades"] + metadata['decades'])
    with col3:
        min_votes = st.slider("Minimum Vote Count", 50, 5000, 250, 50)

    # Filter
    charts_df = pool_df.copy()
    if genre_filter != "All Genres":
        charts_df = charts_df[charts_df['genres'].str.contains(genre_filter, case=False, na=False)]
    if decade_filter != "All Decades":
        charts_df = charts_df[charts_df['decade'] == decade_filter]
    charts_df = charts_df[charts_df['vote_count'] >= min_votes]

    top_charts = charts_df.sort_values('weighted_rating', ascending=False).head(50).reset_index(drop=True)

    st.markdown(f"Displaying Top **{len(top_charts)}** Movies in this category")

    display_df = top_charts[['clean_title', 'year', 'genres', 'weighted_rating', 'vote_average', 'vote_count']]
    display_df.columns = ['Title', 'Year', 'Genres', 'Bayesian Rating', 'Avg Community Rating', 'Total Votes']
    
    st.dataframe(
        display_df.style.format({
            'Year': '{:.0f}',
            'Bayesian Rating': '{:.2f}',
            'Avg Community Rating': '{:.2f}',
            'Total Votes': '{:,}'
        }),
        use_container_width=True,
        height=450
    )

    # Visual Bar Chart of Top 10
    if len(top_charts) > 0:
        st.subheader("Top 10 Rated Films in Selection")
        fig_top10 = px.bar(
            top_charts.head(10),
            x='weighted_rating', y='clean_title', orientation='h',
            text='weighted_rating',
            labels={'weighted_rating': 'Bayesian Weighted Rating (1-5)', 'clean_title': 'Movie Title'},
            color='weighted_rating', color_continuous_scale='Viridis'
        )
        fig_top10.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_top10, use_container_width=True)

# ---------------------------------------------------------
# Page 3: Preference Watchlist Builder
# ---------------------------------------------------------
elif page == "🎯 Preference Watchlist Builder":
    st.markdown("<div class='main-header'>Personalized Watchlist Builder</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Select your taste preferences to generate an instant tailored watchlist.</div>", unsafe_allow_html=True)

    with st.form("taste_form"):
        col1, col2 = st.columns(2)
        with col1:
            fav_genres = st.multiselect("Select Favorite Genres (1 to 3)", metadata['genres'], default=['Sci-Fi', 'Action'])
            fav_decades = st.multiselect("Preferred Era / Decades", metadata['decades'], default=['1990s', '2000s', '2010s'])
        with col2:
            min_score = st.slider("Minimum Bayesian Rating Target", 2.0, 4.5, 3.5, 0.1)
            target_count = st.slider("Watchlist Size", 5, 25, 10)

        submitted = st.form_submit_button("Generate Personalized Watchlist", use_container_width=True)

    if submitted:
        if not fav_genres:
            st.warning("Please select at least one genre to build your recommendation watchlist.")
        else:
            w_df = pool_df.copy()
            # Match any selected genre
            genre_pattern = "|".join(fav_genres)
            w_df = w_df[w_df['genres'].str.contains(genre_pattern, case=False, na=False)]
            
            if fav_decades:
                w_df = w_df[w_df['decade'].isin(fav_decades)]
                
            w_df = w_df[w_df['weighted_rating'] >= min_score]
            curated = w_df.sort_values(['weighted_rating', 'vote_count'], ascending=[False, False]).head(target_count)

            st.markdown(f"### Curated Watchlist ({len(curated)} Movies Found)")
            for rank, (_, row) in enumerate(curated.iterrows(), start=1):
                y_str = f"({int(row['year'])})" if pd.notna(row['year']) else ""
                g_str = " ".join([f"<span class='genre-tag'>{g}</span>" for g in row['genres'].split('|')])
                
                st.markdown(f"""
                <div class='movie-card'>
                    <div style='display:flex; justify-content:space-between; align-items:center;'>
                        <span style='font-size:1.1rem; font-weight:700;'>#{rank}. {row['clean_title']} <span style='color:#64748B; font-weight:400;'>{y_str}</span></span>
                        <span class='star-score'>★ {row['weighted_rating']} / 5.0</span>
                    </div>
                    <div style='margin-top:6px;'>{g_str}</div>
                    <div style='font-size:0.85rem; color:#64748B; margin-top:6px;'>
                        Total Community Ratings: {int(row['vote_count']):,} votes &nbsp;|&nbsp; Raw Average: {row['vote_average']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Page 4: Cinema & Ratings Insights
# ---------------------------------------------------------
elif page == "📊 Cinema & Ratings Insights":
    st.markdown("<div class='main-header'>Cinema & Ratings Analytics</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Empirical analysis of 25M movie ratings, vote distributions, and genre statistics.</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Movie Count by Genre")
        genre_counts = {}
        for g_list in pool_df['genres'].dropna():
            for g in g_list.split('|'):
                g = g.strip()
                if g and g != 'Other':
                    genre_counts[g] = genre_counts.get(g, 0) + 1
        gdf = pd.DataFrame(list(genre_counts.items()), columns=['Genre', 'Count']).sort_values('Count', ascending=False)
        fig_genre = px.bar(
            gdf.head(12), x='Count', y='Genre', orientation='h',
            text='Count', color_discrete_sequence=['#3B82F6']
        )
        fig_genre.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_genre, use_container_width=True)

    with c2:
        st.subheader("Weighted Rating Distribution")
        fig_ratings = px.histogram(
            pool_df, x='weighted_rating', nbins=30,
            color_discrete_sequence=['#10B981'],
            marginal="box",
            labels={'weighted_rating': 'Bayesian Weighted Rating'}
        )
        fig_ratings.update_layout(margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_ratings, use_container_width=True)

    st.subheader("Popularity vs. Rating Quality Scatter")
    sample_movies = pool_df.sample(min(1200, len(pool_df)), random_state=42).copy()
    fig_scatter = px.scatter(
        sample_movies, x='vote_count', y='weighted_rating',
        hover_name='clean_title',
        labels={'vote_count': 'Total Votes (Log Scale)', 'weighted_rating': 'Weighted Rating (1-5)'},
        log_x=True,
        opacity=0.65,
        color='weighted_rating',
        color_continuous_scale="Turbo"
    )
    fig_scatter.update_layout(margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig_scatter, use_container_width=True)

# ---------------------------------------------------------
# Page 5: Model Architecture & Formula
# ---------------------------------------------------------
elif page == "🔬 Model Architecture & Formula":
    st.markdown("<div class='main-header'>Recommendation Architecture & Mathematical Formulation</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>System mechanics, mathematical formulas, and algorithmic design.</div>", unsafe_allow_html=True)

    st.markdown("### 1. IMDb Bayesian Weighted Rating Formulation")
    st.markdown("""
    To avoid bias where a movie with a single 5-star rating ranks above a masterpiece with 80,000 ratings averaging 4.5, we compute the Bayesian Weighted Rating ($WR$):
    """)
    st.latex(r"WR = \left(\frac{v}{v + m}\right) \cdot R + \left(\frac{m}{v + m}\right) \cdot C")
    st.markdown("""
    Where:
    - **$v$**: Number of ratings for the movie (`vote_count`)
    - **$m$**: Minimum vote threshold required for rating stability (default $m=100$)
    - **$R$**: Mean community rating of the movie (`vote_average`)
    - **$C$**: Mean rating across the entire catalog ($C \\approx 3.12$)
    """)

    st.markdown("---")
    st.markdown("### 2. Content-Based TF-IDF Vector Space Model")
    st.markdown("""
    - **Vocabulary:** 8,000 unigrams and bigrams extracted from film title keywords, genre tokens (double weighted), and release decades.
    - **Term Frequency - Inverse Document Frequency ($TF\\text{-}IDF$):**
    """)
    st.latex(r"\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{|D|}{1 + |\{d \in D : t \in d\}|}\right)")
    st.markdown("""
    - **Pairwise Similarity:** Cosine of the angle between feature vectors:
    """)
    st.latex(r"\text{Cosine Similarity}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2}")
