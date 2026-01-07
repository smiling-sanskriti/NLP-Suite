import joblib
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import requests
from sklearn.neighbors import NearestNeighbors

# ---------------- Page Config ----------------
st.set_page_config(page_title="LENSR eXpert (NLP Suite)", layout="wide")


# ✅ INITIALIZE SESSION STATE (FIXES ERROR)
if "menu_open" not in st.session_state:
    st.session_state.menu_open = False

# 🔴 STEP 1: HIDE DEFAULT STREAMLIT SIDEBAR NAVIGATION
st.markdown(
    """
    <style>
    [data-testid="stSidebarNav"] {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- Header ----------------
st.markdown("""
<div style="background-color:#a7d7d7; padding: 15px; border-radius: 10px;">
    <h1 style="color:black; text-align:center;">🎯 LENSR eXpert (NLP Suite)</h1>
</div>
""", unsafe_allow_html=True)


# ---------------- Sidebar ----------------
with st.sidebar:

    # Hamburger menu (always visible)
    if st.button("☰", key="menu_toggle"):
        st.session_state.menu_open = not st.session_state.menu_open

    st.markdown("---")

    # Branding (always visible)
    st.image("sidebar.jpeg", use_container_width=True)
    st.markdown("## 🎯 LENSR eXpert")
    st.caption("AI-powered NLP Suite")

    st.markdown("---")

    # Menu items (shown only when ☰ is clicked)
    if st.session_state.menu_open:

        with st.expander("🧠 About Project"):
            st.write(
                "LENSR eXpert is an AI-powered NLP suite featuring sentiment analysis, "
                "language detection, spam detection, and movie recommendation systems."
            )

        with st.expander("👥 About Us"):
            st.write(
                "We are a team of Machine Learning engineers working on NLP and "
                "recommender system applications."
            )

        with st.expander("📞 Contact"):
            st.write(
                "📧 Email: team@lensr.ai\n\n"
                "📱 Phone: +91 99999 99999"
            )

        with st.expander("💡 Help"):
            st.write(
                "Use the tabs above to explore different NLP modules."
            )

    st.markdown("---")
    st.caption("v1.0 | Academic Project")


# ---------------- Tabs (Active Project Tags) ----------------
tabs = st.tabs([
    "🍔 Food Sentiment", 
    "🌍 Language Detection", 
    "📧 Spam Detection", 
    "🎬 Movie Recommendation", 
    "🤝 Movie Collaboration"
])

# ---------------- Load Models ----------------
# Keep all your models in "models/" folder
# Make sure you trained and saved these beforehand

# Food Sentiment
try:
    sentiment_model = joblib.load("models/food_review_sent.pkl")
except:
    sentiment_model = None

# Language Detection
try:
    lang_model = joblib.load("models/lang_detect.pkl")
except:
    lang_model = None

# Spam Classifier
try:
    spam_model = joblib.load("models/spam_detection.pkl")
except:
    spam_model = None

# Content-based Movie Recommendation
try:
    df = joblib.load("models/movie_df.pkl")          # DataFrame with movie names
    vectors = joblib.load("models/movie_vectors.pkl")  # Movie vectors
    model = joblib.load("models/movie_model.pkl")      # KNN model
except:
    df, vectors, model = None, None, None

# Collaborative filtering Movie Recommendation
try:
    collab_model = joblib.load("models/target_model.pkl")
    collab_df = joblib.load("models/target_df.pkl")
except:
    collab_model, collab_df = None, None

# 🔧 ALIGN collaborative matrix with movie dataframe
if collab_df is not None and df is not None:
    min_movies = min(collab_df.shape[1], len(df))
    collab_df = collab_df.iloc[:, :min_movies]
    df = df.iloc[:min_movies].reset_index(drop=True)


# ---------------- Functions ----------------
def recommend_content(movie):
    """Dummy function for content-based recs"""
    # Replace with your actual similarity logic
    return ["Movie1", "Movie2", "Movie3", "Movie4", "Movie5"]

def recommend_collaborative(user_id):
    """Dummy function for collaborative filtering recs"""
    # Replace with your actual CF logic
    return ["MovieA", "MovieB", "MovieC", "MovieD", "MovieE"]


# ---------------- Food Sentiment ----------------
with tabs[0]:
    st.header("🍔 Food Review Sentiment Analysis")
    review = st.text_input("Enter Review")

    if st.button("Analyze Sentiment", key="sentiment_btn"):
        if sentiment_model:
            result = sentiment_model.predict([review])[0]

            # Map numeric output to label
            sentiment_map = {
                1: "👍 Liked",
                0: "👎 Disliked"
            }

            sentiment_text = sentiment_map.get(result, "Unknown")

            st.success(f"👉 Sentiment Result: {sentiment_text}")
        else:
            st.error("⚠️ Model not found. Please add sentiment_model.pkl")


# ---------------- Language Detection ----------------
with tabs[1]:
    st.header("🌍 Language Detection")
    text = st.text_input("Enter Text")
    if st.button("Detect Language", key="lang_btn"):
        if lang_model:
            result = lang_model.predict([text])[0]
            st.success(f"👉 Language Detected: {result}")
        else:
            st.error("⚠️ Model not found. Please add language_model.pkl")


# ---------------- Spam Detection ----------------
with tabs[2]:
    st.header("📧 Spam Classifier")
    msg = st.text_input("Enter Message")
    if st.button("Check Spam", key="spam_btn"):
        if spam_model:
            result = spam_model.predict([msg])[0]
            st.warning(f"👉 Prediction: {result}")
        else:
            st.error("⚠️ Model not found. Please add spam_model.pkl")

# ---------------- Content-based Movie Recommendation ----------------
with tabs[3]:
    st.header("🎬 Content-based Movie Recommendation")

    if df is not None:
        movie = st.selectbox("Choose a movie:", sorted(df['name'].values))
    else:
        movie = None

    if st.button("Recommend Movies", key="content_movie_btn"):
        if df is not None and model is not None and vectors is not None:

            movie_index = df[df['name'] == movie].index[0]
            movie_vector = vectors[movie_index]

            distances, indexes = model.kneighbors(
                [movie_vector], n_neighbors=6
            )

            st.write("👉 Top Recommended Movies:")

            cols = st.columns(len(indexes[0]))

            for idx, col in enumerate(cols):
                res_df = df.iloc[indexes[0][idx]]

                # Skip same movie
                if res_df['name'] == movie:
                    continue

                try:
                    resp = requests.get(
                        f"http://www.omdbapi.com/?i={res_df.movie_id}&apikey=cdc55223",
                        timeout=5
                    )
                    poster_url = resp.json().get("Poster")
                except:
                    poster_url = None

                if poster_url and poster_url != "N/A":
                    col.image(
                        poster_url,
                        use_container_width=True,
                        caption=res_df['name']
                    )
                else:
                    col.write(res_df['name'])
                    col.warning("Poster not available")

        else:
            st.error("⚠️ Movie dataset or model not loaded properly.")


# ---------------- Collaborative Movie Recommendation ----------------
with tabs[4]:
    st.header("🤝 Movie Collaboration Recommendation")

    user_id = st.number_input(
        "Enter User ID",
        min_value=0,
        step=1
    )

    if st.button("Recommend Movies"):

        # ---------- Safety checks ----------
        if collab_df is None or df is None:
            st.error("⚠️ Collaborative dataset or movie dataset not loaded.")
            st.stop()

        if user_id >= len(collab_df):
            st.error("❌ User ID not found in dataset.")
            st.stop()

        # ---------- Get ratings for this user ----------
        user_ratings = collab_df.iloc[user_id]

        # Sort movies by rating (highest first)
        top_movies = (
            user_ratings
            .sort_values(ascending=False)
            .head(10)
        )

        valid_movies = []

        # ---------- Map movie indexes safely ----------
        for movie_index in top_movies.index:
            try:
                movie_index = int(movie_index)
            except:
                continue

            if 0 <= movie_index < len(df):
                valid_movies.append(df.iloc[movie_index])

        if not valid_movies:
            st.warning("⚠️ No valid recommendations available for this user.")
            st.stop()

        # Limit display to top 5
        valid_movies = valid_movies[:5]

        st.subheader(f"👉 Recommended Movies for User {user_id}")

        cols = st.columns(len(valid_movies))

        # ---------- Display posters ----------
        for col, movie_row in zip(cols, valid_movies):

            poster_url = None

            try:
                response = requests.get(
                    f"http://www.omdbapi.com/?i={movie_row.movie_id}&apikey=cdc55223",
                    timeout=5
                )
                poster_url = response.json().get("Poster")
            except:
                poster_url = None

            if poster_url and poster_url != "N/A":
                col.image(
                    poster_url,
                    use_container_width=True,
                    caption=movie_row["name"]
                )
            else:
                col.markdown(
                    f"""
                    <div style='text-align:center; padding-top:20px;'>
                        🎬 <b>{movie_row['name']}</b><br>
                        <span style='color:gray; font-size:12px;'>
                            Poster not available
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
