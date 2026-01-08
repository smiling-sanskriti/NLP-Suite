import streamlit as st
import joblib
import pandas as pd
import requests 

df=joblib.load("models/movie_df.pkl")
model=joblib.load("models/movie_model.pkl")
vectors=joblib.load("models/movie_vectors.pkl")

st.set_page_config(page_title="Movie Recommendation System", layout="wide")

st.markdown("""
<div style="background-color:brown; padding: 10px; border-radius: 10px;">
    <h1 style="color:white; text-align:center;">Movie Recommendation System</h1>
</div>
""", unsafe_allow_html=True)

st.sidebar.image("sidebar.jpeg")
st.sidebar.header("📞Contact us")
st.sidebar.text("99999999")

st.sidebar.header("🧑‍🤝‍🧑About us")
st.sidebar.text("we are a group of ML Engineers working on Sentiment Analysis")

st.text('')

movie = st.selectbox("Choose a movie:", sorted(df.name))

if st.button("Recommend"):
    if movie in df.name.values:
        movie_index = df[df.name == movie].index[0]
        movie_vector = vectors[movie_index]
        distances, indexes = model.kneighbors([movie_vector], n_neighbors=5)

        # Create columns for posters
        cols = st.columns(len(indexes[0]))

        for idx, col in enumerate(cols):
            res_df = df.iloc[indexes[0][idx]]
            #st.write(res_df.name)  # show movie name
            resp = requests.get(f"http://www.omdbapi.com/?i={res_df.movie_id}&apikey=cdc55223")
            poster_url = resp.json().get('Poster')
            if poster_url:
                col.image(poster_url, use_container_width=True, caption=res_df['name'])
               
    else:
        st.error('Movie not found, please select different name')

