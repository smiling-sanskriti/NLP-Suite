import streamlit as st
import joblib
import pandas as pd

# -------------------------------------------------
# 1️⃣ MUST be the first Streamlit command
# -------------------------------------------------
st.set_page_config(page_title="Sentiment Analysis", layout="wide")

# -------------------------------------------------
# 2️⃣ Safe model loading with error handling
# -------------------------------------------------
try:
    model = joblib.load("models/food_review_sent.pkl")
except FileNotFoundError:
    st.error("❌ Model file 'food_review_sent.pkl' not found in models folder")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# -------------------------------------------------
# UI Header
# -------------------------------------------------
st.markdown("""
<div style="background-color:brown; padding: 10px; border-radius: 10px;">
    <h1 style="color:white; text-align:center;">🍽️ Food Sentiment Prediction</h1>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.image("sidebar.jpeg")
st.sidebar.header("📞 Contact us")
st.sidebar.text("99999999")

st.sidebar.header("🧑‍🤝‍🧑 About us")
st.sidebar.text("We are a group of ML Engineers working on Sentiment Analysis")

st.text("")

# -------------------------------------------------
# 3️⃣ Single Prediction (empty input handled)
# -------------------------------------------------
msg = st.text_input("💬 Enter Your Message", placeholder="Enter your review")

if st.button("Predict"):
    if msg.strip() == "":
        st.warning("⚠️ Please enter a review before predicting")
    else:
        resp = model.predict([msg])[0]
        if resp == 0:
            st.title("👎 Dislike")
        else:
            st.title("👍 Like")
            st.balloons()

# -------------------------------------------------
# 4️⃣ Bulk Prediction (robust CSV handling)
# -------------------------------------------------
st.title("📂 Upload file for bulk prediction")

path = st.file_uploader("Upload CSV file", type=['csv', 'txt'])

if path is not None:
    df = pd.read_csv(path)

    # Handle CSV with or without header
    if df.shape[1] == 1:
        df.columns = ['Msg']

    st.dataframe(df, width=700)

    if st.button("Predict", key='b2'):
        df['Sentiment'] = model.predict(df['Msg'])
        df['Sentiment'] = df['Sentiment'].map({
            0: "👎 Dislike",
            1: "👍 Like"
        })
        st.dataframe(df, width=700)
