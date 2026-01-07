import streamlit as st
import joblib
import pandas as pd

# Load trained language detection model
model = joblib.load("models/Lang_Detect.pkl")

# Streamlit page settings
st.set_page_config(page_title="Language Detection", layout="wide")

# Page title
st.markdown("""
<div style="background-color:darkblue; padding: 10px; border-radius: 10px;">
    <h1 style="color:white; text-align:center;">🌍 Language Detection</h1>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("sidebar.jpeg")
st.sidebar.header("📞 Contact us")
st.sidebar.text("99999999")

st.sidebar.header("🧑‍🤝‍🧑 About us")
st.sidebar.text("We are a group of ML Engineers working on NLP Projects")

st.text('')

# User input
msg = st.text_input("💬 Enter Your Text", placeholder="Type something to detect language")

if st.button("Predict"):
    resp = model.predict([msg])
    st.title(f"🌐 Detected Language: {resp[0]}")

# Bulk prediction
st.title("📂 Upload file for bulk prediction")
path = st.file_uploader("Upload csv/txt file", type=['csv', 'txt'])

if path is not None:
    df = pd.read_csv(path)   # keep CSV headers
    st.dataframe(df, width=700)
    if st.button("Predict", key='b2'):
        df['Language'] = model.predict(df['text'])  # match 'text' column from CSV
        st.dataframe(df, width=700)
