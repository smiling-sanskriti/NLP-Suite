import streamlit as st
import joblib
import pandas as pd
import io

# Load trained spam detection model
model = joblib.load("models/spam_detection.pkl")

# Streamlit page config
st.set_page_config(page_title="Spam Detection", layout="wide")

# Page title
st.markdown("""
<div style="background-color:darkblue; padding: 10px; border-radius: 10px;">
    <h1 style="color:white; text-align:center;">📧 Spam Message Detection</h1>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("sidebar.jpeg")
st.sidebar.header("📞 Contact us")
st.sidebar.text("99999999")

st.sidebar.header("🧑‍🤝‍🧑 About us")
st.sidebar.text("We are a group of ML Engineers working on NLP Spam Detection")

st.text('')

# ---------------------------
# 🚀 SINGLE MESSAGE PREDICTION
# ---------------------------
msg = st.text_input("💬 Enter Your Message", placeholder="Enter your SMS/Email text")

if st.button("Predict"):
    if msg.strip():  # Only predict if input is not empty
        resp = model.predict([msg])
        if resp[0] == 1:
            st.title("🚨 Spam")
        else:
            st.title("✅ Not Spam")
    else:
        st.warning("Please enter a message before predicting.")

# ---------------------------
# 📁 BULK PREDICTION FROM FILE
# ---------------------------
st.title("📂 Upload file for bulk prediction")
uploaded_file = st.file_uploader("Upload .csv or .txt file", type=['csv', 'txt'])

if uploaded_file is not None:
    try:
        # Handle CSV file
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        # Handle TXT file (one message per line)
        elif uploaded_file.name.endswith(".txt"):
            lines = uploaded_file.read().decode("utf-8").splitlines()
            clean_lines = [line.strip() for line in lines if line.strip()]
            df = pd.DataFrame(clean_lines, columns=["Msg"])

        # Standardize column name if needed
        if 'Msg' not in df.columns:
            df.rename(columns={df.columns[-1]: 'Msg'}, inplace=True)

        # Drop rows with NaN or 'None' string
        df.dropna(subset=["Msg"], inplace=True)
        df = df[df['Msg'].str.lower() != 'none']
        df['Msg'] = df['Msg'].astype(str).str.strip()

        # Show cleaned data
        st.subheader("📄 Preview of Uploaded Data")
        st.dataframe(df, width=700)

        # Predict button
        if st.button("Predict", key='b2'):
            predictions = model.predict(df['Msg'])
            df['Prediction'] = pd.Series(predictions).map({"ham": "✅ Not Spam","spam": "🚨 Spam"})


            st.subheader("📊 Prediction Results")
            st.dataframe(df, width=700)

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
