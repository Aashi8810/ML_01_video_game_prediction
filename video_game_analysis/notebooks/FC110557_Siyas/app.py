import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load model and encoders
# -------------------------------
model = joblib.load("/workspaces/ML_01_video_game_prediction/video_game_analysis/notebooks/FC110557_Siyas/xgboost_model.pkl")
encoders = joblib.load("/workspaces/ML_01_video_game_prediction/video_game_analysis/notebooks/FC110557_Siyas/encoded.fil")

st.set_page_config(page_title="Video Game Sales Predictor", page_icon="🎮", layout="centered")

st.title("🎮 Video Game Sales Prediction App")
st.write("Enter game details below to predict **Global Sales (in millions)**.")

# -------------------------------
# Input Widgets
# -------------------------------
genre = st.selectbox("Select Genre", encoders["Genre"].classes_)
platform = st.selectbox("Select Platform", encoders["Platform"].classes_)
publisher = st.selectbox("Select Publisher", encoders["Publisher"].classes_)


year = st.slider("Year of Release", 1980, 2025, 2010)
critic_score = st.slider("Critic Score", 0, 100, 70)
user_score = st.slider("User Score", 0, 10, 7)

# -------------------------------
# Encode Categorical Inputs
# -------------------------------
# Handle unseen values by mapping to "Unknown"
def safe_encode(encoder, value):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        return encoder.transform(["Unknown"])[0]

encoded_genre = safe_encode(encoders["Genre"], genre)
encoded_platform = safe_encode(encoders["Platform"], platform)
encoded_publisher = safe_encode(encoders["Publisher"], publisher)

na_sales = 0.0
eu_sales = 0.0
jp_sales = 0.0
other_sales = 0.0
critic_count = 0
user_count = 0
rating = "Unknown"  # will be encoded
total_rating = (critic_score + user_score) / 2

encoded_rating = safe_encode(encoders["Rating"], rating)

# -------------------------------
# Create Input DataFrame
# -------------------------------
input_data = pd.DataFrame({
    "Platform": [encoded_platform],
    "Year_of_Release": [year],
    "Genre": [encoded_genre],
    "Publisher": [encoded_publisher],
    "NA_Sales": [na_sales],
    "EU_Sales": [eu_sales],
    "JP_Sales": [jp_sales],
    "Other_Sales": [other_sales],
    "Critic_Score": [critic_score],
    "Critic_Count": [critic_count],
    "User_Score": [user_score],
    "User_Count": [user_count],
    "Rating": [encoded_rating],
    "Total_Rating": [total_rating]
})


# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Sales"):
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Predicted Global Sales: **{prediction:.2f} million copies**")
