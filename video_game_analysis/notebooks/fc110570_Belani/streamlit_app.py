# How to run -> navigate to fc110570_Belani : streamlit run streamlit_app.py

import streamlit as st
import pandas as pd
import joblib
from catboost import CatBoostRegressor, Pool

# Load trained model
model_path = "../../models/fc110570_catboost_tuned_model.pkl"
model = joblib.load(model_path)

st.set_page_config(page_title="Game Sales Predictor", layout="centered")

st.title("Video Game Sales Prediction App")
st.write("This app predicts the **global sales** of a video game using the tuned CatBoost model.")

# Known categories from training 
known_platforms = ['PS4', 'XOne', 'PC', 'Switch', 'PS3', 'WiiU']
known_genres = ['Action', 'Sports', 'Racing', 'Role-Playing', 'Shooter', 'Misc']
known_publishers = ['Nintendo', 'Ubisoft', 'EA', 'Activision', 'Sony', 'Capcom', 'Square Enix', 'Other']
known_developers = ['GameFreak', 'Nintendo EPD', 'Ubisoft Montreal', 'EA Vancouver', 'SIE Japan', 'Other']
known_ratings = ['E', 'T', 'M', 'E10+', 'RP']

# User Input Form 
st.subheader("Enter Game Details")

platform = st.selectbox("Platform", known_platforms)
year = st.number_input("Year of Release", min_value=1980, max_value=2025, value=2016)
genre = st.selectbox("Genre", known_genres)
publisher = st.selectbox("Publisher", known_publishers, index=0)
na_sales = st.number_input("North America Sales (in millions)", min_value=0.0, value=1.0)
eu_sales = st.number_input("Europe Sales (in millions)", min_value=0.0, value=0.5)
jp_sales = st.number_input("Japan Sales (in millions)", min_value=0.0, value=0.3)
other_sales = st.number_input("Other Regions Sales (in millions)", min_value=0.0, value=0.2)
critic_score = st.number_input("Critic Score", min_value=0.0, max_value=100.0, value=75.0)
critic_count = st.number_input("Critic Count", min_value=0, value=50)
user_score = st.number_input("User Score", min_value=0.0, max_value=10.0, value=7.5)
user_count = st.number_input("User Count", min_value=0, value=100)
developer = st.selectbox("Developer", known_developers, index=0)
rating = st.selectbox("Rating", known_ratings)

# Prepare Input DataFrame 
input_data = pd.DataFrame({
    'Platform': [platform],
    'Year': [year],
    'Genre': [genre],
    'Publisher': [publisher],
    'NA_Sales': [na_sales],
    'EU_Sales': [eu_sales],
    'JP_Sales': [jp_sales],
    'Other_Sales': [other_sales],
    'CriticScore': [critic_score],
    'Critic_Count': [critic_count],
    'UserScore': [user_score],
    'User_Count': [user_count],
    'Developer': [developer],
    'Rating': [rating]
})

# Match the exact feature order from training 
features = [
    'Platform', 'Year', 'Genre', 'Publisher',
    'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales',
    'CriticScore', 'Critic_Count', 'UserScore', 'User_Count',
    'Developer', 'Rating'
]
input_data = input_data[features]

# Specify categorical features 
categorical_features = ['Platform', 'Genre', 'Publisher', 'Developer', 'Rating']

# Create CatBoost Pool 
input_pool = Pool(data=input_data, cat_features=categorical_features)

# Prediction 
if st.button("Predict Global Sales"):
    prediction = model.predict(input_pool)
    st.success(f"Predicted Global Sales: **{prediction[0]:.2f} million units**")

