import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------- Load model and metadata --------------------
st.set_page_config(page_title="Video Game Sales Predictor", layout="centered")

st.title("🎮 Video Game Global Sales Predictor")

# Load trained SVR model
model = joblib.load('/workspaces/Project_01/video_game_analysis/notebooks/FC110542_Weerakoon/svr_model.pkl')

# Load training columns (to align user input with model expectations)
columns = joblib.load('/workspaces/Project_01/video_game_analysis/notebooks/FC110542_Weerakoon/X_train_columns.pkl')

# -------------------- Input section --------------------
st.header("Enter Game Details")

platform = st.selectbox("Platform", [
    "Wii", "NES", "GB", "DS", "X360", "PS3", "PS2", "SNES", "GBA",
    "PS4", "3DS", "N64", "PS", "XB", "PC", "2600", "PSP", "XOne", "WiiU", "GC", "GEN", "DC"
])

year = st.number_input("Year of Release", min_value=1980, max_value=2025, step=1)

genre = st.selectbox("Genre", [
    "Sports", "Platform", "Racing", "Role-Playing", "Shooter", "Puzzle", "Misc",
    "Simulation", "Action", "Fighting", "Adventure", "Strategy"
])

publisher = st.text_input("Publisher (e.g. Nintendo, EA, Ubisoft, etc.)", "Nintendo")

na_sales = st.number_input("North America Sales (Millions)", min_value=0.0, step=0.01)
eu_sales = st.number_input("Europe Sales (Millions)", min_value=0.0, step=0.01)
jp_sales = st.number_input("Japan Sales (Millions)", min_value=0.0, step=0.01)
other_sales = st.number_input("Other Region Sales (Millions)", min_value=0.0, step=0.01)
user_score = st.number_input("User Score (0.0 - 10.0)", min_value=0.0, max_value=10.0, step=0.1)

# -------------------- Prediction button --------------------
if st.button("Predict Global Sales"):
    # Create a DataFrame from inputs
    input_data = pd.DataFrame({
        "Platform": [platform],
        "Year_of_Release": [year],
        "Genre": [genre],
        "Publisher": [publisher],
        "NA_Sales": [na_sales],
        "EU_Sales": [eu_sales],
        "JP_Sales": [jp_sales],
        "Other_Sales": [other_sales],
        "User_Score": [user_score]
    })

    # One-hot encode the categorical columns
    input_encoded = pd.get_dummies(input_data)

    # Align with training columns (ensure same feature order and missing columns filled with 0)
    input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

    # Predict (no scaler needed)
    prediction_log = model.predict(input_encoded)[0]

    # Convert from log1p scale back to real sales
    predicted_sales = np.expm1(prediction_log)

    # Display result
    st.success(f"💰 Predicted Global Sales: **{predicted_sales:.2f} million units**")

# -------------------- Footer --------------------
st.markdown("---")
st.caption("Built with 💡 SVR Model | Streamlit App by Aashinshana Weerakoon")
