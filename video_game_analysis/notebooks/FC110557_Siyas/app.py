# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# 1. Load Model, Encoders, and Dataset
# -------------------------------
xgb_model = joblib.load("/workspaces/ML_01_video_game_prediction/video_game_analysis/notebooks/FC110557_Siyas/xgb_final_model.pkl")
encoders = joblib.load("/workspaces/ML_01_video_game_prediction/video_game_analysis/notebooks/FC110557_Siyas/encoders.pkl")
train_df = pd.read_csv("/workspaces/ML_01_video_game_prediction/video_game_analysis/notebooks/FC110557_Siyas/vgsales_cleaned.csv")

# -------------------------------
# 2. App Title
# -------------------------------
st.title("🎮 Video Game Global Sales Predictor")
st.write("Predict global sales (in millions) for a video game based on its features.")

# -------------------------------
# 3. Input Form
# -------------------------------
with st.form("prediction_form"):
    st.subheader("Categorical Features")
    col1, col2, col3 = st.columns(3)
    with col1:
        publisher = st.selectbox("Publisher", options=train_df['Publisher'].dropna().unique())
        developer = st.selectbox("Developer", options=train_df['Developer'].dropna().unique())
    with col2:
        genre = st.selectbox("Genre", options=train_df['Genre'].dropna().unique())
        platform = st.selectbox("Platform", options=train_df['Platform'].dropna().unique())
    with col3:
        rating = st.selectbox("Rating", options=train_df['Rating'].dropna().unique())
        # Year input instead of decade
        year_of_release = st.number_input("Year of Release", min_value=int(train_df['Year_of_Release'].min()), 
                                          max_value=int(train_df['Year_of_Release'].max()), 
                                          step=1, value=2010)

    # Convert year to decade
    if year_of_release < 1990:
        decade = "1980s"
    elif year_of_release < 2000:
        decade = "1990s"
    elif year_of_release < 2010:
        decade = "2000s"
    else:
        decade = "2010s"

    # Franchise Input
    franchises = ["Super Mario Bros.",
    "Call of Duty",
    "Grand Theft Auto",
    "Wii Sports",
    "Need for Speed",
    "The Legend of Zelda",
    "Medal of Honor",
    "Metal Gear",
    "The Elder Scrolls",
    "LEGO Star Wars"]
    selected_franchise = st.selectbox("Franchise", options=franchises)

    st.subheader("Numerical Features")
    col1, col2 = st.columns(2)
    with col1:
        critic_score = st.slider("Critic Score (0-100)", min_value=0, max_value=100, step=1)
        user_score = st.slider("User Score (0-10)", min_value=0.0, max_value=10.0, step=0.1)
    with col2:
        critic_count = st.number_input("Critic Count", min_value=0, step=1)
        user_count = st.number_input("User Count", min_value=0, step=1)

    # Columns for button and result
    col_button, col_result = st.columns([1, 2])
    result_placeholder = col_result.empty()

    # Large submit button
    submitted = col_button.form_submit_button("Predict Global Sales")

# -------------------------------
# 4. Prediction Logic
# -------------------------------
if submitted:
    try:
        # Encode categorical features
        publisher_enc = encoders["Publisher"].transform([publisher])[0]
        developer_enc = encoders["Developer"].transform([developer])[0]
        if selected_franchise not in encoders["Franchise"].classes_:
            selected_franchise = "Other"
        franchise_enc = encoders["Franchise"].transform([selected_franchise])[0]
    except:
        st.error("Selected category not found in encoder. Please select a valid option.")
        st.stop()

    # Prepare input DataFrame
    input_df = pd.DataFrame({
        "Publisher": [publisher_enc],
        "Developer": [developer_enc],
        "Franchise": [franchise_enc],
        "Critic_Score": [critic_score],
        "Critic_Count": [critic_count],
        "User_Score": [user_score],
        "User_Count": [user_count],
        **{f"Genre_{g}": [1 if genre == g else 0] for g in train_df['Genre'].dropna().unique()},
        **{f"Platform_{p}": [1 if platform == p else 0] for p in train_df['Platform'].dropna().unique()},
        **{f"Rating_{r}": [1 if rating == r else 0] for r in train_df['Rating'].dropna().unique()},
        **{f"Decade_{d}": [1 if decade == d else 0] for d in train_df['Decade'].dropna().unique()},
    })

    # Align with model features
    train_cols = xgb_model.get_booster().feature_names
    for col in train_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[train_cols]

    # Make prediction
    pred_log = xgb_model.predict(input_df)
    pred_sales = np.expm1(pred_log)[0]

    # Display result in the second column
    result_placeholder.success(f"Predicted Global Sales: **{pred_sales:.2f} million units**")
