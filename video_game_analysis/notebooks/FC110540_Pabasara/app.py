import streamlit as st
import pandas as pd
import joblib

model = joblib.load("train_model.pkl")
encoders = joblib.load("label_encoders.pkl")

st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>
        🎮 Video Game Global Sales Predictor
    </h1>
    <p style='text-align: center; font-size:18px; color: gray;'>
        Predict global video game sales using machine learning
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

st.subheader("Enter Game Details")

col1, col2 = st.columns(2)


with st.form(key="game_form"):
    with col1:
        platform = st.selectbox("Platform", encoders["Platform"].classes_)
        genre = st.selectbox("Genre", encoders["Genre"].classes_)
        publisher = st.text_input("Publisher", "Nintendo")
        developer = st.text_input("Developer", "Ubisoft")
        rating = st.selectbox("Rating", encoders["Rating"].classes_)

    with col2:
        year = st.number_input("Year of Release", 1980, 2025, 2015)
        critic_score = st.number_input("Critic Score (0-100)", 0, 100, 75)
        critic_count = st.number_input("Critic Count", 0, 500, 50)
        user_score = st.number_input("User Score (0-10)", 0.0, 10.0, 7.5)
        user_count = st.number_input("User Count", 0, 1000000, 500)
    
    submit = st.form_submit_button(label="Predict global sales.")

# -------------------- HELPER FUNCTION --------------------
def encode_value(encoder, value):
    """Encode categorical values safely; return -1 if unseen."""
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        return -1

st.markdown("<br>", unsafe_allow_html=True)  # spacing

if submit:
    # Encode categorical features
    encoded_platform = encode_value(encoders["Platform"], platform)
    encoded_genre = encode_value(encoders["Genre"], genre)
    encoded_publisher = encode_value(encoders["Publisher"], publisher)
    encoded_developer = encode_value(encoders["Developer"], developer)
    encoded_rating = encode_value(encoders["Rating"], rating)

    # Create input DataFrame (match training columns)
    input_data = pd.DataFrame({
        "Platform": [encoded_platform],
        "Year_of_Release": [year],
        "Genre": [encoded_genre],
        "Publisher": [encoded_publisher],
        "Critic_Score": [critic_score],
        "Critic_Count": [critic_count],
        "User_Score": [user_score],
        "User_Count": [user_count],
        "Developer": [encoded_developer],
        "Rating": [encoded_rating],
    })

    # Add engineered features
    input_data["Years_Since_Release"] = 2025 - input_data["Year_of_Release"]
    input_data["Critic_User_Score_Ratio"] = input_data["Critic_Score"] / (input_data["User_Score"] + 1e-5)
    input_data["Total_Review_Count"] = input_data["Critic_Count"] + input_data["User_Count"]

    # # Optional: Ensure correct feature order
    # feature_order = [
    #     "Platform", "Year_of_Release", "Genre", "Publisher", "Critic_Score",
    #     "Critic_Count", "User_Score", "User_Count", "Developer", "Rating",
    #     "Years_Since_Release", "Critic_User_Score_Ratio", "Total_Review_Count"
    # ]
    # input_data = input_data[feature_order]

    # Make prediction
    prediction = model.predict(input_data)[0]

    st.markdown(
        f"""
        <div style='text-align: center; margin-top: 40px;'>
            <h2 style='color: #2196F3;'>💰 Predicted Global Sales</h2>
            <h1 style='color: #FF5722;'>{prediction:.2f} million units</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
