import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# ---------------- LOAD FILES ---------------- #

model = pickle.load(open("model.pkl", "rb"))
team_encoder = pickle.load(open("team_encoder.pkl", "rb"))
venue_encoder = pickle.load(open("venue_encoder.pkl", "rb"))

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(page_title="IPL Predictor", page_icon="🏏", layout="centered")

# ---------------- CUSTOM STYLE ---------------- #

st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        font-size: 18px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- TITLE ---------------- #

st.title("🏏 IPL Match Winner Predictor")
st.markdown("### Predict match winning probability using AI 🤖")

# ---------------- INPUT ---------------- #

team_list = sorted(list(team_encoder.classes_))
venue_list = sorted(list(venue_encoder.classes_))

col1, col2 = st.columns(2)

with col1:
    team1 = st.selectbox("🏏 Team 1", team_list)
    team2 = st.selectbox("🏏 Team 2", team_list)

with col2:
    toss = st.selectbox("🪙 Toss Winner", team_list)
    venue = st.selectbox("📍 Venue", venue_list)

# ---------------- PREDICTION ---------------- #

if st.button("🚀 Predict"):

    if team1 == team2:
        st.error("⚠️ Both teams cannot be same!")
        st.stop()

    input_data = [[
        team_encoder.transform([team1])[0],
        team_encoder.transform([team2])[0],
        team_encoder.transform([toss])[0],
        venue_encoder.transform([venue])[0]
    ]]

    probs = model.predict_proba(input_data)
    classes = model.classes_

    result = {}

    for i, prob in enumerate(probs[0]):
        team_name = team_encoder.inverse_transform([classes[i]])[0]

        if team_name == team1 or team_name == team2:
            result[team_name] = round(prob * 100, 2)

    # ---------------- DISPLAY ---------------- #

    st.subheader("🏆 Winning Probability")

    sorted_result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

    for team, prob in sorted_result.items():
        st.markdown(f"### {team} → **{prob}%**")
        st.progress(int(prob))

    winner = max(result, key=result.get)
    st.success(f"🏆 Predicted Winner: {winner}")

    # ---------------- CHART ---------------- #

    st.subheader("📊 Probability Comparison")

    teams = list(result.keys())
    values = list(result.values())

    fig, ax = plt.subplots()
    ax.bar(teams, values)
    ax.set_ylabel("Winning %")
    ax.set_title("Match Prediction")

    st.pyplot(fig)