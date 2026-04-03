# IPL-Match-Prediction
Machine Learning based web app to predict IPL match winning probability using team data, toss result, and venue.
# 🏏 IPL Match Winner Predictor

This project is a Machine Learning based web application that predicts the winning probability of IPL teams based on past match data.

## 🚀 Features
- Predict winning probability between two teams
- Considers:
  - Teams
  - Toss winner
  - Venue
- Interactive UI using Streamlit
- Visual representation using charts

## 🧠 Tech Stack
- Python
- Pandas
- Scikit-learn
- Streamlit
- Matplotlib

## 📊 How it Works
The model is trained on historical IPL match data. It uses:
- Team encoding
- Venue encoding
- Random Forest Classifier

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
