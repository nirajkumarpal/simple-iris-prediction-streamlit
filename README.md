# 🌸 Simple Iris Prediction — Streamlit App

A machine learning–based Streamlit web application that predicts the species of an Iris flower using four input features.  
The project uses a Random Forest classifier trained on the classic Iris dataset.

## 🚀 Live Demo
👉 https://simple-iris-prediction-app.streamlit.app/

## 🧠 Model Details
- Algorithm: Random Forest Classifier
- Dataset: Iris Dataset
- Input Features:
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width
- Output Classes:
  - Setosa
  - Versicolor
  - Virginica

## 🛠 Tech Stack
- Python
- Scikit-learn
- Streamlit
- NumPy
- Pandas
- Matplotlib
- Joblib

## ▶️ Run Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py