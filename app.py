import os
import pickle
import numpy as np
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

ROOT = os.path.dirname(__file__)
MODEL_PATH = os.path.join(ROOT, "model.pkl")
CSV_PATH = os.path.join(ROOT, "data", "iris.csv")


def train_and_save_model(csv_path: str, path: str):
    df = pd.read_csv(csv_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_enc)
    # classes in human-friendly form (strip 'Iris-' prefix if present)
    classes = [c.replace("Iris-", "") for c in le.classes_]
    with open(path, "wb") as f:
        pickle.dump({"model": model, "classes": classes, "label_encoder": le}, f)
    return model, classes, le


def load_model_or_train(csv_path: str, path: str):
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data["model"], data["classes"], data.get("label_encoder")
    else:
        return train_and_save_model(csv_path, path)


def main():
    st.title("🌸 Iris Flower Prediction")
    st.write("Enter flower measurements:")

    model, classes, le = load_model_or_train(CSV_PATH, MODEL_PATH)

    sepal_length = st.number_input("Sepal Length", 0.0, 10.0)
    sepal_width = st.number_input("Sepal Width", 0.0, 10.0)
    petal_length = st.number_input("Petal Length", 0.0, 10.0)
    petal_width = st.number_input("Petal Width", 0.0, 10.0)

    if st.button("Predict"):
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        pred = model.predict(features)[0]
        # if model was trained with LabelEncoder, pred is an int index
        try:
            pred_label = classes[int(pred)]
        except Exception:
            # fallback if model predicts string labels
            pred_label = str(pred)
        st.success(f"Prediction: **{pred_label}**")


if __name__ == "__main__":
    main()
