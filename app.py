import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Label mapping
label_map = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}

# -----------------------------
# UI Title
# -----------------------------
st.title("🌸 Iris Flower Prediction App")

st.write("Enter the features below:")

# -----------------------------
# Input Fields (Form UI)
# -----------------------------
sl = st.number_input("Sepal Length")
sw = st.number_input("Sepal Width")
pl = st.number_input("Petal Length")
pw = st.number_input("Petal Width")

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict"):
    try:
        features = np.array([[sl, sw, pl, pw]])
        pred = model.predict(features)[0]
        label = label_map[int(pred)]

        st.success(f"Predicted Flower: {label}")

    except Exception as e:
        st.error(f"Error: {e}")

# -----------------------------
# TEST DATA SECTION
# -----------------------------
st.subheader("Test Dataset Predictions")

if st.button("Show Test Predictions"):
    try:
        data = pd.read_csv("X_test.csv")
        preds = model.predict(data)
        labels = [label_map[int(p)] for p in preds]

        data["Predicted_Label"] = labels
        st.dataframe(data)

    except Exception as e:
        st.error(f"Error: {e}")
