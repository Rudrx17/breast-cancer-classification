import streamlit as st
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load dataset
data = load_breast_cancer()

X = data.data
Y = data.target

# Train model
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2
)

model = LogisticRegression(max_iter=5000)
model.fit(X_train, Y_train)

# App title
st.title("Breast Cancer Prediction App")

st.write("Enter the values below to predict whether the tumor is Benign or Malignant.")

# User input
input_data = []

for feature in data.feature_names:
    value = st.number_input(feature)
    input_data.append(value)

# Prediction button
if st.button("Predict"):
    prediction = model.predict([input_data])

    if prediction[0] == 0:
        st.error("Malignant Tumor Detected")
    else:
        st.success("Benign Tumor Detected")