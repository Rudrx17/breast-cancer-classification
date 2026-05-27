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

# Title
st.title("Breast Cancer Prediction System")

st.write("Paste 30 comma-separated values for prediction.")

# Single text input
input_text = st.text_area("Enter Input Values")

if st.button("Predict"):

    try:
        # Convert input into float list
        input_data = [float(x) for x in input_text.split(",")]

        # Ensure 30 values
        if len(input_data) != 30:
            st.error("Please enter exactly 30 values.")
        else:
            prediction = model.predict([input_data])

            if prediction[0] == 0:
                st.error("Malignant Tumor Detected")
            else:
                st.success("Benign Tumor Detected")

    except:
        st.error("Invalid input format.")