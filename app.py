import streamlit as st
import numpy as np
import pandas as pd
import pdfplumber
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Breast Cancer Prediction System",
    page_icon="🩺",
    layout="centered"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
    color: white;
}

.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    font-size: 18px;
    font-weight: bold;
}

.result-box {
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# TITLE
# -----------------------------
st.title("🩺 Breast Cancer Prediction System")
st.write("Upload a report or paste 30 comma-separated values to predict whether the tumor is Benign or Malignant.")

# -----------------------------
# LOAD DATASET
# -----------------------------
data = load_breast_cancer()

X = data.data
Y = data.target

# -----------------------------
# TRAIN MODEL
# -----------------------------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2
)

model = LogisticRegression(max_iter=5000)
model.fit(X_train, Y_train)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("About")
st.sidebar.write("""
This AI-powered system predicts breast cancer using Machine Learning.

Model Used:
- Logistic Regression

Dataset:
- Breast Cancer Wisconsin Dataset
""")

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "📂 Upload Report",
    type=["txt", "csv", "pdf"]
)

input_text = ""

# -----------------------------
# READ TXT FILE
# -----------------------------
if uploaded_file is not None:

    st.success("File uploaded successfully!")

    if uploaded_file.type == "text/plain":
        input_text = uploaded_file.read().decode("utf-8")

    elif uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
        input_text = ",".join(map(str, df.values.flatten()))

    elif uploaded_file.type == "application/pdf":

        text = ""

        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()

                if extracted:
                    text += extracted

        st.subheader("Extracted Report Text")
        st.text_area("PDF Content", text, height=200)

# -----------------------------
# MANUAL INPUT
# -----------------------------
manual_input = st.text_area(
    "✍️ Paste 30 Comma-Separated Values",
    value=input_text,
    height=150
)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🔍 Predict"):

    try:

        input_data = [float(x.strip()) for x in manual_input.split(",")]

        if len(input_data) != 30:
            st.error("Please enter exactly 30 numerical values.")

        else:

            prediction = model.predict([input_data])

            probability = model.predict_proba([input_data])

            confidence = np.max(probability) * 100

            st.subheader("Prediction Result")

            if prediction[0] == 0:

                st.markdown(f"""
                <div class="result-box" style="background-color:#ff4b4b;">
                    Malignant Tumor Detected
                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown(f"""
                <div class="result-box" style="background-color:#00c853;">
                    Benign Tumor Detected
                </div>
                """, unsafe_allow_html=True)

            st.info(f"Confidence Score: {confidence:.2f}%")

    except:
        st.error("Invalid input format. Please check your values.")