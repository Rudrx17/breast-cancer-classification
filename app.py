import streamlit as st
import numpy as np
import pandas as pd
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

.stApp {
    background-color: #0E1117;
    color: white;
}

h1, h2, h3 {
    color: white;
}

.stButton > button {
    width: 100%;
    height: 3em;
    border-radius: 12px;
    font-size: 18px;
    font-weight: bold;
    background-color: #ff4b4b;
    color: white;
    border: none;
}

.stButton > button:hover {
    background-color: #ff2e2e;
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# TITLE
# -----------------------------
st.title("🩺 Breast Cancer Prediction System")

st.write(
    "Upload a medical report file containing 30 numerical values "
    "to predict whether the tumor is Benign or Malignant."
)

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
    X,
    Y,
    test_size=0.2,
    random_state=2
)

model = LogisticRegression(max_iter=5000)

model.fit(X_train, Y_train)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("About Project")

st.sidebar.write("""
This AI-powered system predicts breast cancer using Machine Learning.

### Model Used
- Logistic Regression

### Dataset
- Breast Cancer Wisconsin Dataset

### Features
- File Upload
- Real-Time Prediction
- Confidence Score
- Interactive UI
""")

# -----------------------------
# FILE UPLOADER
# -----------------------------
uploaded_file = st.file_uploader(
    "📂 Upload Report File",
    type=["txt", "csv"]
)

input_data = None

# -----------------------------
# PROCESS TXT FILE
# -----------------------------
if uploaded_file is not None:

    st.success("File Uploaded Successfully ✅")

    try:

        # TXT FILE
        if uploaded_file.type == "text/plain":

            content = uploaded_file.read().decode("utf-8")

            values = content.split(",")

            input_data = [float(x.strip()) for x in values]

        # CSV FILE
        elif uploaded_file.type == "text/csv":

            df = pd.read_csv(uploaded_file)

            input_data = df.values.flatten().tolist()

        st.subheader("Extracted Values")

        st.write(input_data)

    except:

        st.error("Could not read uploaded file.")

# -----------------------------
# PREDICTION BUTTON
# -----------------------------
if st.button("🔍 Analyze Report"):

    if input_data is None:

        st.warning("Please upload a valid report file.")

    else:

        try:

            if len(input_data) != 30:

                st.error(
                    "The uploaded report must contain exactly 30 numerical values."
                )

            else:

                prediction = model.predict([input_data])

                probability = model.predict_proba([input_data])

                confidence = np.max(probability) * 100

                st.subheader("Prediction Result")

                # MALIGNANT
                if prediction[0] == 0:

                    st.markdown(f"""
                    <div class="result-box" style="background-color:#ff4b4b;">
                        Malignant Tumor Detected
                    </div>
                    """, unsafe_allow_html=True)

                # BENIGN
                else:

                    st.markdown(f"""
                    <div class="result-box" style="background-color:#00c853;">
                        Benign Tumor Detected
                    </div>
                    """, unsafe_allow_html=True)

                st.info(f"Confidence Score: {confidence:.2f}%")

        except:

            st.error("Prediction failed. Please check your file format.")