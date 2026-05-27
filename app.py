import streamlit as st
import numpy as np
import re
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

st.write("""
Upload a breast cancer diagnostic report PDF.
The system automatically extracts feature values and predicts whether the tumor is Benign or Malignant.
""")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("About Project")

st.sidebar.write("""
### AI Healthcare System

This application uses Machine Learning to classify breast cancer tumors.

### Model Used
- Logistic Regression

### Features
- PDF Upload
- Automatic Value Extraction
- AI Prediction
- Confidence Score
- Professional Dashboard
""")

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
# FILE UPLOADER
# -----------------------------
uploaded_file = st.file_uploader(
    "📄 Upload PDF Report",
    type=["pdf"]
)

input_data = None

# -----------------------------
# PROCESS PDF
# -----------------------------
if uploaded_file is not None:

    st.success("PDF Uploaded Successfully ✅")

    try:

        text = ""

        with pdfplumber.open(uploaded_file) as pdf:

            for page in pdf.pages:

                extracted = page.extract_text()

                if extracted:
                    text += extracted + " "

        # -----------------------------
        # SHOW EXTRACTED TEXT
        # -----------------------------
        st.subheader("Extracted Text")

        st.text_area("PDF Output", text, height=350)

        # -----------------------------
        # CLEAN TEXT
        # -----------------------------
        text = text.replace(",", ".")

        # -----------------------------
        # EXTRACT NUMBERS
        # -----------------------------
        numbers = re.findall(r"\d+\.\d+|\d+", text)

        values = []

        for num in numbers:

            value = float(num)

            # Ignore unrealistic large values
            if value > 3000:
                continue

            values.append(value)

        # -----------------------------
        # TAKE FIRST 30 VALUES
        # -----------------------------
        input_data = values[:30]

        # -----------------------------
        # SHOW VALUES
        # -----------------------------
        st.subheader("Extracted Numerical Values")

        st.write(input_data)

        st.write(f"Total Values Extracted: {len(input_data)}")

    except Exception as e:

        st.error(f"Could not process PDF file: {e}")

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🔍 Analyze Report"):

    if input_data is None:

        st.warning("Please upload a valid PDF report.")

    else:

        try:

            if len(input_data) != 30:

                st.error(
                    "Could not extract exactly 30 numerical values from the PDF."
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

        except Exception as e:

            st.error(f"Prediction failed: {e}")