import streamlit as st
import numpy as np
import re
import pdfplumber
import pytesseract

from PIL import Image

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
# TESSERACT PATH
# -----------------------------
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# -----------------------------
# TITLE
# -----------------------------
st.title("🩺 Breast Cancer Prediction System")

st.write("""
Upload a breast cancer report PDF.
The system extracts values and predicts whether the tumor is Benign or Malignant.
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

    try:

        with pdfplumber.open(uploaded_file) as pdf:

            first_page = pdf.pages[0]

            # Convert PDF page to image
            page_image = first_page.to_image(resolution=300)

            pil_image = page_image.original

            # OCR
            text = pytesseract.image_to_string(pil_image)

        # -----------------------------
        # SHOW OCR TEXT
        # -----------------------------
        st.subheader("Extracted Text")

        st.text_area("OCR Output", text, height=300)

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

            # Ignore huge values
            if value > 3000:
                continue

            values.append(value)

        # Take first 30 values
        input_data = values[:30]

        # -----------------------------
        # SHOW VALUES
        # -----------------------------
        st.subheader("Extracted Numerical Values")

        st.write(input_data)

        st.write(f"Total Values Extracted: {len(input_data)}")

    except Exception as e:

        st.error(f"Error processing PDF: {e}")

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🔍 Analyze Report"):

    if input_data is None:

        st.warning("Please upload a valid PDF report.")

    else:

        if len(input_data) != 30:

            st.error(
                "Could not extract exactly 30 numerical values."
            )

        else:

            prediction = model.predict([input_data])

            probability = model.predict_proba([input_data])

            confidence = np.max(probability) * 100

            st.subheader("Prediction Result")

            if prediction[0] == 0:

                st.error("Malignant Tumor Detected")

            else:

                st.success("Benign Tumor Detected")

            st.info(f"Confidence Score: {confidence:.2f}%")