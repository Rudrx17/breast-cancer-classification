# Breast Cancer Classification using Machine Learning

## 📌 Overview

This project aims to classify breast cancer tumors as **benign** or **malignant** using machine learning techniques.  
The dataset used is the [Breast Cancer Wisconsin (Diagnostic) Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html), which contains features computed from digitized images of fine needle aspirate (FNA) of breast masses.

## 🧠 Approach

1. **Data Loading & Exploration**

   - Loaded dataset from `sklearn.datasets`.
   - Explored feature distributions and correlations.

2. **Data Preprocessing**

   - Feature scaling with StandardScaler.
   - Train-test split (80/20 ratio).

3. **Model Training**

   - Models tried: Logistic Regression, Decision Tree, Random Forest.
   - Evaluated using accuracy, confusion matrix, and ROC curves.

4. **Evaluation**
   - Selected the best model based on accuracy and ROC-AUC score.

## 📊 Results

- **Best Model:** Logistic Regression (example — replace with actual result)
- **Accuracy:** 97% (example)
- **ROC-AUC:** 0.99 (example)

## 🛠️ Installation & Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/Rudrx17/breast-cancer-classification.git
   cd breast-cancer-classification
   ```
