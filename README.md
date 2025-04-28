# Handwritten Digit Recognition using Machine Learning
**Project Team ID:** PTID-CDS-APR-25-2603  
**Project Code:** PRCP-1002-HandwrittenDigits

[GitHub Repository Link](#) (Update your repository link here)

---

## Executive Summary
This project focuses on automating the recognition of handwritten numerical inputs using machine learning and deep learning techniques.  
The goal is to improve operational efficiency, minimize human errors, and support scalability across industries like Finance, Healthcare, Logistics, and Education.

We leverage the MNIST benchmark dataset to demonstrate high-accuracy digit classification through models like CNN, XGBoost, Random Forest, SVM, and KNN.

---

## Business Case

Organizations heavily rely on handwritten numerical inputs (e.g., forms, checks, medical records). Manual handling leads to inefficiencies and errors.  
A machine learning solution can automate digit recognition to improve accuracy, efficiency, and scalability.

### Objective
- Build a machine learning solution to recognize handwritten digits automatically.

### Problem Addressed
- Manual transcription is error-prone, time-consuming, and non-scalable.

### Proposed Solution
- Create a machine learning model capable of classifying handwritten digits using the MNIST dataset.

### Key Benefits
- Enhanced operational efficiency
- Minimized human errors
- Scalable across industries and use-cases

### Applications
- Finance: Automated check processing
- Logistics: Parcel tracking and ZIP code reading
- Healthcare: Digitizing handwritten medical records
- Education: Automating handwritten exam scoring

---

## Domain Analysis

### Dataset Overview
- **Dataset:** MNIST
- **Total Images:** 70,000 (28x28 grayscale pixels)
- **Training Samples:** 60,000
- **Testing Samples:** 10,000
- **Classes:** 10 digits (0 to 9)
- **Balanced Dataset:** Yes
- **Preprocessing:** Normalization (0–1 range), Reshaping, One-hot encoding (for CNN)

---

## Techniques Used

- **Data Cleaning:** Normalization, Reshaping, One-Hot Encoding
- **Exploratory Data Analysis (EDA):** Visualizing sample images, pixel intensity distribution
- **Machine Learning Models:**  
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - Random Forest
  - XGBoost
- **Deep Learning Model:**  
  - Convolutional Neural Network (CNN)
- **Model Evaluation:** Accuracy, Loss, Validation Curves
- **Libraries Used:** TensorFlow, Keras, Scikit-learn, XGBoost, Matplotlib, Seaborn

---

## Data Preprocessing
- **Normalization:** Pixel values scaled from [0, 255] to [0, 1]
- **Reshaping:**  
  - Flattened for traditional ML models
  - 4D (28,28,1) reshaping for CNN
- **Label Encoding:** One-hot encoding for CNN training

---

## Model Building and Accuracy Scores

| Model                   | Accuracy  |
|--------------------------|-----------|
| Logistic Regression      | ~92%      |
| K-Nearest Neighbors (KNN) | ~95%      |
| Support Vector Machine   | ~96%      |
| Random Forest            | ~96.5%    |
| XGBoost                  | ~97%      |
| **Convolutional Neural Network (CNN)** | **~99%** |

**Insight:** CNN outperforms all traditional machine learning models due to its ability to capture spatial patterns.

---

## Key Visualizations

- 📈 Training and Validation Accuracy Curves
- 📉 Training and Validation Loss Curves
- 📊 Model Accuracy Comparison (Bar Chart)
- 🔍 Random Sample Prediction Visualization (Correct and Incorrect highlighted)

---

## Observations and Insights

- **CNN** provides the best accuracy (~99%) by learning hierarchical spatial features.
- **XGBoost** adapts very well to flattened image data and achieves high accuracy (~97%).
- **Random Forest** and **SVM** also perform competitively but require more computational time.
- **KNN** is slower for large datasets, suitable for small, simple applications.
- **Logistic Regression** serves as a good baseline but is not ideal for image data.

---

## Challenges and Solutions

| Challenge                          | Solution                                    |
|------------------------------------|---------------------------------------------|
| Raw pixel intensity inconsistencies | Normalization [0–1] scaling                |
| Loss of spatial features in flattened images | Applied CNN architecture               |
| Model overfitting                   | Dropout layers, Regularization             |
| Computational cost for deep learning | Used batch processing, optimized learning rates |
| Difficult model comparison          | Created a comparison framework using common metrics |

---

## Technologies Used

- **Programming Language:** Python
- **Libraries:** TensorFlow, Keras, Scikit-learn, XGBoost, Seaborn, Matplotlib
- **Tools:** Jupyter Notebook, Google Colab
- **ML Techniques:** Supervised Learning, Classification, Hyperparameter Tuning

---

## Future Work

- Deploy the CNN model using Flask or FastAPI for real-time digit recognition.
- Create a mobile app that digitizes handwritten notes instantly.
- Experiment with more complex datasets (e.g., EMNIST, real-world handwriting datasets).
- Integrate Transfer Learning to boost model robustness.

---

## Project Structure

