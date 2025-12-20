# 🚗 AutoPredict Pro – AI-Powered Car Price Prediction

AutoPredict Pro is an end-to-end **Machine Learning & Deep Learning web application** that predicts the **market value of used cars** based on technical specifications, usage details, and seller information. The system provides accurate, data-driven price estimates through a **professional Streamlit interface**.

🔗 **Live Demo**: [AutoPredict Pro Web App](https://car-price-prediction-pro.streamlit.app/)

---

## 📌 Problem Statement

Pricing used cars manually is difficult due to multiple influencing factors such as:
- Manufacturing year and Depreciation.
- Mileage and engine capacity.
- Fuel type and transmission modes.
- Ownership history and brand equity.

This often results in inconsistent pricing and poor decision-making for both buyers and sellers. **AutoPredict Pro** solves this by using AI models to deliver **fair and reliable car price predictions**.

---

## 🏗️ System Architecture

The application follows a structured pipeline to ensure data integrity and prediction accuracy:



1. **User Input:** Features collected via the Streamlit UI.
2. **Data Validation:** Checking for logical consistency in inputs.
3. **Preprocessing:** Categorical encoding (One-Hot) and Feature Scaling via the saved pipeline.
4. **Model Inference:** Data is passed to either the **Random Forest** or **ANN** model.
5. **Output:** The predicted value is inverse-transformed and displayed in INR.

---

## 🧠 Models Used

### 1️⃣ Random Forest Regressor (Machine Learning)
- Handles non-linear relationships effectively by aggregating multiple decision trees.
- Highly robust to outliers and prevents overfitting.
- **Best suited for:** General market trends and older/mid-range vehicles.

### 2️⃣ Artificial Neural Network (Deep Learning)
- A multi-layer dense neural network built with TensorFlow/Keras.
- Learns complex hidden patterns within the high-dimensional feature space.
- **Best suited for:** Capturing subtle price variations in premium and modern vehicles.

---

## 🛠️ Tech Stack

### Frontend & Deployment
- **Streamlit:** For the interactive web dashboard.
- **HTML/CSS:** Custom styling for the "Glass-morphism" UI effect.
- **Lottie:** Smooth animations for enhanced UX.

### Backend & Machine Learning
- **Python:** Core programming language.
- **Scikit-learn:** For Random Forest and preprocessing.
- **TensorFlow / Keras:** For the Deep Learning ANN architecture.
- **Pandas & NumPy:** For data manipulation and numerical computation.

### Model Persistence
- **Joblib / Pickle:** For serializing the trained models and scalers.

---

## 📂 Project Structure

```text
car-price-prediction/
├── app.py                     # Main Streamlit application script
├── rf_model.pkl               # Trained Random Forest model (Scikit-learn)
├── dl_model.h5                # Trained Deep Learning model (Keras/TF)
├── preprocessor.pkl           # Feature preprocessing pipeline
├── y_scaler.pkl               # Target scaler for DL model
├── options.pkl                # Dropdown feature options for UI
├── requirements.txt           # Python project dependencies
├── assets/                    # Image assets (car.png, logo.png)
└── README.md                  # Project documentation
