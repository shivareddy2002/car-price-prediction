<!-- Header Banner -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:ff4b4b,100:f9c74f&height=180&section=header&text=%20AI%20Powered%20Car%20Price%20Prediction&fontSize=36&fontColor=ffffff&animation=fadeIn&fontAlignY=35"/>
</p>
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
## 🔍 Features

### ✅ Core Features
- **Dual-Model Logic:** Real-time comparison between Machine Learning and Deep Learning predictions.
- **INR Formatted Output:** Results displayed in Indian Rupee (₹) format for local market relevance.
- **Professional UI:** Clean, modern, and responsive user interface built with Streamlit.

### 📊 Market Analysis & XAI
- **Feature Importance:** Visualizes the impact of key features such as Manufacturing Year, Max Power, and Kilometers Driven on car price.
- **Depreciation Trends:** Insights into how vehicle value decreases over time.
- **Market Demand Indicators:** Popularity analysis of fuel types and transmission modes.

---

## 🧪 Input Parameters

| Feature | Description |
|-------|------------|
| Brand | Car manufacturer (Maruti, BMW, Hyundai, etc.) |
| Year | Manufacturing year |
| Fuel Type | Petrol / Diesel / CNG / LPG |
| Transmission | Manual / Automatic |
| KM Driven | Total distance covered by the vehicle |
| Engine | Engine capacity in CC |
| Max Power | Power output in BHP |
| Owner | Number of previous owners |
| Seller Type | Individual or Professional Dealer |

---

## 🔮 Future Enhancements

- 📍 **Location Intelligence:** City/RTO-based price adjustments.
- 📸 **Computer Vision:** Image-based vehicle damage detection for automatic price correction.
- 📉 **Live Market Data:** Integration with real-time used car listings using APIs.
- 🏢 **B2B Dashboard:** Bulk valuation and analytics tools for car dealerships.
## 👨‍💻 Author  

**Lomada Siva Gangi Reddy**  
🎓 B.Tech CSE (Data Science), RGMCET (2021–2025)  
🎯 Aspiring Data Analyst | Skilled in Python, SQL, Power BI, and Data Science  
📍 Open to **Internships & Job Offers**

📬 **Contact Me**  
- 📞 9346493592  
- [💼 LinkedIn](https://www.linkedin.com/in/lomada-siva-gangi-reddy-a64197280/)  [🌐 GitHub](https://github.com/shivareddy2002)  

---
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:f9c74f,100:ff4b4b&height=120&section=footer"/>
</p>
