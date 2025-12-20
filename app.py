import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import time
import requests
import os
from streamlit_lottie import st_lottie
from tensorflow.keras.models import load_model
import tensorflow as tf

# ==========================================
# 1. PAGE CONFIGURATION & THEME
# ==========================================
st.set_page_config(
    page_title="AutoPredict Pro | AI Car Valuation",
    # page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Professional Look
st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
    .main-header { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #1E3A8A; font-size: 42px; font-weight: 800; text-align: center; margin-bottom: 0px; }
    .footer { position: fixed; left: 0; bottom: 0; width: 100%; background-color: #1E3A8A; color: white; text-align: center; padding: 10px; font-size: 14px; z-index: 100; }
    .stat-card { background-color: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; height: 100%; }
    .result-card { background-color: #ffffff; padding: 30px; border-radius: 20px; border-left: 10px solid #10B981; text-align: center; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); }
    .info-box { background-color: #EFF6FF; border-left: 5px solid #3B82F6; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. ASSET LOADING
# ==========================================
@st.cache_resource
def load_assets():
    required_files = ['preprocessor.pkl', 'y_scaler.pkl', 'rf_model.pkl', 'dl_model.h5', 'options.pkl']
    for f in required_files:
        if not os.path.exists(f): return None, None, None, None, None, None
    try:
        prep = joblib.load('preprocessor.pkl')
        y_sc = joblib.load('y_scaler.pkl')
        rf = joblib.load('rf_model.pkl')
        # Using custom_objects to prevent the 'mse' deserialization error
        dl = load_model('dl_model.h5', custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
        with open('options.pkl', 'rb') as f: opt = pickle.load(f)
        try:
            r = requests.get("https://assets10.lottiefiles.com/packages/lf20_V99dn2.json")
            lottie = r.json() if r.status_code == 200 else None
        except: lottie = None
        return prep, y_sc, rf, dl, opt, lottie
    except Exception as e:
        st.error(f"❌ Asset Error: {e}")
        return None, None, None, None, None, None

preprocessor, y_scaler, rf_model, dl_model, options, car_anim = load_assets()

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.image("car_logo.png", width=1000)
    st.title("AutoPredict Pro")
    page = st.radio("Navigation Menu", ["🏠 Home", "📈 Market Analysis", "🏢 For Dealerships", "ℹ️ About AI"])
    st.markdown("---")
    # # st.write("**Model:** v2.4 (Stable)")
    # st.write("**Region:** India (Market-Adjusted)")
    # st.write("**Last Update:** Dec 20, 2025")
    st.markdown("""
        **Developed By SIVA REDDY** 
    """)
    st.markdown("""
        [LinkedIn](https://www.linkedin.com/in/lomada-siva-gangi-reddy-a64197280/) | 
        [GitHub](https://github.com/shivareddy2002) | 
        [Portfolio](https://lsgr-portfolio-pulse.vercel.app/)
    """)
# ==========================================
# 4. PAGE LOGIC
# ==========================================

# --- PAGE 1: HOME ---
if page == "🏠 Home":
    st.markdown('<div class="main-header">Car Price Prediction (AutoPredict Pro) </div>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #4B5563; font-size: 18px;'>High-Precision AI Valuation Engine</p>", unsafe_allow_html=True)
    
    if options is None:
        st.warning("⚠️ Files missing. Please ensure model_train.py has been run successfully in this folder.")
        st.stop()

    col_left, col_right = st.columns([1, 1.2])
    with col_left:
        if car_anim: st_lottie(car_anim, height=300)
        model_choice = st.selectbox("Intelligence Engine", ["Random Forest (ML Optimized)", "Neural Network (Deep Learning)"])
        st.image("car.png", width=1000)
        st.info("💡 **Pro Tip:** Neural Networks are highly sensitive to 'Max Power' and 'Year'. Use them for modern or luxury cars.")

    with col_right:
                # SECTION 1: VEHICLE IDENTITY
        st.markdown("## Vehicle Identity ")
        c1, c2 = st.columns(2)
        with c1:
            brand = st.selectbox("Brand Name", options['name'])
            year = st.select_slider("Manufacturing Year", options=list(range(2000, 2026)), value=2018)
            fuel = st.selectbox("Fuel Type", options['fuel'])
            trans = st.selectbox("Transmission", options['transmission'])

        with c2:
            km = st.number_input("Kilometers Driven", 0, 1_000_000, 45_000, step=1000)
            seller = st.selectbox("Seller Category", options['seller_type'])
            owner = st.selectbox("Previous Owners", options['owner'])
            seats = st.selectbox("Seating Capacity", [4, 5, 7], index=1)

        #st.divider()
        st.markdown("## Technical Specifications")

        c3, c4 = st.columns(2)
        with c3:
            engine = st.slider("Engine Capacity (CC)", 600, 5000, 1200, step=100)
            power = st.number_input("Max Power (BHP)", 30.0, 500.0, 85.0)

        with c4:
            mileage = st.number_input("Mileage (kmpl)", 5.0, 40.0, 18.5)
            model_choice = st.radio(
                "🧠 Intelligence Engine",
                ["Random Forest (ML Optimized)", "Neural Network (Deep Learning)"],
                horizontal=True
            )

    if st.button("CALCULATE ESTIMATED MARKET VALUE", use_container_width=True, type="primary"):
        
        input_df = pd.DataFrame({
            'year': [year], 'km_driven': [km], 'seats': [seats], 'max_power': [power],
            'Mileage': [mileage], 'Engine': [engine], 'name': [brand], 'fuel': [fuel],
            'seller_type': [seller], 'transmission': [trans], 'owner': [owner]
        })
        try:
            processed = preprocessor.transform(input_df)
            if "Random" in model_choice:
                price = rf_model.predict(processed)[0]
            else:
                price = y_scaler.inverse_transform(dl_model.predict(processed))[0][0]
            
        #     st.markdown(f'<div class="result-card"><h3 style="color:#6B7280">Estimated Valuation</h3><h1 style="color:#065F46; font-size:55px">₹ {price:,.2f}</h1></div>', unsafe_allow_html=True)
        #     # st.balloons()
        # except Exception as e:
        #     st.error(f"Valuation failed: {e}")
            st.markdown(f"""
                <div class="result-card">
                        <h3 style="color:#6B7280">Estimated Valuation</h3>
                        <h1 style="color:#065F46; font-size:55px">₹ {price:,.2f}</h1>
                        <p style="color:#6B7280">Note: Final price may vary based on physical inspection.</p>
                </div>
                """, unsafe_allow_html=True)
            #st.balloons()
            st.toast("Prediction completed successfully ✅")

        except Exception as e:
            st.error(f"Valuation failed: {e}")

# --- PAGE 2: MARKET ANALYSIS ---
elif page == "📈 Market Analysis":
    st.title("Market Analysis & Insights 📈")
    st.markdown("---")
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Market Demand", "High", "+4.2%")
    k2.metric("Avg. Resale (5yr)", "₹ 4.8L", "-1.1%")
    k3.metric("Popular Fuel", "Petrol", "Steady")
    k4.metric("Top Brand", "Maruti", "60% Share")

    st.subheader("Depreciation & Demand Visuals")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Value Depreciation vs Age (Estimated)**")
        chart_data = pd.DataFrame({'Years': range(1, 11), 'Value (%)': [100, 85, 72, 63, 55, 48, 42, 37, 33, 30]})
        st.line_chart(chart_data.set_index('Years'))
    with c2:
        st.write("**Market Interest by Fuel Type**")
        fuel_data = pd.DataFrame({'Fuel': ['Petrol', 'Diesel', 'CNG', 'LPG'], 'Popularity': [55, 30, 12, 3]})
        st.bar_chart(fuel_data.set_index('Fuel'))

    st.markdown('<div class="info-box"><h4>Market Alerts</h4><ul><li><b>CNG Surge:</b> Demand for factory-fitted CNG cars is up 22%.</li><li><b>Automatic Preference:</b> Urban buyers now prioritize Automatic over Manual by 2:1.</li><li><b>Color Impact:</b> White and Silver cars hold 5% higher resale value.</li></ul></div>', unsafe_allow_html=True)

# --- PAGE 3: FOR DEALERSHIPS ---
elif page == "🏢 For Dealerships":
    st.title("Enterprise Solutions 🏢")
    st.subheader("Bulk Valuation & Inventory Management")
    st.markdown("This section is designed for volume sellers to optimize their Trade-In programs.")
    
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.write("### 📤 Bulk Valuation")
            st.write("Upload your inventory CSV to get valuations for up to 500 cars at once.")
            st.file_uploader("Upload CSV File", type=["csv"])
            st.button("Process Bulk Inventory", type="primary")

    with col2:
        with st.container(border=True):
            st.write("### 🔒 API Access")
            st.write("Integrate AutoPredict Pro logic directly into your dealership's website.")
            st.code("POST /api/v2/valuation\nAuth: Bearer YOUR_TOKEN", language="bash")
            st.button("Request API Key")

    st.divider()
    st.write("### Partnership Features")
    p1, p2, p3 = st.columns(3)
    p1.write("✅ **Priority Processing**")
    p2.write("✅ **Market Heatmaps**")
    p3.write("✅ **White Label Reports**")

# --- PAGE 4: ABOUT AI ---
elif page == "ℹ️ About AI":
    st.title("How the AI Works ℹ️")
    st.markdown("AutoPredict Pro utilizes a **Hybrid Model Selection** system to ensure valuation accuracy.")

    st.subheader("1. Random Forest Regressor")
    st.markdown("This model creates an 'ensemble' of 100 Decision Trees. It is great for handling non-linear relationships in car data.")
    st.info("🎯 **Best for:** Consistent market behavior and older car models.")

    st.subheader("2. Artificial Neural Network (ANN)")
    st.markdown("Our ANN uses 3 layers of artificial neurons to identify hidden patterns, such as the luxury status of a brand.")
    st.info("🚀 **Best for:** Identifying outliers and high-value luxury car trends.")

    st.subheader("3. Data Preprocessing")
    st.write("**The AI converts categorical data into numerical formats:**")
    st.markdown("""
    - **One-Hot Encoding:** Converts text (like Brand) into binary columns.
    - **Standard Scaling:** Normalizes features like KM and Year so they have equal weight.
    """)
    # Section 4: Feature Engineering
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("4. Feature Engineering & Weighting")
    st.write("Not all features are equal. Our AI weights features based on market influence:")
    st.progress(0.85, text="**Manufacturing Year (Age)** - 40% Impact")
    st.progress(0.70, text="**Max Power (Performance)** - 25% Impact")
    st.progress(0.40, text="**Kilometers Driven** - 20% Impact")
    st.progress(0.20, text="**Brand & Fuel Type** - 15% Impact")
    st.markdown('</div>', unsafe_allow_html=True)
    st.subheader("AI Architecture ℹ️")

    st.info("AutoPredict Pro uses a multi-stage pipeline: Data cleaning -> Feature Encoding -> Ensemble Regression.")

# ==========================================
# 5. FOOTER
# ==========================================
st.markdown("<br><br><br><br>", unsafe_allow_html=True)
st.markdown("""
    <div class="footer">
        Powered by TensorFlow & Scikit-Learn | © 2025 AutoPredict Pro - By Shiva
    </div>
    """, unsafe_allow_html=True)