import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ── 1. Page Config ────────────────────────────────────────────────
st.set_page_config(page_title="AutoValuate Pro", page_icon="🚘", layout="wide")

# Custom CSS for the "Commercial" look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .prediction-card {
        padding: 40px; border-radius: 15px; background: #1E2A5E;
        color: white; text-align: center; margin: 20px 0;
    }
    .impact-box {
        padding: 20px; border-radius: 10px; background: #ffffff;
        border-left: 5px solid #1E2A5E; box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# ── 2. Data & Model Loading ───────────────────────────────────────
@st.cache_data
def load_car_data():
    df = pd.read_csv("car.csv")
    # Purge bikes/scooters for a focused Car App
    exclude = ['royal enfield', 'ktm', 'bajaj', 'honda cb', 'yamaha', 'tvs', 'hero', 'activa', 'pulsar', 'shine']
    df = df[~df['Car_Name'].str.lower().str.contains('|'.join(exclude))]
    
    brand_map = {
        'ritz': 'Maruti Suzuki', 'sx4': 'Maruti Suzuki', 'ciaz': 'Maruti Suzuki', 'wagon r': 'Maruti Suzuki',
        'swift': 'Maruti Suzuki', 'fortuner': 'Toyota', 'innova': 'Toyota', 'corolla altis': 'Toyota',
        'i20': 'Hyundai', 'i10': 'Hyundai', 'verna': 'Hyundai', 'city': 'Honda', 'amaze': 'Honda'
    }
    df['Brand'] = df['Car_Name'].apply(lambda x: brand_map.get(x.lower(), "Other"))
    return df

try:
    df = load_car_data()
    model = joblib.load("car_price_model.pkl")
except:
    st.error("Ensure 'car.csv' and 'car_price_model.pkl' are in the directory.")
    st.stop()

# ── 3. UI Layout ──────────────────────────────────────────────────
st.title("🚘 AutoValuate Pro: AI Valuation Engine")
st.write("Professional ML system for car selling price prediction and market analysis.")

col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    st.subheader("📍 Vehicle Specifications")
    brand = st.selectbox("Brand", sorted(df['Brand'].unique()))
    models = sorted(df[df['Brand'] == brand]['Car_Name'].unique())
    car_model = st.selectbox("Model", models)
    
    c1, c2 = st.columns(2)
    year = c1.number_input("Purchase Year", 2000, 2024, 2018)
    p_price = c2.number_input("Showroom Price (₹L)", 0.5, 100.0, 8.0)
    
    c3, c4 = st.columns(2)
    kms = c3.number_input("KM Driven", 0, 500000, 30000)
    owner = c4.selectbox("Previous Owners", [0, 1, 3])
    
    fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    trans = st.radio("Transmission", ["Manual", "Automatic"], horizontal=True)
    seller = st.radio("Seller Type", ["Dealer", "Individual"], horizontal=True)

# ── 4. Prediction Logic & Result ──────────────────────────────────
with col_right:
    st.subheader("📊 Prediction & Analysis")
    
    # Prep features
    age = 2026 - year # Based on current year 2026
    f_val = {"Petrol": 0, "Diesel": 1, "CNG": 2}[fuel]
    s_val = {"Dealer": 0, "Individual": 1}[seller]
    t_val = {"Manual": 0, "Automatic": 1}[trans]
    
    input_data = [[p_price, age, kms, f_val, s_val, t_val, owner]]
    
    if st.button("RUN VALUATION ANALYSIS", use_container_width=True):
        prediction = model.predict(input_data)[0]
        final_price = max(0.1, prediction)

        st.markdown(f"""
            <div class="prediction-card">
                <p style="opacity: 0.8; margin-bottom: 0;">ESTIMATED SELLING PRICE</p>
                <h1 style="font-size: 3.5em; margin-top: 0;">₹ {final_price:.2f} Lakhs</h1>
                <p>{brand} {car_model} • {year}</p>
            </div>
        """, unsafe_allow_html=True)

        # Analysis Metrics
        retention = (final_price / p_price) * 100
        m1, m2 = st.columns(2)
        m1.metric("Value Retained", f"{retention:.1f}%")
        m2.metric("Market Depreciation", f"{100-retention:.1f}%", delta_color="inverse")

        # FEATURE IMPORTANCE (The "Why")
        st.markdown("#### Why this price?")
        # Using feature importances from your trained Random Forest model
        importances = model.feature_importances_
        feature_names = ["Original Price", "Vehicle Age", "Mileage (KMs)", "Fuel Type", "Seller", "Trans", "Owners"]
        
        impact_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        impact_df = impact_df.sort_values(by='Importance', ascending=True)

        fig = px.bar(impact_df, x='Importance', y='Feature', orientation='h', 
                     title="Factors Affecting Your Valuation",
                     color_discrete_sequence=['#1E2A5E'], template="plotly_white")
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

        st.info(f"💡 **Analysis:** For this {brand}, the most critical factor in your valuation was **{impact_df.iloc[-1]['Feature']}**.")
    else:
        st.info("Fill details on the left and click Analyze to generate report.")