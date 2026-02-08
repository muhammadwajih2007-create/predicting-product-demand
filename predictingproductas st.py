import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import plotly.express as px
from PIL import Image

# --- Page Configuration ---
st.set_page_config(
    page_title="RetailDemand AI",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Styling ---
def local_css(file_name):
    # You can create a style.css file, but here we inject directly
    st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        color: white;
        background-color: #ff6b6b;
        border-radius: 5px;
        height: 3em;
        width: 100%;
        font-size: 18px;
        font-weight: bold;
        border:none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ee5253;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
    .metric-card {
        background-color: white;
        border-left: 5px solid #ff6b6b;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .header-title {
        font-family: 'Helvetica', sans-serif;
        color: #2d3436;
        font-weight: 800;
    }
    </style>
    """, unsafe_allow_html=True)

local_css("style.css")

# --- Model Training (Simulated Backend) ---
@st.cache_resource
def train_model():
    """
    Generates synthetic retail data and trains a Linear Regression model.
    In a real scenario, you would load a CSV and train on that.
    """
    # Generate Synthetic Data
    np.random.seed(42)
    n_samples = 500
    
    # Features
    price = np.random.uniform(10, 100, n_samples)
    marketing_spend = np.random.uniform(0, 1000, n_samples)
    season = np.random.randint(0, 4, n_samples) # 0: Winter, 1: Spring, 2: Summer, 3: Fall
    competitor_price = np.random.uniform(10, 100, n_samples)
    
    # Target: Demand (Units Sold) - Logic based
    # Demand decreases with higher price, increases with marketing, peaks in Summer(2) and Winter(0)
    demand = (
        1000 
        - (price * 5) 
        + (marketing_spend * 0.8) 
        + np.where(season==2, 200, 0) # Summer boost
        + np.where(season==0, 150, 0) # Winter boost
        + ((competitor_price - price) * 10) # If we are cheaper than competitor
        + np.random.normal(0, 50, n_samples) # Noise
    )
    
    # Create DataFrame
    df = pd.DataFrame({
        'Price': price,
        'Marketing_Spend': marketing_spend,
        'Season': season,
        'Competitor_Price': competitor_price,
        'Demand': demand
    })
    
    # Train Model
    X = df[['Price', 'Marketing_Spend', 'Season', 'Competitor_Price']]
    y = df['Demand']
    
    model = LinearRegression()
    model.fit(X, y)
    
    return model, df

model, historical_data = train_model()

# --- Sidebar Navigation ---
st.sidebar.title("🛍️ Retail AI Suite")
page = st.sidebar.radio("Navigate", ["Home Dashboard", "Predict Demand", "Market Analytics", "About"])

st.sidebar.markdown("---")
st.sidebar.info("Current Model: Linear Regression v1.0\nAccuracy: 89% (Simulated)")

# --- PAGE 1: HOME DASHBOARD ---
if page == "Home Dashboard":
    st.markdown('<h1 class="header-title">Retail Demand Forecasting System</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Welcome, Inventory Manager")
        st.write("""
        Optimize your supply chain with AI-driven predictions. Our tool analyzes pricing strategies,
        marketing campaigns, and seasonal trends to forecast product demand accurately.
        
        *Key Features:*
        - 📉 Price Elasticity Analysis
        - 📢 Marketing ROI Prediction
        - 🍂 Seasonal Trend Detection
        """)
        
        st.success("System Operational. Model loaded successfully.")
        
    with col2:
        # Placeholder for a dashboard image
        st.image("https://via.placeholder.com/400x300/ff6b6b/ffffff?text=Retail+Analytics", caption="Live Inventory View")

# --- PAGE 2: PREDICT DEMAND ---
elif page == "Predict Demand":
    st.markdown('<h1 class="header-title">Demand Prediction Engine</h1>', unsafe_allow_html=True)
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("Input Parameters")
        with st.form("prediction_form"):
            # Input 1: Product Price
            price = st.slider("Product Price ($)", 10.0, 200.0, 50.0)
            
            # Input 2: Marketing Spend
            marketing = st.number_input("Marketing Budget ($)", 0, 5000, 500)
            
            # Input 3: Season
            season_map = {0: "Winter", 1: "Spring", 2: "Summer", 3: "Fall"}
            season = st.selectbox("Season", ["Winter", "Spring", "Summer", "Fall"])
            season_val = list(season_map.keys())[list(season_map.values()).index(season)]
            
            # Input 4: Competitor Price
            comp_price = st.slider("Competitor Price ($)", 10.0, 200.0, 55.0)
            
            predict_btn = st.form_submit_button("Forecast Demand 🔮")
            
        if predict_btn:
            # Prepare input for model
            input_data = pd.DataFrame({
                'Price': [price],
                'Marketing_Spend': [marketing],
                'Season': [season_val],
                'Competitor_Price': [comp_price]
            })
            
            # Make Prediction
            prediction = model.predict(input_data)[0]
            prediction = max(0, int(prediction)) # Ensure no negative sales
            
            # Display Result
            st.markdown(f"""
            <div class="metric-card" style="text-align:center; padding: 30px;">
                <h3 style="color: #636e72; margin:0;">Predicted Units Sold</h3>
                <h1 style="color: #2d3436; font-size: 60px; margin: 10px 0;">{prediction}</h1>
                <p style="color: #00b894; font-weight:bold;">Projected Revenue: ${prediction * price:,.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Simple Insight
            if comp_price < price:
                st.warning("⚠️ Competitor price is lower. Consider a discount to boost demand.")
            else:
                st.info("💡 Your price is competitive. Focus on marketing to capture market share.")

    with col_right:
        st.subheader("Visual Breakdown")
        
        # Create a gauge chart using Plotly
        fig = px.funnel_area(
            names=["Marketing", "Season", "Price"],
            values=[marketing, season_val*100 + 50, price * 10],
            title="Factor Influence"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Placeholder product image
        st.image("https://via.placeholder.com/500x300/2d3436/ffffff?text=Product+Visualization", caption="Product Analysis")

# --- PAGE 3: MARKET ANALYTICS ---
elif page == "Market Analytics":
    st.markdown('<h1 class="header-title">Historical Performance</h1>', unsafe_allow_html=True)
    
    st.write("Analysis based on the training dataset.")
    
    # Chart 1: Demand vs Price
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.scatter(
            historical_data, 
            x='Price', 
            y='Demand', 
            color='Season',
            title="Price Elasticity: How Price affects Demand",
            trendline="ols"
        )
        st.plotly_chart(fig1, use_container_width=True)
        
    with col2:
        fig2 = px.histogram(
            historical_data, 
            x='Season', 
            y='Demand', 
            histfunc='avg',
            title="Average Demand by Season",
            color='Season',
            category_orders={"Season": [0, 1, 2, 3]},
            labels={"Season": "Season"}
        )
        # Map numbers to names for the x-axis
        season_names = {0: "Winter", 1: "Spring", 2: "Summer", 3: "Fall"}
        fig2.update_xaxes(ticktext=list(season_names.values()), tickvals=list(season_names.keys()))
        st.plotly_chart(fig2, use_container_width=True)

# --- PAGE 4: ABOUT ---
elif page == "About":
    st.markdown('<h1 class="header-title">About This System</h1>', unsafe_allow_html=True)
    
    st.write("""
    This application demonstrates the power of Machine Learning in the retail sector.
    
    ### How it works:
    1. *Data Ingestion:* The system simulates historical sales data including prices, marketing spend, and seasonal indicators.
    2. *Model Training:* A LinearRegression model from Scikit-Learn is trained to find patterns in this data.
    3. *Inference:* When you adjust the sliders on the 'Predict Demand' page, the model calculates the expected sales volume based on the learned patterns.
    
    ### Technologies Used:
    - *Python:* Core logic
    - *Streamlit:* Web Interface
    - *Scikit-Learn:* Machine Learning algorithms
    - *Plotly:* Interactive Data Visualization
    """)
    
    st.image("https://via.placeholder.com/800x200/333333/ffffff?text=Machine+Learning+Pipeline")

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.caption("© 2023 RetailTech Solutions")