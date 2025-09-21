import streamlit as st
import numpy as np
import time
import pandas as pd
from sklearn.linear_model import LinearRegression

# ==========================
# App Configuration & Model
# ==========================

st.set_page_config(
    page_title="Crypto Predictor",
    page_icon="🔮",
    layout="centered", # Use centered layout for a mobile feel
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def load_model():
    """A dummy model for demonstration."""
    dummy_model = LinearRegression()
    dummy_model.fit(np.array([[1, 1, 1, 1]]), np.array([1]))
    return dummy_model

model = load_model()

# ==========================
# Session State Initialization
# ==========================

if 'active_page' not in st.session_state:
    st.session_state.active_page = 'Predictor'
if 'history' not in st.session_state:
    st.session_state.history = []

# ==========================
# Custom CSS for Mobile App Look
# ==========================

st.markdown("""
<style>
    /* Reset and Base Styles */
    body {
        margin: 0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
    }

    /* Main App Container */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Hide Streamlit Header/Footer */
    header, footer {
        visibility: hidden;
    }

    /* Main Content Area */
    .main-content {
        padding: 1rem 1rem 6rem 1rem; /* Padding for content */
        max-width: 600px;
        margin: auto;
    }
    
    /* Custom Card */
    .custom-card {
        background-color: #161B22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 1rem;
        transition: box-shadow 0.3s ease;
    }
    .custom-card:hover {
        box-shadow: 0 0 15px rgba(88, 166, 255, 0.2);
    }
    
    h1 {
        font-size: 1.8rem;
        font-weight: 600;
        color: #f0f6fc;
        display: flex;
        align-items: center;
    }
    
    h1 .icon {
        font-size: 2rem;
        margin-right: 0.8rem;
    }

    h2 {
        font-size: 1.3rem;
        color: #8b949e;
        border-bottom: 1px solid #30363d;
        padding-bottom: 0.5rem;
        margin-top: 0;
    }

    /* Sliders */
    .stSlider [data-baseweb="slider"] {
        padding-bottom: 10px;
    }
    
    /* Predict Button */
    div.stButton > button {
        background: linear-gradient(45deg, #3672f8, #58a6ff);
        color: white;
        width: 100%;
        border-radius: 8px;
        padding: 16px 0; /* Increased padding for a larger button */
        font-weight: bold;
        font-size: 1.1rem; /* Larger font size */
        border: none;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s, box-shadow 0.2s;
        margin-top: 1rem; /* Added margin on top */
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(88, 166, 255, 0.3);
    }

    /* Prediction Result */
    .prediction-result {
        background-color: rgba(35, 134, 54, 0.2);
        border: 1px solid #2ea043;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        margin-top: 1.5rem;
    }
    .prediction-result .label {
        font-size: 1rem;
        color: #8b949e;
    }
    .prediction-result .value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #3fb950;
    }

    /* Market List */
    .crypto-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem 0;
        border-bottom: 1px solid #30363d;
    }
    .crypto-item:last-child {
        border-bottom: none;
    }
    .crypto-info {
        display: flex;
        align-items: center;
    }
    .crypto-icon {
        width: 40px;
        height: 40px;
        margin-right: 1rem;
    }
    .crypto-name .symbol {
        font-size: 1.1rem;
        font-weight: 600;
        color: #f0f6fc;
    }
    .crypto-name .fullname {
        font-size: 0.9rem;
        color: #8b949e;
    }
    .crypto-price .price {
        font-size: 1.1rem;
        font-weight: 600;
        color: #f0f6fc;
        text-align: right;
    }
    .crypto-price .change {
        font-size: 0.9rem;
        text-align: right;
    }
    .positive { color: #3fb950; }
    .negative { color: #f85149; }

    /* History List */
    .history-item {
        background-color: #0d1117;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        border: 1px solid #30363d;
    }

</style>
""", unsafe_allow_html=True)

# ==========================
# Page Rendering Functions
# ==========================

def render_predictor():
    """Renders the main predictor interface."""
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("<h2>Input Market Features</h2>", unsafe_allow_html=True)

    feature1 = st.slider("Market Cap (in billions USD)", min_value=1.0, max_value=2000.0, value=500.0, step=1.0)
    feature2 = st.slider("24h Trading Volume (in billions USD)", min_value=0.1, max_value=500.0, value=50.0, step=0.1)
    feature3 = st.slider("Daily Transactions (in thousands)", min_value=1.0, max_value=2000.0, value=300.0, step=1.0)
    feature4 = st.slider("Active Addresses (in thousands)", min_value=1.0, max_value=2000.0, value=800.0, step=1.0)

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🚀 Predict Price"):
        with st.spinner('Oracle is consulting the stars...'):
            time.sleep(1.5)
            features = np.array([[feature1, feature2, feature3, feature4]])
            prediction = model.predict(features) * 100
            pred_value = f"${prediction[0]:,.2f}"
            
            st.session_state.history.insert(0, {"value": pred_value, "features": [feature1, feature2, feature3, feature4]})
            
            st.markdown(
                f"""
                <div class="custom-card prediction-result">
                    <div class="label">Predicted Value</div>
                    <div class="value">{pred_value}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

def render_market():
    """Renders the simulated live market page."""
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("<h2>Live Market</h2>", unsafe_allow_html=True)
    
    # Dummy data
    market_data = [
        {"symbol": "BTC", "name": "Bitcoin", "price": 68123.45, "change": 2.5, "icon": "https://s2.coinmarketcap.com/static/img/coins/64x64/1.png"},
        {"symbol": "ETH", "name": "Ethereum", "price": 3567.89, "change": -1.2, "icon": "https://s2.coinmarketcap.com/static/img/coins/64x64/1027.png"},
        {"symbol": "SOL", "name": "Solana", "price": 165.21, "change": 5.8, "icon": "https://s2.coinmarketcap.com/static/img/coins/64x64/5426.png"},
        {"symbol": "DOGE", "name": "Dogecoin", "price": 0.158, "change": 0.5, "icon": "https://s2.coinmarketcap.com/static/img/coins/64x64/74.png"},
    ]
    
    for item in market_data:
        change_class = "positive" if item["change"] >= 0 else "negative"
        st.markdown(
            f"""
            <div class="crypto-item">
                <div class="crypto-info">
                    <img src="{item['icon']}" class="crypto-icon" alt="{item['name']}">
                    <div class="crypto-name">
                        <div class="symbol">{item['symbol']}</div>
                        <div class="fullname">{item['name']}</div>
                    </div>
                </div>
                <div class="crypto-price">
                    <div class="price">${item['price']:,.2f}</div>
                    <div class="change {change_class}">{'+' if item['change'] >= 0 else ''}{item['change']}%</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

def render_history():
    """Renders the prediction history page."""
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("<h2>Prediction History</h2>", unsafe_allow_html=True)

    if not st.session_state.history:
        st.info("No predictions made yet.")
    else:
        for i, record in enumerate(st.session_state.history):
            st.markdown(
                f"""
                <div class="history-item">
                    <b>Prediction #{len(st.session_state.history) - i}:</b> <span style="float: right; font-size: 1.5em; color: #3fb950; font-weight: bold;">{record['value']}</span><br>
                    <small>Features: {', '.join(map(str, record['features']))}</small>
                </div>
                """,
                unsafe_allow_html=True
            )
    st.markdown('</div>', unsafe_allow_html=True)


# ==========================
# Main App Layout
# ==========================
st.markdown('<div class="main-content">', unsafe_allow_html=True)
st.markdown('<h1><span class="icon">🔮</span>Crypto Predictor</h1>', unsafe_allow_html=True)

# Page content based on session state
if st.session_state.active_page == 'Predictor':
    render_predictor()
elif st.session_state.active_page == 'Market':
    render_market()
elif st.session_state.active_page == 'History':
    render_history()

st.markdown('</div>', unsafe_allow_html=True)


# ==========================
# Vertical Navigation Logic
# ==========================

# This container will hold our buttons and we'll style it with CSS
st.markdown('<div class="vertical-nav">', unsafe_allow_html=True)
nav_container = st.container()
with nav_container:
    if st.button("🔮 Predictor", key="pred_btn_v"):
        st.session_state.active_page = 'Predictor'
        st.rerun()
    if st.button("📈 Market", key="market_btn_v"):
        st.session_state.active_page = 'Market'
        st.rerun()
    if st.button("📜 History", key="history_btn_v"):
        st.session_state.active_page = 'History'
        st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# CSS for the vertical navigation container and buttons
vertical_nav_css = """
<style>
    .vertical-nav {
        position: fixed;
        right: 20px;
        top: 50%;
        transform: translateY(-50%);
        z-index: 1000;
        display: flex;
        flex-direction: column;
        gap: 15px; /* Space between buttons */
    }
    
    /* This targets the Streamlit button styling specifically within our vertical nav */
    .vertical-nav .stButton > button {
        background-color: #161B22;
        color: #c9d1d9; /* Lighter text for readability */
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 12px;
        width: 130px; 
        font-weight: 500;
        font-size: 1rem;
        text-align: left; /* Align icon and text to the left */
        transition: all 0.2s ease-in-out;
    }
    
    .vertical-nav .stButton > button:hover {
        border-color: #58a6ff;
        color: #58a6ff;
    }

    /* Override the default Streamlit button styling for this specific container */
    .vertical-nav div[data-testid="stButton"] button {
        background: linear-gradient(45deg, #3672f8, #58a6ff) !important;
        background-color: #161B22 !important;
    }
</style>
"""
st.markdown(vertical_nav_css, unsafe_allow_html=True)

# Dynamic CSS to highlight the active button
active_page = st.session_state.active_page
active_button_selector = ""
# We need to target the element-container that Streamlit wraps around each button
if active_page == 'Predictor':
    active_button_selector = ".vertical-nav .element-container:nth-child(1) .stButton > button"
elif active_page == 'Market':
    active_button_selector = ".vertical-nav .element-container:nth-child(2) .stButton > button"
elif active_page == 'History':
    active_button_selector = ".vertical-nav .element-container:nth-child(3) .stButton > button"

active_button_css = f"""
<style>
    {active_button_selector} {{
        background-color: #58a6ff !important;
        color: white !important; /* Use important to override Streamlit's default styles */
        border: 1px solid #58a6ff !important;
    }}
</style>
"""
st.markdown(active_button_css, unsafe_allow_html=True)

