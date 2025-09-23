import streamlit as st

import numpy as np

import time

import requests

from sklearn.linear_model import LinearRegression



# ==========================

# App Configuration & Model

# ==========================



st.set_page_config(

    page_title="CryptoCurrency Prediction",

    page_icon="🔮",

    layout="centered",

    initial_sidebar_state="collapsed"

)



@st.cache_resource

def load_model():

    """A dummy model that responds to input for demonstration."""

    dummy_model = LinearRegression()

    # Fit on a zero array just to initialize the model structure

    dummy_model.fit(np.zeros((1, 4)), np.zeros(1))

    # Manually set coefficients and intercept for a dynamic response

    dummy_model.coef_ = np.array([0.1, 0.2, 0.05, 0.15])

    dummy_model.intercept_ = 150.0  # Set a base value for the prediction

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

# Live API Data Fetching

# ==========================



@st.cache_data(ttl=900) # Cache data for 15 minutes to avoid API rate limits

def get_live_prices():

    """

    Fetches live cryptocurrency prices from the CoinGecko API.

    """

    url = "https://api.coingecko.com/api/v3/coins/markets"

    coin_ids = "bitcoin,ethereum,ripple,tether,binancecoin,solana,usd-coin,dogecoin"

    params = {

        "vs_currency": "usd",

        "ids": coin_ids,

        "order": "market_cap_desc",

        "per_page": 10,

        "page": 1,

        "sparkline": "false"

    }

    try:

        response = requests.get(url, params=params)

        response.raise_for_status()  # Raise an error for bad responses (4xx or 5xx)

        return response.json()

    except requests.exceptions.RequestException as e:

        st.error(f"Error fetching live prices: {e}")

        return None





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

    

    

    h1 {

        font-size: 1.8rem;

        font-weight: 600;

        color: #f0f6fc;

        display: flex;

        align-items: center;

        margin-bottom: 2rem;

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

    .main-content .stButton > button {

        background: linear-gradient(45deg, #3672f8, #58a6ff);

        color: white;

        width: 100%;

        border-radius: 8px;

        padding: 16px 0;

        font-weight: bold;

        font-size: 1.1rem;

        border: none;

        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);

        transition: transform 0.2s, box-shadow 0.2s;

        margin-top: 1rem;

    }

    .main-content .stButton > button:hover {

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



    /* Bottom Navigation Bar */

    .bottom-nav {

        position: fixed;

        bottom: 0;

        left: 0;

        width: 100%;

        background: linear-gradient(to right, #0f2027, #203a43, #2c5364); 

        box-shadow: 0 -4px 12px rgba(0, 0, 0, 0.25);

        display: flex;

        justify-content: space-around;

        padding: 0.5rem 0;

        z-index: 100;

    }

    

    .nav-item {

        display: flex;

        flex-direction: column;

        align-items: center;

        color: #8b949e;

        cursor: pointer;

        font-size: 0.75rem;

        width: 33.33%;

        padding: 0.25rem 0;

        border: none;

        background: none;

        transition: color 0.2s;

    }

    .nav-item.active {

        color: #58a6ff;

    }

    .nav-item:hover {

        color: #58a6ff;

    }

    .nav-item svg {

        width: 24px;

        height: 24px;

        margin-bottom: 4px;

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

    

    /* Transparent Nav Button Container */

    .bottom-nav-container {

        position: fixed;

        bottom: 0;

        left: 0;

        width: 100%;

        z-index: 101; /* Above the visual nav bar */

        height: 65px; /* Match visual bar height */

    }

    .bottom-nav-container .stButton > button {

        background: transparent;

        border: none;

        color: transparent; /* Hide button text */

        width: 100%;

        height: 65px;

    }

    .bottom-nav-container .stButton > button:hover {

        background-color: rgba(88, 166, 255, 0.1);

        border-radius: 10px;

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

            prediction = model.predict(features)

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

    """Renders the live market page using data from an API."""

   

    st.markdown("<h2>Live Market</h2>", unsafe_allow_html=True)

    

    market_data = get_live_prices()

    

    if market_data:

        for item in market_data:

            change_class = "positive" if item.get("price_change_percentage_24h", 0) >= 0 else "negative"

            st.markdown(

                f"""

                <div class="crypto-item">

                    <div class="crypto-info">

                        <img src="{item.get('image', '')}" class="crypto-icon" alt="{item.get('name', '')}">

                        <div class="crypto-name">

                            <div class="symbol">{item.get('symbol', '').upper()}</div>

                            <div class="fullname">{item.get('name', '')}</div>

                        </div>

                    </div>

                    <div class="crypto-price">

                        <div class="price">${item.get('current_price', 0):,.2f}</div>

                        <div class="change {change_class}">{item.get('price_change_percentage_24h', 0):.2f}%</div>

                    </div>

                </div>

                """,

                unsafe_allow_html=True

            )

    else:

        st.warning("Could not retrieve live market data. Please try again later.")

    

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

st.markdown('<h1><span class="icon">🔮</span>Crypto Currency Prediction</h1>', unsafe_allow_html=True)



# Page content based on session state

if st.session_state.active_page == 'Predictor':

    render_predictor()

elif st.session_state.active_page == 'Market':

    render_market()

elif st.session_state.active_page == 'History':

    render_history()



st.markdown('</div>', unsafe_allow_html=True)





# ==========================

# Bottom Navigation Logic

# ==========================



# This container holds invisible buttons overlaid on the visual nav bar

st.markdown('<div class="bottom-nav-container">', unsafe_allow_html=True)

nav_cols = st.columns(3)

with nav_cols[0]:

    if st.button("Predictor", key="pred_btn_h", use_container_width=True):

        st.session_state.active_page = 'Predictor'

        st.rerun()

with nav_cols[1]:

    if st.button("Market", key="market_btn_h", use_container_width=True):

        st.session_state.active_page = 'Market'

        st.rerun()

with nav_cols[2]:

    if st.button("History", key="history_btn_h", use_container_width=True):

        st.session_state.active_page = 'History'

        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)



# HTML for the visual navigation bar

st.markdown(

    f"""

    <div class="bottom-nav">

        <div class="nav-item {'active' if st.session_state.active_page == 'Predictor' else ''}">

            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M10.5 6h9.75M10.5 6a1.5 1.5 0 11-3 0m3 0a1.5 1.5 0 10-3 0M3.75 6H7.5m3 12h9.75m-9.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-3.75 0H7.5m9-6h3.75m-3.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-9.75 0h9.75" /></svg>

            Predictor

        </div>

        <div class="nav-item {'active' if st.session_state.active_page == 'Market' else ''}">

            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M2.25 18L9 11.25l4.306 4.307a11.95 11.95 0 015.814-5.519l2.74-1.22m0 0l-5.94-2.28m5.94 2.28l-2.28 5.941" /></svg>

            Market

        </div>

        <div class="nav-item {'active' if st.session_state.active_page == 'History' else ''}">

            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25" /></svg>

            History

        </div>

    </div>

    """, unsafe_allow_html=True

)




Show thinking
