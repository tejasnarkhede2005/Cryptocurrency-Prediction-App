import streamlit as st
import pickle
import numpy as np
import base64

# ==========================
# Load the Saved Model
# ==========================
@st.cache_resource
def load_model():
    # This is a placeholder for model loading.
    # In a real scenario, you would load your trained model.
    # For demonstration, we'll create a dummy model.
    from sklearn.linear_model import LinearRegression
    # Create a dummy model that predicts based on the first feature
    dummy_model = LinearRegression()
    # Fit with some dummy data so it can predict
    dummy_model.fit(np.array([[1,1,1,1]]), np.array([1]))
    return dummy_model

model = load_model()

# ==========================
# Page Config
# ==========================
st.set_page_config(
    page_title="Crypto Oracle",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# Custom CSS for Navbar and Styling
# ==========================
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background-color: #f0f2f6;
    }

    /* Navigation Bar */
    .navbar {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        z-index: 1;
        background-color: #ffffff;
        padding: 1rem 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: all 0.3s ease;
    }

    .navbar:hover {
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }

    .navbar-brand {
        font-size: 1.75rem;
        font-weight: bold;
        color: #1a1a1a;
        text-decoration: none;
    }
    
    .navbar-brand .icon {
        margin-right: 0.5rem;
    }

    .nav-links {
        list-style: none;
        margin: 0;
        padding: 0;
        display: flex;
        align-items: center;
    }

    .nav-item {
        margin-left: 1.5rem;
    }

    .nav-link {
        color: #555;
        text-decoration: none;
        font-weight: 500;
        font-size: 1rem;
        padding: 0.5rem 0;
        position: relative;
        transition: color 0.3s ease;
    }

    .nav-link::after {
        content: '';
        position: absolute;
        width: 0;
        height: 2px;
        bottom: 0;
        left: 0;
        background-color: #007bff;
        transition: width 0.3s ease;
    }

    .nav-link:hover {
        color: #007bff;
    }

    .nav-link:hover::after {
        width: 100%;
    }
    
    /* Main content padding to avoid overlap with navbar */
    .main-content {
        padding-top: 80px; /* Adjust based on navbar height */
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    /* Card styling for the app */
    .card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-top: 2rem;
    }
    
    /* Custom button style */
    div.stButton > button {
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: bold;
        border: none;
        transition: background-color 0.3s, transform 0.2s;
    }

    div.stButton > button:hover {
        background-color: #0056b3;
        transform: scale(1.05);
    }
    
    /* Success message styling */
    .stSuccess {
        background-color: #e6ffed;
        border-left: 5px solid #4caf50;
        border-radius: 8px;
        padding: 1rem;
        color: #2e7d32;
    }

</style>
""", unsafe_allow_html=True)

# ==========================
# Navbar HTML
# ==========================
st.markdown("""
<nav class="navbar">
    <a href="#" class="navbar-brand"><span class="icon">🔮</span> Crypto Oracle</a>
    <ul class="nav-links">
        <li class="nav-item"><a href="#" class="nav-link">Home</a></li>
        <li class="nav-item"><a href="#" class="nav-link">About</a></li>
        <li class="nav-item"><a href="#" class="nav-link">Dashboard</a></li>
        <li class="nav-item"><a href="#" class="nav-link">Contact</a></li>
    </ul>
</nav>
""", unsafe_allow_html=True)


# Main content container
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# ==========================
# App Title inside the main content
# ==========================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.title("📈 Cryptocurrency Price Predictor")
st.write("This app uses a Machine Learning model to predict cryptocurrency values based on market features.")
st.markdown('</div>', unsafe_allow_html=True)

# ==========================
# User Input Section in a card
# ==========================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🔧 Enter Market Features")

col1, col2 = st.columns(2)
with col1:
    feature1 = st.number_input("Market Cap (in billions USD)", min_value=0.0, step=1.0, value=500.0)
    feature2 = st.number_input("24h Trading Volume (in billions USD)", min_value=0.0, step=0.1, value=50.0)
with col2:
    feature3 = st.number_input("Daily Transactions (in thousands)", min_value=0.0, step=1.0, value=300.0)
    feature4 = st.number_input("Active Addresses (in thousands)", min_value=0.0, step=1.0, value=800.0)

# Collect into numpy array
features = np.array([[feature1, feature2, feature3, feature4]])
st.markdown('</div>', unsafe_allow_html=True)

# ==========================
# Prediction Section
# ==========================
st.markdown('<div class="card">', unsafe_allow_html=True)
if st.button("🚀 Predict Price"):
    with st.spinner('Analyzing the market...'):
        # In a real app, you might have some delay or complex calculation
        import time
        time.sleep(1) 
        
        # NOTE: Using a dummy model, so prediction is not real
        # The placeholder model just uses the first feature as prediction
        prediction = model.predict(features) * 100 
        
        st.success(f"**Predicted Value:** `${prediction[0]:,.2f}`")
st.markdown('</div>', unsafe_allow_html=True)


# Close the main content div
st.markdown('</div>', unsafe_allow_html=True)
