import streamlit as st
import pickle
import numpy as np

# ==========================
# Load the Saved Model
# ==========================
@st.cache_resource
def load_model():
    with open("Cryptocurrency_Model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# ==========================
# Streamlit Page Config
# ==========================
st.set_page_config(
    page_title="Crypto Prediction App",
    page_icon="💹",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ==========================
# App Title
# ==========================
st.title("💹 Cryptocurrency Prediction App")
st.write("This app uses a Machine Learning model to predict cryptocurrency values.")

# ==========================
# User Input Section
# ==========================
st.subheader("🔧 Enter Input Features")

# Example input fields (update according to dataset features)
feature1 = st.number_input("Feature 1 (e.g., Market Cap)", min_value=0.0, step=1.0)
feature2 = st.number_input("Feature 2 (e.g., Volume)", min_value=0.0, step=1.0)
feature3 = st.number_input("Feature 3 (e.g., Transactions)", min_value=0.0, step=1.0)
feature4 = st.number_input("Feature 4 (e.g., Active Addresses)", min_value=0.0, step=1.0)

# Collect into numpy array
features = np.array([[feature1, feature2, feature3, feature4]])

# ==========================
# Prediction Button
# ==========================
if st.button("🚀 Predict"):
    prediction = model.predict(features)
    st.success(f"✅ Predicted Value: {prediction[0]}")
