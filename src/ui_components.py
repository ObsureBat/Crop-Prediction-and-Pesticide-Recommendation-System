"""Streamlit UI components"""
import streamlit as st
from .config import INPUT_RANGES

def create_sidebar():
    """Create the sidebar navigation"""
    return st.sidebar.selectbox("Choose a page", ["Train Model", "Make Prediction"])

def create_input_form():
    """Create the input form for predictions"""
    col1, col2 = st.columns(2)
    
    with col1:
        nitrogen = st.number_input("Nitrogen (N)", *INPUT_RANGES['N'], 50)
        phosphorus = st.number_input("Phosphorus (P)", *INPUT_RANGES['P'], 50)
        potassium = st.number_input("Potassium (K)", *INPUT_RANGES['K'], 50)
        temperature = st.number_input("Temperature (°C)", *INPUT_RANGES['temperature'], 25.0)
    
    with col2:
        humidity = st.number_input("Humidity (%)", *INPUT_RANGES['humidity'], 50.0)
        ph = st.number_input("pH", *INPUT_RANGES['ph'], 7.0)
        rainfall = st.number_input("Rainfall (mm)", *INPUT_RANGES['rainfall'], 100.0)
    
    return nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall

def display_results(predicted_crop, pesticides, feature_importance_plot):
    """Display prediction results"""
    st.success(f"Predicted Crop: {predicted_crop}")
    
    st.subheader("Recommended Pesticides:")
    for pesticide in pesticides:
        st.write(f"• {pesticide}")
    
    st.subheader("Feature Importance for this Prediction")
    st.plotly_chart(feature_importance_plot)