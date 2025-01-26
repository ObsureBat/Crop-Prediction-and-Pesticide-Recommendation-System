"""Main Streamlit application for Crop Prediction System"""
import streamlit as st
import pandas as pd
import os
import joblib
import plotly.express as px
import plotly.graph_objects as go
from src.data_processor import DataProcessor
from src.models.lightgbm_model import LightGBMModel
from src.pesticide_recommender import PesticideRecommender

def train_model():
    """Train the model and return success status"""
    try:
        data_processor = DataProcessor()
        model = LightGBMModel()
        file_path = "data/Completed_Crop_Pesticide_Dataset.csv"
        X_train, X_test, y_train, y_test = data_processor.load_and_preprocess(file_path)
        metrics = model.train(X_train, X_test, y_train, y_test)
        os.makedirs("models", exist_ok=True)
        model.save_model("models/crop_prediction_model.joblib")
        return True, metrics
    except Exception as e:
        return False, str(e)

def show_advanced_analysis(df, predicted_crop=None):
    """Show advanced analysis with improved UI"""
    st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .chart-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.header("🌾 Advanced Crop Analysis Dashboard")
    
    # Quick Stats
    st.subheader("📊 Quick Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Crops", len(df['label'].unique()))
    with col2:
        st.metric("Avg Temperature", f"{df['temperature'].mean():.1f}°C")
    with col3:
        st.metric("Avg Rainfall", f"{df['rainfall'].mean():.1f}mm")
    with col4:
        st.metric("Avg Humidity", f"{df['humidity'].mean():.1f}%")
    
    # Crop-specific analysis if available
    if predicted_crop:
        st.markdown(f"### 🎯 Analysis for {predicted_crop}")
        df_crop = df[df['label'] == predicted_crop]
        
        if not df_crop.empty:
            # Create tabs for different analyses
            tab1, tab2, tab3 = st.tabs(["Environmental", "Soil Health", "Growth Metrics"])
            
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    # Temperature distribution
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=df_crop['temperature'],
                        name='Temperature',
                        nbinsx=20,
                        marker_color='#72B7B2'
                    ))
                    fig.update_layout(
                        title='Temperature Distribution',
                        xaxis_title='Temperature (°C)',
                        yaxis_title='Count',
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Rainfall vs Humidity
                    fig = px.scatter(
                        df_crop,
                        x='rainfall',
                        y='humidity',
                        title='Rainfall vs Humidity',
                        color_discrete_sequence=['#72B7B2']
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                # Soil Health Metrics
                soil_metrics = ['Soil Microbial Activity (%)', 'Soil Temp (°C)', 'Soil pH Variability']
                values = df_crop[soil_metrics].mean()
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=soil_metrics,
                        y=values,
                        marker_color=['#72B7B2', '#8E44AD', '#2E86C1']
                    )
                ])
                fig.update_layout(
                    title='Soil Health Indicators',
                    yaxis_title='Value',
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    # NDVI Distribution
                    fig = px.box(
                        df_crop,
                        y='NDVI (Normalized Difference Vegetation Index)',
                        title='NDVI Distribution'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Chlorophyll Content
                    fig = px.violin(
                        df_crop,
                        y='Chlorophyll Content (SPAD)',
                        title='Chlorophyll Content Distribution'
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # General Analysis Section
    st.markdown("### 📈 General Crop Analysis")
    
    # Interactive Analysis Selection
    analysis_type = st.selectbox(
        "Choose Analysis Type",
        ["Soil Analysis", "Weather Patterns", "Growth Indicators"],
        format_func=lambda x: {
            "Soil Analysis": "🌱 Soil Analysis",
            "Weather Patterns": "🌤️ Weather Patterns",
            "Growth Indicators": "📊 Growth Indicators"
        }[x]
    )
    
    if analysis_type == "Soil Analysis":
        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(
                df,
                y=['Soil Microbial Activity (%)', 'Soil pH Variability'],
                title='Soil Health Distribution',
                color_discrete_sequence=['#72B7B2', '#2E86C1']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(
                df,
                x='Soil Temp (°C)',
                y='NDVI (Normalized Difference Vegetation Index)',
                title='Soil Temperature vs NDVI',
                color_discrete_sequence=['#72B7B2']
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Weather Patterns":
        col1, col2 = st.columns(2)
        with col1:
            avg_rainfall = df.groupby('label')['rainfall'].mean().sort_values(ascending=False)
            fig = px.bar(
                avg_rainfall,
                title='Average Rainfall by Crop',
                color_discrete_sequence=['#2E86C1']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(
                df,
                x='Wind Speed (km/h)',
                y='Cloud Cover (%)',
                title='Wind Speed vs Cloud Cover',
                color_discrete_sequence=['#72B7B2']
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Growth Indicators":
        # Create a correlation heatmap for growth indicators
        growth_cols = [
            'NDVI (Normalized Difference Vegetation Index)',
            'Chlorophyll Content (SPAD)',
            'Evapotranspiration Rate (mm/day)'
        ]
        corr = df[growth_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr,
            x=growth_cols,
            y=growth_cols,
            colorscale='Viridis'
        ))
        fig.update_layout(
            title='Growth Indicators Correlation',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

def main():
    st.title("Crop Prediction and Pesticide Recommendation System")
    
    # Initialize objects
    data_processor = DataProcessor()
    model = LightGBMModel()
    recommender = PesticideRecommender()
    
    # Load dataset for advanced analysis
    df = pd.read_csv("data/Completed_Crop_Pesticide_Dataset.csv")
    
    # Check if model exists, if not train it
    if not os.path.exists("models/crop_prediction_model.joblib"):
        with st.spinner("Training initial model..."):
            success, metrics = train_model()
            if success:
                st.success("Initial model trained successfully!")
            else:
                st.error(f"Error training initial model: {metrics}")
                return
    
    # Navigation
    page = st.sidebar.selectbox("Navigation", ["Crop Prediction", "Advanced Analysis"])
    
    if page == "Crop Prediction":
        # Input form for basic features
        col1, col2 = st.columns(2)
        
        with col1:
            n = st.number_input("Nitrogen (N)", 0, 140, 50)
            p = st.number_input("Phosphorus (P)", 0, 145, 50)
            k = st.number_input("Potassium (K)", 0, 205, 50)
            temp = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
        
        with col2:
            humidity = st.number_input("Humidity (%)", 0.0, 100.0, 50.0)
            ph = st.number_input("pH", 0.0, 14.0, 7.0)
            rainfall = st.number_input("Rainfall (mm)", 0.0, 300.0, 100.0)
        
        predicted_crop = None
        if st.button("Predict"):
            try:
                model.load_model("models/crop_prediction_model.joblib")
                
                input_data = pd.DataFrame([[n, p, k, temp, humidity, ph, rainfall]], 
                                       columns=data_processor.features)
                
                scaled_input = data_processor.preprocess_input(input_data)
                prediction = model.predict(scaled_input)
                predicted_crop = data_processor.decode_prediction(prediction[0])
                
                st.success(f"Recommended Crop: {predicted_crop}")
                
                pesticides = recommender.get_pesticide_recommendations(predicted_crop)
                if pesticides:
                    st.subheader("Pest Management Details")
                    for pest_info in pesticides:
                        with st.expander(f"🌱 Pest: {pest_info['Pest']}", expanded=True):
                            st.markdown(f"""
                            **Recommended Pesticide:** {pest_info['Pesticide']}
                            
                            **Active Ingredient:** {pest_info['Active Ingredient']}
                            
                            **Usage Guidelines:** {pest_info['Usage Guidelines']}
                            """)
                else:
                    st.info("No specific pesticide recommendations available for this crop.")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
        
        # Show advanced analysis for predicted crop
        if predicted_crop:
            if st.button("Show Advanced Analysis for Predicted Crop"):
                show_advanced_analysis(df, predicted_crop)
    
    else:  # Advanced Analysis page
        show_advanced_analysis(df)
    
    # Add retrain button in sidebar
    if st.sidebar.button("Retrain Model"):
        with st.spinner("Retraining model..."):
            success, metrics = train_model()
            if success:
                st.success("Model retrained successfully!")
                st.sidebar.write("Model Performance:")
                st.sidebar.write(f"Accuracy: {metrics['accuracy']:.2f}")
            else:
                st.error(f"Error retraining model: {metrics}")

if __name__ == "__main__":
    main()