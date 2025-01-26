"""Utility functions for the application"""
import pandas as pd
import plotly.express as px
import numpy as np
import shap

def create_feature_importance_plot(model, input_data):
    """Create feature importance plot using SHAP values"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)
    
    fig = px.bar(
        x=np.abs(shap_values).mean(0),
        y=input_data.columns,
        orientation='h',
        title='Feature Importance (SHAP values)'
    )
    return fig

def create_input_dataframe(input_values, feature_columns):
    """Create a DataFrame from input values"""
    return pd.DataFrame([input_values], columns=feature_columns)