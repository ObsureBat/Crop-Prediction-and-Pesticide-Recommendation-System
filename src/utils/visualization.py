"""Visualization utilities"""
import plotly.express as px
import numpy as np
import shap

def create_feature_importance_plot(feature_importance, feature_names):
    """Create a bar plot of feature importance"""
    fig = px.bar(
        x=feature_importance,
        y=feature_names,
        orientation='h',
        title='Feature Importance (SHAP values)'
    )
    return fig