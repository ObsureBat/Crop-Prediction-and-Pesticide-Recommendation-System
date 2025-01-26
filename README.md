# Crop Prediction and Pesticide Recommendation System

This application uses machine learning to predict suitable crops based on environmental factors and provides pesticide recommendations.

## Features

- Crop prediction based on environmental factors
- Pesticide recommendations for predicted crops
- Interactive UI with Streamlit
- Model training interface
- Feature importance visualization
- Detailed prediction analysis

## Setup

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Place your crop dataset in the `data` folder as `crop_data.csv`

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. First, go to the "Train Model" page and train the model
2. Switch to "Make Prediction" to input environmental factors
3. Get crop predictions and pesticide recommendations

## Data Format

The input CSV file should have the following columns:
- N (Nitrogen)
- P (Phosphorus)
- K (Potassium)
- temperature
- humidity
- ph
- rainfall
- label (crop name)

## Model Details

- Uses LightGBM for prediction
- Features standardization
- SHAP values for feature importance
- Model persistence for reuse