"""Configuration settings for the application"""

# Model parameters
MODEL_PARAMS = {
    'objective': 'multiclass',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'verbose': -1
}

# File paths
MODEL_PATH = 'models/crop_prediction_model.joblib'
DATA_PATH = 'data/crop_data.csv'

# Feature columns
FEATURE_COLUMNS = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

# Input ranges for validation
INPUT_RANGES = {
    'N': (0, 140),
    'P': (0, 145),
    'K': (0, 205),
    'temperature': (0.0, 50.0),
    'humidity': (0.0, 100.0),
    'ph': (0.0, 14.0),
    'rainfall': (0.0, 300.0)
}