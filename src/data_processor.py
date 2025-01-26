"""Data processing and preparation module"""
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.scaler_file = "models/scaler.joblib"
        self.label_encoder_file = "models/label_encoder.joblib"
        
        # Define only the basic features needed for prediction
        self.features = [
            'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'
        ]
        
    def load_and_preprocess(self, file_path):
        """Load and preprocess the dataset"""
        try:
            df = pd.read_csv(file_path)
            X = df[self.features]  # Only basic features
            y = df['label']
            
            X_scaled = self.scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
            y_encoded = self.label_encoder.fit_transform(y)
            
            os.makedirs("models", exist_ok=True)
            joblib.dump(self.scaler, self.scaler_file)
            joblib.dump(self.label_encoder, self.label_encoder_file)
            
            return train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
            
        except Exception as e:
            raise Exception(f"Error processing data: {str(e)}")
    
    def get_feature_ranges(self, file_path):
        """Get min-max ranges for all features"""
        df = pd.read_csv(file_path)
        return {col: (df[col].min(), df[col].max()) for col in self.features}
    
    def preprocess_input(self, input_data):
        """Preprocess input data for prediction"""
        try:
            if os.path.exists(self.scaler_file):
                self.scaler = joblib.load(self.scaler_file)
            else:
                raise Exception("Model not trained. Please train the model first.")
            return self.scaler.transform(input_data)
        except Exception as e:
            raise Exception(f"Error preprocessing input: {str(e)}")
    
    def decode_prediction(self, prediction):
        """Decode the model's prediction back to crop name"""
        try:
            if os.path.exists(self.label_encoder_file):
                self.label_encoder = joblib.load(self.label_encoder_file)
            else:
                raise Exception("Model not trained. Please train the model first.")
            return self.label_encoder.inverse_transform([prediction])[0]
        except Exception as e:
            raise Exception(f"Error decoding prediction: {str(e)}")