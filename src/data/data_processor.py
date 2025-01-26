"""Data processing and preparation module"""
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_and_preprocess(self, file_path):
        df = pd.read_csv(file_path)
        
        X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
        y = df['label']
        
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        y_encoded = self.label_encoder.fit_transform(y)
        
        return train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
    
    def preprocess_input(self, input_data):
        return self.scaler.transform(input_data)
    
    def decode_prediction(self, prediction):
        return self.label_encoder.inverse_transform([prediction])[0]