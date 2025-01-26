"""LightGBM implementation of crop prediction model"""
import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import joblib
from .base_model import BaseModel
from ..config import MODEL_PARAMS

class LightGBMModel(BaseModel):
    def __init__(self):
        self.model = None
        
    def train(self, X_train, X_test, y_train, y_test):
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test)
        
        params = {
            **MODEL_PARAMS,
            'num_class': len(np.unique(y_train)),
            'metric': 'multi_logloss'
        }
        
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[test_data],
            early_stopping_rounds=10,
            verbose_eval=False
        )
        
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        return metrics
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        predictions = self.model.predict(X)
        return np.argmax(predictions, axis=1)
    
    def save_model(self, path):
        if self.model is None:
            raise ValueError("No model to save. Please train the model first.")
        joblib.dump(self.model, path)
    
    def load_model(self, path):
        self.model = joblib.load(path)