"""Model training and prediction module"""
import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import joblib
from .config import MODEL_PARAMS

class ModelTrainer:
    def __init__(self):
        self.model = None
        
    def train(self, X_train, X_test, y_train, y_test):
        """Train the LightGBM model"""
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
            verbose_eval=10
        )
        
        # Evaluate model
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        return accuracy, report
    
    def predict(self, X):
        """Make predictions using the trained model"""
        predictions = self.model.predict(X)
        return np.argmax(predictions, axis=1)
    
    def save_model(self, path):
        """Save the trained model"""
        joblib.dump(self.model, path)
    
    def load_model(self, path):
        """Load a trained model"""
        self.model = joblib.load(path)