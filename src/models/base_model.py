"""Base model interface for crop prediction models"""
from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def train(self, X_train, X_test, y_train, y_test):
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Make predictions"""
        pass
    
    @abstractmethod
    def save_model(self, path):
        """Save model to disk"""
        pass
    
    @abstractmethod
    def load_model(self, path):
        """Load model from disk"""
        pass