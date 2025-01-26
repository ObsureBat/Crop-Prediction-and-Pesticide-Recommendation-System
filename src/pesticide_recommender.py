"""Pesticide recommendation module"""
import pandas as pd

class PesticideRecommender:
    def __init__(self):
        self.pest_data = None
        
    def load_pest_data(self, file_path):
        """Load pest and pesticide data from CSV"""
        self.pest_data = pd.read_csv(file_path)
    
    def get_pesticide_recommendations(self, crop):
        """Get detailed pesticide recommendations for a given crop"""
        if self.pest_data is None:
            self.load_pest_data("data/Completed_Crop_Pesticide_Dataset.csv")
            
        # Filter data for the predicted crop
        crop_pests = self.pest_data[self.pest_data['label'].str.lower() == crop.lower()]
        
        if crop_pests.empty:
            return None
            
        # Get unique pest-pesticide combinations
        recommendations = crop_pests[['Pest', 'Pesticide', 'Active Ingredient', 'Usage Guidelines']].drop_duplicates()
        
        return recommendations.to_dict('records')