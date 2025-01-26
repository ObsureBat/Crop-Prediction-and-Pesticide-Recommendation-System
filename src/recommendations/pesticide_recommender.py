"""Pesticide recommendation module"""
from ..utils.data_validation import validate_crop_name

class PesticideRecommender:
    def __init__(self):
        self.pesticide_map = {
            'rice': ['Carbofuran', 'Chlorpyrifos', 'Fipronil'],
            'maize': ['Atrazine', 'Glyphosate', 'Pendimethalin'],
            # ... (rest of the pesticide map)
        }
    
    def get_pesticide_recommendations(self, crop):
        crop = validate_crop_name(crop)
        return self.pesticide_map.get(crop.lower(), [])