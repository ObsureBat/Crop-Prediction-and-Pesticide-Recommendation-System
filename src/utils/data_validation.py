"""Validation utilities for data processing"""
def validate_crop_name(crop):
    """Validate and sanitize crop name"""
    if not isinstance(crop, str):
        raise ValueError("Crop name must be a string")
    return crop.strip().lower()

def validate_input_ranges(value, min_val, max_val, name):
    """Validate input values are within acceptable ranges"""
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    if value < min_val or value > max_val:
        raise ValueError(f"{name} must be between {min_val} and {max_val}")
    return value