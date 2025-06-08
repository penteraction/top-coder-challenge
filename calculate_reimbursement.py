#!/usr/bin/env python3

import sys
import os
import pickle
import numpy as np
from random_forest_model import engineer_features

# Global variables to cache models
_RF_MODEL = None
_GB_MODEL = None
_CACHE = {}

# Preload models at module import time
try:
    with open('rf_model.pkl', 'rb') as f:
        _RF_MODEL = pickle.load(f)
    with open('gb_model.pkl', 'rb') as f:
        _GB_MODEL = pickle.load(f)
except Exception as e:
    # If loading fails, we'll try again in load_models()
    pass

def load_models():
    """Load pre-trained models once"""
    global _RF_MODEL, _GB_MODEL
    
    # Only load models if they haven't been loaded yet
    if _RF_MODEL is None or _GB_MODEL is None:
        try:
            with open('rf_model.pkl', 'rb') as f:
                _RF_MODEL = pickle.load(f)
            with open('gb_model.pkl', 'rb') as f:
                _GB_MODEL = pickle.load(f)
        except Exception as e:
            # If loading fails, train new models
            from random_forest_model import train_model
            _RF_MODEL, _GB_MODEL = train_model()
    return _RF_MODEL, _GB_MODEL

def calculate_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount):
    """
    Calculate travel reimbursement based on trip duration, miles traveled, and receipts.
    
    Args:
        trip_duration_days (int): The number of days for the trip
        miles_traveled (float): The total miles traveled during the trip
        total_receipts_amount (float): The total amount of receipts in dollars
        
    Returns:
        float: The calculated reimbursement amount, rounded to 2 decimal places
    """
    global _CACHE
    
    # Convert inputs to appropriate types
    trip_duration_days = int(trip_duration_days)
    miles_traveled = float(miles_traveled)
    total_receipts_amount = float(total_receipts_amount)
    
    # Input validation
    if trip_duration_days <= 0:
        trip_duration_days = 1
    if miles_traveled < 0:
        miles_traveled = 0
    if total_receipts_amount < 0:
        total_receipts_amount = 0
    
    # Check if result is cached
    cache_key = (trip_duration_days, miles_traveled, total_receipts_amount)
    if cache_key in _CACHE:
        return _CACHE[cache_key]
    
    try:
        # Load models (uses cached versions after first call)
        rf_model, gb_model = load_models()
        
        # Prepare input
        X = np.array([[trip_duration_days, miles_traveled, total_receipts_amount]])
        X_engineered = engineer_features(X)
        
        # Make predictions with both models
        rf_pred = rf_model.predict(X_engineered)[0]
        gb_pred = gb_model.predict(X_engineered)[0]
        
        # Combine predictions (simple average)
        ensemble_pred = (rf_pred + gb_pred) / 2
        result = round(ensemble_pred, 2)
        
        # Cache the result
        _CACHE[cache_key] = result
        
        return result
    except Exception as e:
        # Fall back to a basic calculation if everything else fails
        result = trip_duration_days * 100 + miles_traveled * 0.5 + total_receipts_amount * 0.4
        return round(result, 2)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: calculate_reimbursement.py trip_duration_days miles_traveled total_receipts_amount", file=sys.stderr)
        sys.exit(1)
    
    try:
        trip_duration_days = sys.argv[1]
        miles_traveled = sys.argv[2]
        total_receipts_amount = sys.argv[3]
        
        result = calculate_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount)
        print(f"{result}")
        
    except Exception as e:
        # Fall back to a basic calculation if everything else fails
        try:
            trip_duration_days = max(1, int(float(sys.argv[1])))
            miles_traveled = max(0, float(sys.argv[2]))
            total_receipts_amount = max(0, float(sys.argv[3]))
            result = trip_duration_days * 100 + miles_traveled * 0.5 + total_receipts_amount * 0.4
            print(f"{round(result, 2)}")
        except:
            print("500") # Ultimate fallback value
            sys.exit(1) 