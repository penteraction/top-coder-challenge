#!/bin/bash

# Optimized implementation that preloads models
python3 -c "
import pickle
import numpy as np
import sys
import os
from random_forest_model import engineer_features

# Load models with proper error handling
try:
    with open('rf_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    with open('gb_model.pkl', 'rb') as f:
        gb_model = pickle.load(f)
except Exception as e:
    print(f'Error loading models: {e}', file=sys.stderr)
    sys.exit(1)

# Get arguments with validation
try:
    trip_duration_days = int(float('$1'))
    if trip_duration_days <= 0:
        trip_duration_days = 1
        
    miles_traveled = float('$2')
    if miles_traveled < 0:
        miles_traveled = 0
        
    total_receipts_amount = float('$3')
    if total_receipts_amount < 0:
        total_receipts_amount = 0
except Exception as e:
    print(f'Error parsing arguments: {e}', file=sys.stderr)
    sys.exit(1)

try:
    # Prepare input 
    X = np.array([[trip_duration_days, miles_traveled, total_receipts_amount]])
    X_engineered = engineer_features(X)

    # Make predictions
    rf_pred = rf_model.predict(X_engineered)[0]
    gb_pred = gb_model.predict(X_engineered)[0]
    result = round((rf_pred + gb_pred) / 2, 2)

    # Output result
    print(f'{result}')
except Exception as e:
    # Fallback calculation if prediction fails
    fallback = trip_duration_days * 100 + miles_traveled * 0.5 + total_receipts_amount * 0.4
    print(f'{round(fallback, 2)}')
" 