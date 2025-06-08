#!/usr/bin/env python3

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import pickle
import sys
import datetime

def engineer_features(X):
    """Create advanced features based on domain knowledge from interviews and analysis"""
    # Extract original features
    trip_duration = X[:, 0]
    miles = X[:, 1]
    receipts = X[:, 2]
    
    # Efficiency metrics (mentioned by Kevin as highly important)
    miles_per_day = miles / np.maximum(trip_duration, 1)
    receipts_per_day = receipts / np.maximum(trip_duration, 1)
    receipts_per_mile = receipts / np.maximum(miles, 1)
    
    # Special 5-day trip features (mentioned by Lisa and others as getting bonuses)
    is_5_day_trip = (trip_duration == 5).astype(float)
    
    # Specific duration categories based on interview insights
    is_1_to_2_day = ((trip_duration >= 1) & (trip_duration <= 2)).astype(float)
    is_3_to_4_day = ((trip_duration >= 3) & (trip_duration <= 4)).astype(float)
    is_6_to_7_day = ((trip_duration >= 6) & (trip_duration <= 7)).astype(float)
    is_8_plus_day = (trip_duration >= 8).astype(float)
    
    # Efficiency bonus zones based on Kevin's insights
    low_efficiency = ((miles_per_day > 0) & (miles_per_day < 150)).astype(float)
    optimal_efficiency = ((miles_per_day >= 150) & (miles_per_day <= 250)).astype(float)
    high_efficiency = (miles_per_day > 250).astype(float)
    
    # Receipt amount categories based on interview insights
    low_receipts_per_day = (receipts_per_day < 75).astype(float)
    medium_receipts_per_day = ((receipts_per_day >= 75) & (receipts_per_day <= 120)).astype(float)
    high_receipts_per_day = (receipts_per_day > 120).astype(float)
    
    # "Sweet spot combo" (Kevin's insight)
    is_5_day_trip_array = (trip_duration == 5)
    high_miles_per_day = (miles_per_day >= 180)
    low_spend_per_day = (receipts_per_day < 100)
    sweet_spot_combo = np.logical_and(np.logical_and(is_5_day_trip_array, high_miles_per_day), low_spend_per_day).astype(float)
    
    # "Vacation penalty" (Kevin's insight)
    is_8_plus_day_array = (trip_duration >= 8)
    high_receipts_per_day_array = (receipts_per_day > 120)
    vacation_penalty = np.logical_and(is_8_plus_day_array, high_receipts_per_day_array).astype(float)
    
    # Tiered mileage calculation
    miles_0_to_100 = np.minimum(miles, 100)
    miles_100_to_300 = np.maximum(0, np.minimum(miles - 100, 200))
    miles_300_to_700 = np.maximum(0, np.minimum(miles - 300, 400))
    miles_over_700 = np.maximum(0, miles - 700)
    
    # Receipt tiers based on Lisa's insights
    receipts_0_to_300 = np.minimum(receipts, 300)
    receipts_300_to_800 = np.maximum(0, np.minimum(receipts - 300, 500))
    receipts_800_to_1500 = np.maximum(0, np.minimum(receipts - 800, 700))
    receipts_over_1500 = np.maximum(0, receipts - 1500)
    
    # Transformations to capture non-linear relationships
    sqrt_miles = np.sqrt(miles)
    log_miles = np.log1p(miles)
    
    # Interaction terms - Kevin mentioned these are critical
    duration_miles = trip_duration * miles
    duration_receipts = trip_duration * receipts
    miles_receipts = miles * receipts
    
    # Special case - 5-day trip interactions
    five_day_miles = is_5_day_trip * miles
    five_day_receipts = is_5_day_trip * receipts
    
    # Most important features based on previous model analysis
    return np.column_stack([
        # Original features
        trip_duration, miles, receipts,
        
        # Trip duration categories
        is_1_to_2_day,
        is_3_to_4_day,
        is_5_day_trip,
        is_6_to_7_day,
        is_8_plus_day,
        
        # Efficiency metrics
        miles_per_day,
        receipts_per_day,
        receipts_per_mile,
        
        # Efficiency categories
        low_efficiency,
        optimal_efficiency,
        high_efficiency,
        
        # Receipt amount categories
        low_receipts_per_day,
        medium_receipts_per_day,
        high_receipts_per_day,
        
        # Special combinations
        sweet_spot_combo,
        vacation_penalty,
        
        # Tiered calculations
        miles_0_to_100,
        miles_100_to_300,
        miles_300_to_700,
        miles_over_700,
        receipts_0_to_300,
        receipts_300_to_800,
        receipts_800_to_1500,
        receipts_over_1500,
        
        # Non-linear transformations
        sqrt_miles,
        log_miles,
        
        # Interaction terms
        duration_miles,
        duration_receipts,
        miles_receipts,
        
        # Special 5-day interactions
        five_day_miles,
        five_day_receipts
    ])

def identify_trip_clusters(X, y, n_clusters=6):
    """Identify natural clusters in the data as mentioned by Kevin"""
    # Extract basic features
    trip_duration = X[:, 0]
    miles = X[:, 1]
    receipts = X[:, 2]
    
    # Create features for clustering
    miles_per_day = miles / np.maximum(trip_duration, 1)
    receipts_per_day = receipts / np.maximum(trip_duration, 1)
    
    # Features for clustering
    cluster_features = np.column_stack([
        trip_duration,
        miles,
        receipts,
        miles_per_day,
        receipts_per_day
    ])
    
    # Normalize features for clustering
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(cluster_features)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_features)
    
    return clusters

def load_data():
    """Load and preprocess data"""
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    X = []
    y = []
    trip_durations = []  # Store trip durations for stratified sampling
    
    for case in data:
        input_data = case['input']
        trip_duration = input_data['trip_duration_days']
        X.append([
            trip_duration,
            input_data['miles_traveled'],
            input_data['total_receipts_amount']
        ])
        y.append(case['expected_output'])
        trip_durations.append(trip_duration)
    
    X = np.array(X)
    y = np.array(y)
    
    # Identify clusters in the data
    clusters = identify_trip_clusters(X, y)
    
    # Create duration categories for stratified sampling
    # Ensure we have balanced representation of all trip durations
    duration_categories = pd.cut(
        trip_durations, 
        bins=[0, 1, 2, 3, 4, 5, 6, 7, 10, 14, 100], 
        labels=False
    )
    
    # Engineer features
    X_engineered = engineer_features(X)
    
    # Split data for validation - use stratified sampling to ensure all trip durations are represented
    X_train, X_val, y_train, y_val, cats_train, cats_val, clusters_train, clusters_val = train_test_split(
        X_engineered, y, duration_categories, clusters, 
        test_size=0.15, random_state=42, stratify=duration_categories
    )
    
    return X_engineered, y, X_train, X_val, y_train, y_val, duration_categories, X, clusters

def train_model():
    """Train the model with insights from interviews"""
    X_engineered, y, X_train, X_val, y_train, y_val, duration_cats, X_raw, clusters = load_data()
    
    # Train a Random Forest model
    rf_model = RandomForestRegressor(
        n_estimators=800,      # More trees for better stability
        max_depth=10,          # Limited depth to prevent overfitting
        min_samples_split=12,  # Larger splits for better generalization
        min_samples_leaf=6,    # Larger leaves for better generalization
        max_features=0.4,      # Use 40% of features for more diversity in trees
        bootstrap=True,
        oob_score=True,
        max_samples=0.7,       # Use 70% of samples per tree for diversity
        n_jobs=-1,
        random_state=42
    )
    
    # Also train a Gradient Boosting model to capture more complex patterns
    gb_model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    
    print("Training Random Forest model...")
    rf_model.fit(X_engineered, y)
    
    print("Training Gradient Boosting model...")
    gb_model.fit(X_engineered, y)
    
    # Save models
    with open('rf_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    
    with open('gb_model.pkl', 'wb') as f:
        pickle.dump(gb_model, f)
    
    print("Models trained and saved")
    
    # Make predictions with both models
    rf_preds = rf_model.predict(X_engineered)
    gb_preds = gb_model.predict(X_engineered)
    
    # Combine predictions (simple average)
    ensemble_preds = (rf_preds + gb_preds) / 2
    
    # Calculate errors
    rf_errors = np.abs(rf_preds - y)
    gb_errors = np.abs(gb_preds - y)
    ensemble_errors = np.abs(ensemble_preds - y)
    
    # Validation on held-out data
    rf_val_preds = rf_model.predict(X_val)
    gb_val_preds = gb_model.predict(X_val)
    ensemble_val_preds = (rf_val_preds + gb_val_preds) / 2
    
    rf_val_errors = np.abs(rf_val_preds - y_val)
    gb_val_errors = np.abs(gb_val_preds - y_val)
    ensemble_val_errors = np.abs(ensemble_val_preds - y_val)
    
    # Print model performance
    print("\nRandom Forest Performance:")
    print(f"  Training Mean Absolute Error: ${np.mean(rf_errors):.2f}")
    print(f"  Validation Mean Absolute Error: ${np.mean(rf_val_errors):.2f}")
    print(f"  Out-of-bag score: {rf_model.oob_score_:.4f}")
    print(f"  Exact matches (±$0.01): {np.sum(rf_errors <= 0.01)}/{len(y)} ({np.sum(rf_errors <= 0.01)/len(y)*100:.2f}%)")
    print(f"  Close matches (±$1.00): {np.sum(rf_errors <= 1.00)}/{len(y)} ({np.sum(rf_errors <= 1.00)/len(y)*100:.2f}%)")
    
    print("\nGradient Boosting Performance:")
    print(f"  Training Mean Absolute Error: ${np.mean(gb_errors):.2f}")
    print(f"  Validation Mean Absolute Error: ${np.mean(gb_val_errors):.2f}")
    print(f"  Exact matches (±$0.01): {np.sum(gb_errors <= 0.01)}/{len(y)} ({np.sum(gb_errors <= 0.01)/len(y)*100:.2f}%)")
    print(f"  Close matches (±$1.00): {np.sum(gb_errors <= 1.00)}/{len(y)} ({np.sum(gb_errors <= 1.00)/len(y)*100:.2f}%)")
    
    print("\nEnsemble Performance:")
    print(f"  Training Mean Absolute Error: ${np.mean(ensemble_errors):.2f}")
    print(f"  Validation Mean Absolute Error: ${np.mean(ensemble_val_errors):.2f}")
    print(f"  Exact matches (±$0.01): {np.sum(ensemble_errors <= 0.01)}/{len(y)} ({np.sum(ensemble_errors <= 0.01)/len(y)*100:.2f}%)")
    print(f"  Close matches (±$1.00): {np.sum(ensemble_errors <= 1.00)}/{len(y)} ({np.sum(ensemble_errors <= 1.00)/len(y)*100:.2f}%)")
    
    # Analyze errors by trip duration
    print("\nError analysis by trip duration:")
    unique_durations = sorted(set(X_raw[:, 0]))
    for duration in unique_durations:
        mask = X_raw[:, 0] == duration
        if np.sum(mask) > 0:
            duration_errors = ensemble_errors[mask]
            print(f"  Days={int(duration)}: Mean=${np.mean(duration_errors):.2f}, Median=${np.median(duration_errors):.2f}, Count={np.sum(mask)}")
    
    # Focus on the 5-day trips
    five_day_mask = X_raw[:, 0] == 5
    five_day_errors = ensemble_errors[five_day_mask]
    five_day_X = X_raw[five_day_mask]
    five_day_y = y[five_day_mask]
    five_day_preds = ensemble_preds[five_day_mask]
    
    print("\n5-day trip analysis:")
    print(f"  Mean error: ${np.mean(five_day_errors):.2f}")
    print(f"  Median error: ${np.median(five_day_errors):.2f}")
    print(f"  Max error: ${np.max(five_day_errors):.2f}")
    
    # Analyze high error cases for 5-day trips
    high_error_indices = np.argsort(five_day_errors)[-5:]  # Top 5 highest errors
    print("\n  Top 5 highest error 5-day trips:")
    for idx in high_error_indices:
        miles = five_day_X[idx, 1]
        receipts = five_day_X[idx, 2]
        expected = five_day_y[idx]
        predicted = five_day_preds[idx]
        error = five_day_errors[idx]
        print(f"    Miles: {miles}, Receipts: ${receipts:.2f}, Expected: ${expected:.2f}, Predicted: ${predicted:.2f}, Error: ${error:.2f}")
    
    # Cluster analysis
    print("\nCluster analysis:")
    for cluster in range(max(clusters) + 1):
        mask = clusters == cluster
        if np.sum(mask) > 0:
            cluster_errors = ensemble_errors[mask]
            cluster_X = X_raw[mask]
            avg_duration = np.mean(cluster_X[:, 0])
            avg_miles = np.mean(cluster_X[:, 1])
            avg_receipts = np.mean(cluster_X[:, 2])
            print(f"  Cluster {cluster}:")
            print(f"    Avg Days: {avg_duration:.1f}, Avg Miles: {avg_miles:.1f}, Avg Receipts: ${avg_receipts:.2f}")
            print(f"    Mean Error: ${np.mean(cluster_errors):.2f}, Count: {np.sum(mask)}")
    
    # Feature importance analysis
    feature_names = [
        'trip_duration', 'miles', 'receipts',
        'is_1_to_2_day', 'is_3_to_4_day', 'is_5_day_trip', 'is_6_to_7_day', 'is_8_plus_day',
        'miles_per_day', 'receipts_per_day', 'receipts_per_mile',
        'low_efficiency', 'optimal_efficiency', 'high_efficiency',
        'low_receipts_per_day', 'medium_receipts_per_day', 'high_receipts_per_day',
        'sweet_spot_combo', 'vacation_penalty',
        'miles_0_to_100', 'miles_100_to_300', 'miles_300_to_700', 'miles_over_700',
        'receipts_0_to_300', 'receipts_300_to_800', 'receipts_800_to_1500', 'receipts_over_1500',
        'sqrt_miles', 'log_miles',
        'duration_miles', 'duration_receipts', 'miles_receipts'
    ]
    
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nTop Random Forest Feature Importance:")
    for i in range(min(20, len(feature_names))):  # Print top 20 features
        if indices[i] < len(feature_names):
            print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    
    return rf_model, gb_model

def predict(trip_duration_days, miles_traveled, total_receipts_amount):
    """Make prediction using the saved models"""
    # Load models
    try:
        with open('rf_model.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        with open('gb_model.pkl', 'rb') as f:
            gb_model = pickle.load(f)
    except FileNotFoundError:
        print("Models not found. Training new models...")
        rf_model, gb_model = train_model()
    
    # Prepare input
    X = np.array([[trip_duration_days, miles_traveled, total_receipts_amount]])
    X_engineered = engineer_features(X)
    
    # Make predictions with both models
    rf_pred = rf_model.predict(X_engineered)[0]
    gb_pred = gb_model.predict(X_engineered)[0]
    
    # Combine predictions (simple average)
    ensemble_pred = (rf_pred + gb_pred) / 2
    
    return round(ensemble_pred, 2)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Training mode
        print("Training models...")
        train_model()
        print("Done!")
        
    elif len(sys.argv) == 4:
        # Prediction mode
        try:
            trip_duration_days = int(sys.argv[1])
            miles_traveled = int(sys.argv[2])
            total_receipts_amount = float(sys.argv[3])
            
            result = predict(trip_duration_days, miles_traveled, total_receipts_amount)
            print(f"{result}")
            
        except ValueError:
            print("Error: Please provide valid numeric inputs")
            sys.exit(1)
    else:
        print("Usage:")
        print("  Training mode: python random_forest_model.py")
        print("  Prediction mode: python random_forest_model.py <trip_duration_days> <miles_traveled> <total_receipts_amount>")
        sys.exit(1) 