import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score

from data_preparation import load_and_prepare_data
from feature_engineering import get_train_test_split

def evaluate_fairness():
    print("Loading data and models...")
    df = load_and_prepare_data('telco_11_1_3.csv')
    X_train, X_test, y_train, y_test = get_train_test_split(df)
    
    preprocessor = joblib.load('preprocessor.joblib')
    xgb_model = joblib.load('xgb_model.joblib')
    
    X_test_transformed = preprocessor.transform(X_test)
    
    # Define mapping
    class_mapping = {'Detractor': 0, 'Passive': 1, 'Promoter': 2}
    inv_mapping = {0: 'Detractor', 1: 'Passive', 2: 'Promoter'}
    
    xgb_preds_num = xgb_model.predict(X_test_transformed)
    xgb_preds = [inv_mapping[p] for p in xgb_preds_num]
    
    # Calculate fairness on Senior Citizen
    print("\n--- Fairness Audit ---")
    print("Checking recall for Detractor class across 'Senior Citizen' feature")
    
    # We want to see how well we recall Detractors for Senior Citizens vs Non-Senior
    for senior_val in ['Yes', 'No']:
        mask = X_test['Senior Citizen'] == senior_val
        y_test_subset = y_test[mask]
        preds_subset = np.array(xgb_preds)[mask]
        
        # We calculate recall for Detractors manually
        true_detractors = sum(y_test_subset == 'Detractor')
        if true_detractors == 0:
            continue
            
        correct_detractors = sum((y_test_subset == 'Detractor') & (preds_subset == 'Detractor'))
        recall = correct_detractors / true_detractors
        print(f"Recall for Detractors (Senior Citizen = {senior_val}): {recall:.4f} (Count: {true_detractors})")
        
    print("\nChecking recall for Detractor class across 'Gender' feature")
    for gender_val in ['Male', 'Female']:
        mask = X_test['Gender'] == gender_val
        y_test_subset = y_test[mask]
        preds_subset = np.array(xgb_preds)[mask]
        
        true_detractors = sum(y_test_subset == 'Detractor')
        if true_detractors == 0:
            continue
            
        correct_detractors = sum((y_test_subset == 'Detractor') & (preds_subset == 'Detractor'))
        recall = correct_detractors / true_detractors
        print(f"Recall for Detractors (Gender = {gender_val}): {recall:.4f} (Count: {true_detractors})")

if __name__ == "__main__":
    evaluate_fairness()
