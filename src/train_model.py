import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import joblib

from data_preparation import load_and_prepare_data
from feature_engineering import get_feature_pipeline, get_train_test_split

def train_and_evaluate_models():
    print("Loading and preparing data...")
    df = load_and_prepare_data('telco_11_1_3.csv')
    X_train, X_test, y_train, y_test = get_train_test_split(df)
    
    print("Engineering features...")
    preprocessor = get_feature_pipeline()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Save the preprocessor
    joblib.dump(preprocessor, 'preprocessor.joblib')
    
    # Define mapping for XGBoost (needs numeric labels)
    class_mapping = {'Detractor': 0, 'Passive': 1, 'Promoter': 2}
    y_train_num = y_train.map(class_mapping)
    y_test_num = y_test.map(class_mapping)
    
    # 1. Baseline Model: Logistic Regression (handles multi-class automatically)
    print("\n--- Training Baseline (Logistic Regression) ---")
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(X_train_transformed, y_train)
    lr_preds = lr.predict(X_test_transformed)
    
    print("Baseline Classification Report:")
    print(classification_report(y_test, lr_preds))
    print(f"Baseline Macro-F1: {f1_score(y_test, lr_preds, average='macro'):.4f}")
    
    # 2. Advanced Model: XGBoost
    print("\n--- Training XGBoost ---")
    # compute class weights since we have imbalance
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train_num)
    weights = compute_class_weight('balanced', classes=classes, y=y_train_num)
    weight_dict = dict(zip(classes, weights))
    
    # xgb expects sample weights
    sample_weights = y_train_num.map(weight_dict)
    
    xgb_model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        random_state=42,
        eval_metric='mlogloss',
        use_label_encoder=False
    )
    
    xgb_model.fit(X_train_transformed, y_train_num, sample_weight=sample_weights)
    
    xgb_preds_num = xgb_model.predict(X_test_transformed)
    
    # Reverse mapping
    inv_mapping = {v: k for k, v in class_mapping.items()}
    xgb_preds = [inv_mapping[p] for p in xgb_preds_num]
    
    print("XGBoost Classification Report:")
    print(classification_report(y_test, xgb_preds))
    print(f"XGBoost Macro-F1: {f1_score(y_test, xgb_preds, average='macro'):.4f}")
    
    # Save the best model
    joblib.dump(xgb_model, 'xgb_model.joblib')
    print("\nModels saved: 'preprocessor.joblib', 'xgb_model.joblib'")

if __name__ == "__main__":
    train_and_evaluate_models()
