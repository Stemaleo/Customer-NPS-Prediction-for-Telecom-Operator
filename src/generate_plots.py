import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from data_preparation import load_and_prepare_data
from feature_engineering import get_train_test_split

def generate_plots():
    print("Loading data and models...")
    df = load_and_prepare_data('telco_11_1_3.csv')
    X_train, X_test, y_train, y_test = get_train_test_split(df)
    
    preprocessor = joblib.load('preprocessor.joblib')
    xgb_model = joblib.load('xgb_model.joblib')
    
    artifact_dir = r'C:\Users\Utilisateur\.gemini\antigravity-ide\brain\da9a8a4d-832e-4da0-819e-44677c2aeca3\\'
    
    # Target Distribution Plot
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='NPS_Category', order=['Detractor', 'Passive', 'Promoter'], hue='NPS_Category', palette='Set2', legend=False)
    plt.title('NPS Category Distribution (Imbalanced Target)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(artifact_dir + 'nps_distribution.png')
    plt.close()
    
    # Confusion Matrix
    X_test_transformed = preprocessor.transform(X_test)
    class_mapping = {0: 'Detractor', 1: 'Passive', 2: 'Promoter'}
    xgb_preds_num = xgb_model.predict(X_test_transformed)
    xgb_preds = [class_mapping[p] for p in xgb_preds_num]
    
    cm = confusion_matrix(y_test, xgb_preds, labels=['Detractor', 'Passive', 'Promoter'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Detractor', 'Passive', 'Promoter'])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues')
    plt.title('XGBoost Confusion Matrix')
    plt.tight_layout()
    plt.savefig(artifact_dir + 'confusion_matrix.png')
    plt.close()
    
    # SHAP Feature Importance
    print("Generating SHAP plots (this may take a moment)...")
    explainer = shap.TreeExplainer(xgb_model)
    X_test_transformed_sample = X_test_transformed[:500] 
    shap_values = explainer.shap_values(X_test_transformed_sample)
    
    feature_names = preprocessor.get_feature_names_out()
    feature_names = [f.split('__')[-1] for f in feature_names]
    
    # SHAP summary plot for Detractor class
    # If shap_values is a list (older SHAP or some versions of XGBoost)
    if isinstance(shap_values, list):
        shap_data = shap_values[0]
    else:
        # Multiclass output is (n_samples, n_features, n_classes)
        shap_data = shap_values[:, :, 0]
        
    shap.summary_plot(shap_data, X_test_transformed_sample, feature_names=feature_names, show=False)
    plt.title('SHAP Summary Plot: Drivers for Detractor Class')
    plt.tight_layout()
    plt.savefig(artifact_dir + 'shap_summary_detractor.png')
    plt.close()

    
    print("All plots generated successfully: nps_distribution.png, confusion_matrix.png, shap_summary_detractor.png")

if __name__ == "__main__":
    generate_plots()
