import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def get_feature_pipeline():
    """
    Creates and returns a scikit-learn feature engineering pipeline.
    """
    # Define columns by type
    numeric_features = [
        'Age', 'Number of Dependents', 'Latitude', 'Longitude', 'Population',
        'Number of Referrals', 'Tenure in Months', 'Avg Monthly Long Distance Charges',
        'Avg Monthly GB Download', 'Monthly Charge', 'Total Charges', 'Total Refunds',
        'Total Extra Data Charges', 'Total Long Distance Charges', 'Total Revenue'
    ]
    
    categorical_features = [
        'Gender', 'Under 30', 'Senior Citizen', 'Married', 'Dependents', 'City',
        'Referred a Friend', 'Offer', 'Phone Service', 'Multiple Lines', 'Internet Service',
        'Internet Type', 'Online Security', 'Online Backup', 'Device Protection Plan',
        'Premium Tech Support', 'Streaming TV', 'Streaming Movies', 'Streaming Music',
        'Unlimited Data', 'Contract', 'Paperless Billing', 'Payment Method'
    ]
    
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop' # Drop columns not explicitly specified (like Customer ID, Zip Code, etc.)
    )
    
    return preprocessor

def get_train_test_split(df, target_col='NPS_Category'):
    """
    Splits the data simulating the 15% respondent vs 85% silent base problem.
    The training set will be 15%, and test set will be 85%.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Train is 15%, test is 85%, stratified by the NPS category
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.15, stratify=y, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    from data_preparation import load_and_prepare_data
    df = load_and_prepare_data('telco_11_1_3.csv')
    X_train, X_test, y_train, y_test = get_train_test_split(df)
    
    print(f"Train size (respondents): {len(X_train)} (15%)")
    print(f"Test size (silent base): {len(X_test)} (85%)")
    
    preprocessor = get_feature_pipeline()
    X_train_transformed = preprocessor.fit_transform(X_train)
    
    print(f"Number of engineered features: {X_train_transformed.shape[1]}")
