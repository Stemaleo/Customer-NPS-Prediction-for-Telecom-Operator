import pandas as pd
import numpy as np

def load_and_prepare_data(file_path):
    """
    Loads the telco churn dataset, maps NPS targets, and cleans data.
    """
    df = pd.read_csv(file_path)
    
    # Target Construction: Mapping Satisfaction Score to NPS
    # 5 -> Promoter
    # 4 -> Passive
    # <=3 -> Detractor
    def map_nps(score):
        if score == 5:
            return 'Promoter'
        elif score == 4:
            return 'Passive'
        elif score <= 3:
            return 'Detractor'
        return np.nan
        
    df['NPS_Category'] = df['Satisfaction Score'].apply(map_nps)
    
    # Remove rows where Satisfaction Score is missing (since we need it for training/eval)
    df = df.dropna(subset=['NPS_Category'])
    
    # Handle missing Total Charges (usually for customers with 0 tenure)
    # We'll fill them with 0 or the monthly charge
    df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
    df['Total Charges'] = df['Total Charges'].fillna(0)
    
    # Drop leakage columns: columns that are known only after a customer churns
    # Churn Label, Churn Score, CLTV (maybe keep CLTV? Let's drop to be safe or keep if it's predictive, but instructions say Churn Score and Churn Value leak)
    # In instructions: "the dataset also contains a Churn Score and a Churn Value. Used naively, these will leak into the target"
    # Actually, Churn Category and Churn Reason are also leakage.
    leakage_cols = ['Churn Label', 'Churn Score', 'Churn Category', 'Churn Reason']
    
    # Check if Churn Value is present, drop it
    if 'Churn Value' in df.columns:
        leakage_cols.append('Churn Value')
        
    df = df.drop(columns=[col for col in leakage_cols if col in df.columns])
    
    return df

if __name__ == "__main__":
    df = load_and_prepare_data('telco_11_1_3.csv')
    print(f"Dataset shape after preparation: {df.shape}")
    print("NPS Category Distribution:")
    print(df['NPS_Category'].value_counts(normalize=True))
