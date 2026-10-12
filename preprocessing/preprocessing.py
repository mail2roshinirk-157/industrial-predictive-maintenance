import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from config import DATASET_PATH, FEATURE_COLUMNS, TARGET_COLUMN, MODELS_DIR

def load_data(filepath=None):
    if filepath is None:
        filepath = DATASET_PATH
    if not os.path.exists(filepath):
        alt_path = os.path.join(base_dir, filepath)
        if os.path.exists(alt_path):
            filepath = alt_path
        else:
            filepath = DATASET_PATH
    return pd.DataFrame(pd.read_csv(filepath))

def preprocess_data(df, is_training=True, feature_cols=None):
    # Missing value handling (Forward Fill)
    df.ffill(inplace=True)
    
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS
    
    # Resolve model directory
    os.makedirs(MODELS_DIR, exist_ok=True)
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    
    if is_training:
        target_col = TARGET_COLUMN
        X = df[feature_cols]
        y = df[target_col]
        
        # Train/Test Split (identical row-based 80/20 split)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Feature Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save the scaler for inference if using default 10 features
        if feature_cols == FEATURE_COLUMNS:
            joblib.dump(scaler, scaler_path)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    else:
        # Inference mode
        scaler = joblib.load(scaler_path)
        X = df[feature_cols]
        X_scaled = scaler.transform(X)
        return X_scaled
