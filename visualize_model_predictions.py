import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Ensure project modules can be imported
base_dir = os.path.dirname(os.path.abspath(__file__))
for folder in ['preprocessing', 'model', 'utils']:
    folder_path = os.path.join(base_dir, folder)
    if os.path.exists(folder_path) and folder_path not in sys.path:
        sys.path.append(folder_path)
if base_dir not in sys.path:
    sys.path.append(base_dir)

from preprocessing import load_data

def generate_actual_vs_predicted_plot(output_filename="actual_vs_predicted_rul.png"):
    print("Loading telemetry dataset...")
    df = load_data()
    df.ffill(inplace=True)
    
    feature_cols = ["Temperature", "Vibration", "Motor_Current"]
    target_col = "Remaining_Useful_Life_Days"
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Use identical train/test split as training (80/20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Load trained models & scaler
    models_dir = os.path.join(base_dir, "models")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    rf_path = os.path.join(models_dir, "rf_rul_model.pkl")
    xgb_path = os.path.join(models_dir, "xgb_rul_model.pkl")
    
    if not os.path.exists(scaler_path) or not os.path.exists(rf_path) or not os.path.exists(xgb_path):
        raise FileNotFoundError("Model artifacts not found in models/ directory. Please run model/train_model.py first.")
        
    print("Loading model artifacts...")
    scaler = joblib.load(scaler_path)
    rf_model = joblib.load(rf_path)
    xgb_model = joblib.load(xgb_path)
    
    X_test_scaled = scaler.transform(X_test)
    
    print("Generating predictions on test set...")
    rf_preds = rf_model.predict(X_test_scaled)
    xgb_preds = xgb_model.predict(X_test_scaled)
    
    # Calculate evaluation metrics
    rf_r2 = r2_score(y_test, rf_preds)
    rf_mae = mean_absolute_error(y_test, rf_preds)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    
    xgb_r2 = r2_score(y_test, xgb_preds)
    xgb_mae = mean_absolute_error(y_test, xgb_preds)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
    
    # Set up matplotlib plotting style
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True, dpi=300)
    
    # Min/Max boundaries for diagonal line
    min_val = min(y_test.min(), min(rf_preds.min(), xgb_preds.min())) - 5
    max_val = max(y_test.max(), max(rf_preds.max(), xgb_preds.max())) + 5
    
    # --- Subplot 1: Random Forest Regressor ---
    ax1 = axes[0]
    ax1.scatter(
        y_test, 
        rf_preds, 
        color="#2563eb", 
        alpha=0.55, 
        edgecolors="none", 
        s=35, 
        label="RF Predictions"
    )
    ax1.plot(
        [min_val, max_val], 
        [min_val, max_val], 
        color="#dc2626", 
        linestyle="--", 
        linewidth=2, 
        label="Perfect Prediction (y = x)"
    )
    
    rf_metrics_str = f"$R^2$: {rf_r2:.4f}\nMAE: {rf_mae:.2f} days\nRMSE: {rf_rmse:.2f} days"
    ax1.text(
        0.05, 0.92, 
        rf_metrics_str, 
        transform=ax1.transAxes, 
        fontsize=11, 
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#cbd5e1', alpha=0.9)
    )
    
    ax1.set_title("Random Forest Regressor (Bagging Ensemble)", fontsize=14, fontweight="bold", pad=12)
    ax1.set_xlabel("Actual Remaining Useful Life (Days)", fontsize=12, fontweight="medium")
    ax1.set_ylabel("Predicted Remaining Useful Life (Days)", fontsize=12, fontweight="medium")
    ax1.set_xlim(min_val, max_val)
    ax1.set_ylim(min_val, max_val)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", frameon=True, fontsize=10)
    
    # --- Subplot 2: XGBoost Regressor ---
    ax2 = axes[1]
    ax2.scatter(
        y_test, 
        xgb_preds, 
        color="#059669", 
        alpha=0.55, 
        edgecolors="none", 
        s=35, 
        label="XGBoost Predictions"
    )
    ax2.plot(
        [min_val, max_val], 
        [min_val, max_val], 
        color="#dc2626", 
        linestyle="--", 
        linewidth=2, 
        label="Perfect Prediction (y = x)"
    )
    
    xgb_metrics_str = f"$R^2$: {xgb_r2:.4f}\nMAE: {xgb_mae:.2f} days\nRMSE: {xgb_rmse:.2f} days"
    ax2.text(
        0.05, 0.92, 
        xgb_metrics_str, 
        transform=ax2.transAxes, 
        fontsize=11, 
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#cbd5e1', alpha=0.9)
    )
    
    ax2.set_title("XGBoost Regressor (Gradient Boosted Trees)", fontsize=14, fontweight="bold", pad=12)
    ax2.set_xlabel("Actual Remaining Useful Life (Days)", fontsize=12, fontweight="medium")
    ax2.set_xlim(min_val, max_val)
    ax2.set_ylim(min_val, max_val)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right", frameon=True, fontsize=10)
    
    # Main Figure Title
    fig.suptitle("Actual RUL vs. Predicted RUL: Random Forest vs. XGBoost Benchmark", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save output plot
    output_path = os.path.join(base_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[SUCCESS] Scatter plot visualization saved to: {output_path}")
    print(f"Random Forest Metrics -> R2: {rf_r2:.4f}, MAE: {rf_mae:.2f} days, RMSE: {rf_rmse:.2f} days")
    print(f"XGBoost Metrics       -> R2: {xgb_r2:.4f}, MAE: {xgb_mae:.2f} days, RMSE: {xgb_rmse:.2f} days")

if __name__ == "__main__":
    generate_actual_vs_predicted_plot()
