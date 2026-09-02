# Industrial Predictive Maintenance (Phase 1)

This is an AI-based Industrial Predictive Maintenance System. It simulates machine degradation, trains and benchmarks two ensemble Machine Learning models (**Random Forest Regressor** and **XGBoost Regressor**) to predict Remaining Useful Life (RUL), and visualizes predictions on a real-time Streamlit SCADA dashboard.

## Project Structure
- `simulator/simulator.py`: Logic for generating realistic synthetic sensor degradation across 4 lifecycle stages.
- `datasets/dataset_generator.py`: Generates `datasets/sensor_data.csv` using the simulator.
- `preprocessing/preprocessing.py`: Handles missing values and feature scaling using `StandardScaler`.
- `model/train_model.py`: Trains both Random Forest and XGBoost Regressors on an identical train/test split, evaluates metrics (MAE, RMSE, R²), and outputs a side-by-side comparison table with performance analysis.
- `model/predict.py`: Inference engine supporting both Random Forest and XGBoost models, temporal smoothing filters, and future trend extrapolation.
- `model/maintenance_engine.py`: Maintenance decision engine translating RUL predictions and machine health into actionable servicing policies.
- `dashboard.py`: Interactive Streamlit SCADA dashboard allowing model selection (Random Forest, XGBoost, or Side-by-Side Comparison), real-time sensor streams, and diagnostic metrics.
- `utils/utils.py`: Helper functions for status colors and markdown formatting.

## Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate the Dataset (Optional)
If you wish to re-generate the synthetic run-to-failure data:
```bash
python datasets/dataset_generator.py
```
*Note: This will create `datasets/sensor_data.csv`.*

### 3. Train & Benchmark the Models
```bash
python model/train_model.py
```
*Note: This will train both Random Forest and XGBoost models on the same 80/20 train/test split, print a benchmark comparison table to the console, and save `models/rf_rul_model.pkl`, `models/xgb_rul_model.pkl`, and `models/scaler.pkl`.*

### 4. Run the Streamlit Dashboard
```bash
streamlit run dashboard.py
```
Open your browser to the URL provided (usually `http://localhost:8501`). You can use the sidebar radio buttons to switch between **Random Forest Regressor**, **XGBoost Regressor**, or **Side-by-Side Comparison (Both)**.
