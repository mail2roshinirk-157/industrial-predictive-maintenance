# Industrial Predictive Maintenance (Phase 1)

This is a proof-of-concept AI-based Predictive Maintenance System. It simulates industrial machine degradation, trains a Machine Learning model (Random Forest Regressor) to predict the Remaining Useful Life (RUL) of the machine, and visualizes the predictions on a real-time Streamlit dashboard.

## Project Structure
- `simulator.py`: Contains the logic for generating realistic synthetic sensor degradation across 4 stages.
- `dataset_generator.py`: Generates `datasets/sensor_data.csv` using the simulator.
- `preprocessing.py`: Handles missing values and feature scaling.
- `train_model.py`: Trains the Random Forest Regressor and outputs evaluation metrics (MAE, RMSE, R²).
- `predict.py`: The inference engine that loads the model and predicts RUL based on live sensor data. It also contains logic to extrapolate future sensor trends.
- `dashboard.py`: A Streamlit application that acts as the industrial monitoring interface.
- `utils.py`: Helper functions for UI styling.

## Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate the Dataset
If you wish to re-generate the synthetic run-to-failure data:
```bash
python dataset_generator.py
```
*Note: This will create `datasets/sensor_data.csv`.*

### 3. Train the Model
```bash
python train_model.py
```
*Note: This will output metrics to the console and save the model to `models/rf_rul_model.pkl` and `models/scaler.pkl`.*

### 4. Run the Dashboard
```bash
streamlit run dashboard.py
```
Open your browser to the URL provided (usually `http://localhost:8501`). You can check the "Auto-Play Live Stream" box in the sidebar to simulate a live data feed from an industrial machine.
