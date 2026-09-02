import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
  timeout: 15000,
});

export const getStatus = async () => {
  const res = await api.get('/status');
  return res.data;
};

export const getCurrentData = async () => {
  const res = await api.get('/current');
  return res.data;
};

export const getHistoryData = async () => {
  const res = await api.get('/history');
  return res.data;
};

export const getModelMetrics = async () => {
  try {
    const res = await api.get('/model', { timeout: 15000 });
    return res.data;
  } catch (err) {
    console.warn('Model metrics endpoint unavailable or timed out, returning fallback metrics:', err.message);
    return {
      isFallback: true,
      algorithm: 'XGBoost Regressor',
      mae: 28.30,
      rmse: 37.67,
      r2_score: 0.8757,
      dataset_size: 91250,
      train_size: 73000,
      test_size: 18250,
      feature_importance: [
        { feature: 'Temperature', importance: 0.48 },
        { feature: 'Vibration', importance: 0.35 },
        { feature: 'Motor_Current', importance: 0.17 }
      ],
      scatter_plot: { actual: [220, 180, 140, 90, 40], predicted: [215, 178, 142, 88, 39] },
      residuals: [5, 2, -2, 2, 1],
      models: {
        random_forest: {
          algorithm: 'Random Forest Regressor',
          train_r2: 0.9705,
          test_r2: 0.8571,
          mae: 29.17,
          rmse: 40.39
        },
        xgboost: {
          algorithm: 'XGBoost Regressor',
          train_r2: 0.8827,
          test_r2: 0.8757,
          mae: 28.30,
          rmse: 37.67
        }
      }
    };
  }
};



export const getRecentLogs = async () => {
  const res = await api.get('/logs');
  return res.data;
};

export const getMaintenanceData = async () => {
  const res = await api.get('/maintenance');
  return res.data;
};

export const postControlAction = async (action, speed = 1.0) => {
  const res = await api.post('/control', { action, speed });
  return res.data;
};
