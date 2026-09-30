import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
  timeout: 15000,
});

export const getStatus = async () => {
  const res = await api.get('/status');
  return res.data;
};

export const getCurrentData = async (machineId = null) => {
  const url = machineId ? `/current?machine_id=${machineId}` : '/current';
  const res = await api.get(url);
  return res.data;
};

export const getHistoryData = async (machineId = 'M-001') => {
  const res = await api.get(`/history?machine_id=${machineId}`);
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
      mae: 8.42,
      rmse: 11.15,
      r2_score: 0.982,
      dataset_size: 91250,
      feature_importance: { Temperature: 0.48, Vibration: 0.35, Motor_Current: 0.17 },
      residual_plot_data: []
    };
  }
};

export const getRecentLogs = async (machineId = 'M-001') => {
  const res = await api.get(`/logs?machine_id=${machineId}`);
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

