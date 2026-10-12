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
    console.warn('Model metrics endpoint unavailable or timed out:', err.message);
    return null;
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

