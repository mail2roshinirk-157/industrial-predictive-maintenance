import React, { useState, useEffect, useRef } from 'react';
import TopBar from './components/TopBar';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import Analytics from './pages/Analytics';
import Maintenance from './pages/Maintenance';
import Evaluation from './pages/Evaluation';
import {
  getStatus,
  getCurrentData,
  getHistoryData,
  getModelMetrics,
  getRecentLogs,
  getMaintenanceData,
  postControlAction,
} from './services/api';

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [selectedMachineId, setSelectedMachineId] = useState('M-001');
  const [backendStatus, setBackendStatus] = useState(true);
  const [statusData, setStatusData] = useState(null);
  const [currentData, setCurrentData] = useState(null);
  const [historyData, setHistoryData] = useState(null);
  const [modelData, setModelData] = useState(null);
  const [maintenanceData, setMaintenanceData] = useState(null);
  const [logs, setLogs] = useState([]);
  const [speed, setSpeed] = useState(1.0);
  const [autoPlay, setAutoPlay] = useState(false);

  const consecutiveFailuresRef = useRef(0);

  // Initial fetch for static model metrics and system status
  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        const sData = await getStatus();
        setStatusData(sData);
        setBackendStatus(true);
        consecutiveFailuresRef.current = 0;
      } catch {
        // Do not flag disconnected on single initial mount error
      }

      try {
        const mData = await getModelMetrics();
        setModelData(mData);
      } catch (err) {
        console.error('Failed to load model metrics:', err);
      }
    };

    fetchInitialData();
  }, []);

  // Primary 1000ms tick loop for live telemetry (/api/current)
  useEffect(() => {
    let isSubscribed = true;

    const pollLiveCurrent = async () => {
      try {
        const curr = await getCurrentData(selectedMachineId);
        if (isSubscribed) {
          setCurrentData(curr);
          setAutoPlay(curr.auto_play);
          setSpeed(curr.simulation_speed);
          consecutiveFailuresRef.current = 0;
          setBackendStatus(true);
        }
      } catch {
        if (isSubscribed) {
          consecutiveFailuresRef.current += 1;
          if (consecutiveFailuresRef.current >= 3) {
            setBackendStatus(false);
          }
        }
      }
    };

    pollLiveCurrent();
    const interval = setInterval(pollLiveCurrent, 1000);

    return () => {
      isSubscribed = false;
      clearInterval(interval);
    };
  }, [selectedMachineId]);

  // Secondary lower-frequency poll (every 3s) or on machine change for history, logs & maintenance data
  useEffect(() => {
    let isSubscribed = true;

    const fetchSecondaryData = async () => {
      try {
        const [hist, lg, maint, sData] = await Promise.all([
          getHistoryData(selectedMachineId),
          getRecentLogs(selectedMachineId),
          getMaintenanceData(),
          getStatus().catch(() => null),
        ]);

        if (isSubscribed) {
          setHistoryData(hist);
          setLogs(lg.logs || []);
          setMaintenanceData(maint);
          if (sData) setStatusData(sData);
        }
      } catch (err) {
        console.warn('Failed to fetch secondary history/maintenance data:', err);
      }
    };

    fetchSecondaryData();
    const interval = setInterval(fetchSecondaryData, 3000);

    return () => {
      isSubscribed = false;
      clearInterval(interval);
    };
  }, [selectedMachineId]);

  // Handle control actions (Start, Pause, Next, Reset, Speed)
  const handleControlAction = async (action, actionSpeed) => {
    try {
      const targetSpeed = actionSpeed !== undefined ? actionSpeed : speed;
      const res = await postControlAction(action, targetSpeed);

      if (res) {
        setAutoPlay(res.auto_play);
        setSpeed(res.simulation_speed);
      }
    } catch (err) {
      console.error('Control action error:', err);
    }
  };

  return (
    <div className="min-h-screen bg-[#F5F7FA] flex flex-col font-sans text-gray-900 antialiased selection:bg-blue-100">
      {/* Top Bar across entire page */}
      <TopBar backendStatus={backendStatus} statusData={statusData} currentData={currentData} />

      <div className="flex flex-1 overflow-hidden">
        {/* Left Sidebar */}
        <Sidebar
          activeTab={activeTab}
          setActiveTab={setActiveTab}
          currentData={currentData}
          backendStatus={backendStatus}
          onControlAction={handleControlAction}
          speed={speed}
          setSpeed={setSpeed}
          autoPlay={autoPlay}
        />

        {/* Main Content Area */}
        <main className="flex-1 overflow-y-auto p-6 max-w-[1600px] mx-auto w-full">
          {activeTab === 'dashboard' && (
            <Dashboard
              currentData={currentData}
              historyData={historyData}
              logs={logs}
              selectedMachineId={selectedMachineId}
              onSelectMachine={setSelectedMachineId}
              modelData={modelData}
            />
          )}
          {activeTab === 'analytics' && (
            <Analytics
              historyData={historyData}
              selectedMachineId={selectedMachineId}
              onSelectMachine={setSelectedMachineId}
            />
          )}
          {activeTab === 'maintenance' && (
            <Maintenance
              maintenanceData={maintenanceData}
              historyData={historyData}
              selectedMachineId={selectedMachineId}
              onSelectMachine={(mId) => {
                setSelectedMachineId(mId);
                setActiveTab('dashboard');
              }}
            />
          )}
          {activeTab === 'evaluation' && <Evaluation modelData={modelData} />}
        </main>
      </div>
    </div>
  );
}

export default App;

