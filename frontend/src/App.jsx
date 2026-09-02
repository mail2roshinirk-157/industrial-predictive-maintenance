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
      } catch (err) {
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

  // Fixed 1000ms polling loop for visualization only
  useEffect(() => {
    let isSubscribed = true;

    const pollData = async () => {
      try {
        // Fetch telemetry state in parallel for visualization
        const [curr, hist, lg, maint] = await Promise.all([
          getCurrentData(),
          getHistoryData(),
          getRecentLogs(),
          getMaintenanceData(),
        ]);

        if (isSubscribed) {
          setCurrentData(curr);
          setHistoryData(hist);
          setLogs(lg.logs || []);
          setMaintenanceData(maint);
          setAutoPlay(curr.auto_play);
          setSpeed(curr.simulation_speed);

          // Reset failure counter on successful polling cycle
          consecutiveFailuresRef.current = 0;
          setBackendStatus(true);
        }
      } catch (err) {
        if (isSubscribed) {
          consecutiveFailuresRef.current += 1;
          // Only show "Disconnected" after 3 consecutive failed polling cycles
          if (consecutiveFailuresRef.current >= 3) {
            setBackendStatus(false);
          }
        }
      }
    };

    // Immediate first fetch
    pollData();

    // Fixed 1-second interval (1000ms) - independent of simulation speed
    const interval = setInterval(pollData, 1000);

    return () => {
      isSubscribed = false;
      clearInterval(interval);
    };
  }, []);

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
      <TopBar backendStatus={backendStatus} statusData={statusData} />

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
            <Dashboard currentData={currentData} historyData={historyData} logs={logs} />
          )}
          {activeTab === 'analytics' && <Analytics historyData={historyData} />}
          {activeTab === 'maintenance' && (
            <Maintenance maintenanceData={maintenanceData} historyData={historyData} />
          )}
          {activeTab === 'evaluation' && <Evaluation modelData={modelData} />}
        </main>
      </div>
    </div>
  );
}

export default App;
