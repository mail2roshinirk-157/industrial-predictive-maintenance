import React, { useState, useEffect } from 'react';
import { Cpu, Database, Clock, ShieldCheck } from 'lucide-react';

const TopBar = ({ backendStatus, statusData, currentData }) => {
  const [timeStr, setTimeStr] = useState(new Date().toLocaleTimeString());

  useEffect(() => {
    const timer = setInterval(() => {
      setTimeStr(new Date().toLocaleTimeString());
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  const machines = currentData?.machines || [];
  const fleetSize = currentData?.fleet_size || statusData?.fleet_size || machines.length || 8;
  const machineRange = machines.length > 0
    ? `(${machines[0].Machine_ID} ... ${machines[machines.length - 1].Machine_ID})`
    : '';

  return (
    <header className="bg-slate-900 text-white px-6 py-3 shadow-md flex items-center justify-between text-xs sm:text-sm font-medium border-b border-slate-800">
      <div className="flex items-center space-x-6">
        <div className="flex items-center space-x-2">
          <span className="bg-blue-600 text-white px-2 py-0.5 rounded text-xs font-bold tracking-wide">FLEET</span>
          <span className="font-semibold text-slate-100">{fleetSize} Machines {machineRange}</span>
        </div>
        <div className="hidden md:flex items-center space-x-1.5 text-slate-300">
          <Cpu className="w-4 h-4 text-blue-400" />
          <span>Live Fleet Monitoring</span>
        </div>
        <div className="hidden lg:flex items-center space-x-1.5 text-slate-400">
          <Database className="w-4 h-4 text-slate-400" />
          <span>Industrial Equipment Fleet</span>
        </div>
      </div>

      <div className="flex items-center space-x-6">
        <div className="hidden sm:flex items-center space-x-1.5 text-slate-300">
          <ShieldCheck className="w-4 h-4 text-emerald-400" />
          <span>Predictive Maintenance Engine</span>
        </div>
        <div className="flex items-center space-x-1.5 text-slate-300 font-mono">
          <Clock className="w-4 h-4 text-slate-400" />
          <span>{timeStr}</span>
        </div>
        <div className="flex items-center space-x-2 bg-slate-800 px-3 py-1 rounded-full border border-slate-700">
          <span className={`w-2 h-2 rounded-full ${statusData?.active_data_source === 'mqtt' ? (statusData?.mqtt_connected ? 'bg-cyan-400 animate-pulse' : 'bg-amber-400') : 'bg-blue-400'}`}></span>
          <span className="text-[11px] font-semibold text-slate-300">
            {statusData?.active_data_source === 'mqtt'
              ? `MQTT (${statusData?.mqtt_connected ? 'CONNECTED' : 'WAITING'})`
              : 'SCADA SIMULATOR'}
          </span>
        </div>
        <div className="flex items-center space-x-2 bg-slate-800 px-3 py-1 rounded-full border border-slate-700">
          <span className={`w-2.5 h-2.5 rounded-full ${backendStatus ? 'bg-emerald-500 animate-pulse' : 'bg-red-500'}`}></span>
          <span className="text-xs font-semibold text-slate-200">{backendStatus ? 'SYSTEM ONLINE' : 'DISCONNECTED'}</span>
        </div>
      </div>
    </header>
  );
};

export default TopBar;

