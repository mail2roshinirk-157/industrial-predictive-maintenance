import React from 'react';
import { Thermometer, Activity, Zap, Calendar, ShieldAlert, Cpu, Server, AlertTriangle, CheckCircle, ShieldCheck, Wind, Gauge } from 'lucide-react';
import GaugeCard from '../components/GaugeCard';
import MetricCard from '../components/MetricCard';
import StatusBadge from '../components/StatusBadge';
import LiveChart from '../components/LiveChart';
import SensorTable from '../components/SensorTable';

const Dashboard = ({ currentData, historyData, logs, selectedMachineId, onSelectMachine }) => {
  const machines = currentData?.machines || [];
  const selectedMachine = machines.find((m) => m.Machine_ID === selectedMachineId) || currentData || {};

  const health = selectedMachine?.Machine_Health ?? selectedMachine?.machine_health ?? 100;
  const statusStr = selectedMachine?.Machine_Status || selectedMachine?.machine_status || 'Healthy';
  const machineName = selectedMachine?.Machine_Name || selectedMachine?.machine_name || selectedMachineId || 'Machine M-001';
  const predictedRul = selectedMachine?.Predicted_RUL ?? selectedMachine?.predicted_rul_days ?? '--';
  const recommendedAction = selectedMachine?.Recommended_Action || selectedMachine?.recommended_action || 'Continue Normal Operation';

  // Dynamic Fleet KPIs
  const kpi = currentData?.fleet_kpi || {
    total_machines: machines.length,
    healthy_count: machines.filter((m) => m.Machine_Status === 'Healthy').length,
    monitor_count: machines.filter((m) => ['Slight Wear', 'Moderate Wear'].includes(m.Machine_Status)).length,
    warning_count: machines.filter((m) => m.Machine_Status === 'Warning').length,
    critical_count: machines.filter((m) => m.Machine_Status === 'Critical').length,
    high_priority_count: machines.filter((m) => m.Inspection_Priority === 'High').length,
    urgent_priority_count: machines.filter((m) => m.Inspection_Priority === 'Urgent').length,
    critical_priority_count: machines.filter((m) => m.Inspection_Priority === 'Critical').length,
    avg_health: currentData?.fleet_avg_health || 100,
    avg_predicted_rul: Math.round(machines.reduce((acc, m) => acc + (m.Predicted_RUL || 0), 0) / (machines.length || 1)),
    critical_alerts_count: machines.filter((m) => ['Urgent', 'Critical'].includes(m.Inspection_Priority)).length,
  };

  const needsAttentionCount = kpi.monitor_count + kpi.warning_count;

  // Real telemetry baseline deviation indicators
  const abnormalIndicators = selectedMachine?.Abnormal_Indicators || selectedMachine?.abnormal_indicators || [];

  const getGaugeColor = (s) => {
    switch (s) {
      case 'Healthy':
        return '#22C55E';
      case 'Slight Wear':
        return '#2563EB';
      case 'Moderate Wear':
        return '#F59E0B';
      case 'Warning':
        return '#D97706';
      case 'Critical':
        return '#EF4444';
      default:
        return '#22C55E';
    }
  };

  const statusColor = getGaugeColor(statusStr);

  // 8 Telemetry Feature Values
  const tempVal = parseFloat(selectedMachine?.Temperature ?? selectedMachine?.temperature ?? 62.0);
  const oilTempVal = parseFloat(selectedMachine?.Oil_Temperature ?? selectedMachine?.oil_temperature ?? 45.0);
  const vibVal = parseFloat(selectedMachine?.Vibration ?? selectedMachine?.vibration ?? 0.20);
  const rpmVal = parseFloat(selectedMachine?.RPM ?? selectedMachine?.rpm ?? 1750.0);
  const pressVal = parseFloat(selectedMachine?.Pressure ?? selectedMachine?.pressure ?? 5.0);
  const flowVal = parseFloat(selectedMachine?.Flow_Rate ?? selectedMachine?.flow_rate ?? 150.0);
  const currVal = parseFloat(selectedMachine?.Motor_Current ?? selectedMachine?.motor_current ?? 8.0);
  const pwrVal = parseFloat(selectedMachine?.Power_Consumption ?? selectedMachine?.power_consumption ?? 15.0);
  const loadVal = parseFloat(selectedMachine?.Load_Percent ?? selectedMachine?.load_percent ?? 70.0);
  const debVal = parseInt(selectedMachine?.Oil_Debris_Count ?? selectedMachine?.oil_debris_count ?? 5);

  const getSensorCondition = (type, val) => {
    if (type === 'temp') {
      if (val >= 75.0) return { label: 'High', color: 'bg-red-100 text-red-800 border-red-300' };
      if (val >= 68.0) return { label: 'Elevated', color: 'bg-amber-100 text-amber-800 border-amber-300' };
      return { label: 'Normal', color: 'bg-emerald-100 text-emerald-800 border-emerald-300' };
    }
    if (type === 'oil_temp') {
      if (val >= 58.0) return { label: 'High', color: 'bg-red-100 text-red-800 border-red-300' };
      if (val >= 50.0) return { label: 'Elevated', color: 'bg-amber-100 text-amber-800 border-amber-300' };
      return { label: 'Normal', color: 'bg-emerald-100 text-emerald-800 border-emerald-300' };
    }
    if (type === 'vib') {
      if (val >= 1.0) return { label: 'High', color: 'bg-red-100 text-red-800 border-red-300' };
      if (val >= 0.5) return { label: 'Elevated', color: 'bg-amber-100 text-amber-800 border-amber-300' };
      return { label: 'Normal', color: 'bg-emerald-100 text-emerald-800 border-emerald-300' };
    }
    if (type === 'rpm') {
      if (val < 1680) return { label: 'Below Expected', color: 'bg-amber-100 text-amber-800 border-amber-300' };
      if (val >= 1850) return { label: 'High', color: 'bg-red-100 text-red-800 border-red-300' };
      return { label: 'Normal', color: 'bg-emerald-100 text-emerald-800 border-emerald-300' };
    }
    if (type === 'press') {
      if (val < 3.8) return { label: 'Below Expected', color: 'bg-amber-100 text-amber-800 border-amber-300' };
      if (val >= 6.5) return { label: 'High', color: 'bg-red-100 text-red-800 border-red-300' };
      return { label: 'Normal', color: 'bg-emerald-100 text-emerald-800 border-emerald-300' };
    }
    if (type === 'flow') {
      if (val < 130.0) return { label: 'Below Expected', color: 'bg-amber-100 text-amber-800 border-amber-300' };
      if (val >= 170.0) return { label: 'High', color: 'bg-red-100 text-red-800 border-red-300' };
      return { label: 'Normal', color: 'bg-emerald-100 text-emerald-800 border-emerald-300' };
    }
    if (type === 'curr') {
      if (val >= 12.0) return { label: 'High', color: 'bg-red-100 text-red-800 border-red-300' };
      if (val >= 10.0) return { label: 'Elevated', color: 'bg-amber-100 text-amber-800 border-amber-300' };
      return { label: 'Normal', color: 'bg-emerald-100 text-emerald-800 border-emerald-300' };
    }
    if (type === 'power') {
      if (val >= 22.0) return { label: 'High', color: 'bg-red-100 text-red-800 border-red-300' };
      if (val >= 18.0) return { label: 'Elevated', color: 'bg-amber-100 text-amber-800 border-amber-300' };
      return { label: 'Normal', color: 'bg-emerald-100 text-emerald-800 border-emerald-300' };
    }
    if (type === 'load') {
      if (val >= 92.0) return { label: 'Heavy Load', color: 'bg-red-100 text-red-800 border-red-300' };
      if (val >= 82.0) return { label: 'Elevated', color: 'bg-amber-100 text-amber-800 border-amber-300' };
      return { label: 'Nominal', color: 'bg-emerald-100 text-emerald-800 border-emerald-300' };
    }
    if (type === 'debris') {
      if (val >= 150) return { label: 'High Wear', color: 'bg-red-100 text-red-800 border-red-300' };
      if (val >= 50) return { label: 'Elevated', color: 'bg-amber-100 text-amber-800 border-amber-300' };
      return { label: 'Clean', color: 'bg-emerald-100 text-emerald-800 border-emerald-300' };
    }
    return { label: 'Normal', color: 'bg-emerald-100 text-emerald-800 border-emerald-300' };
  };

  return (
    <div className="space-y-6">
      {/* Client-Facing Fleet KPI Banner */}
      <div className="bg-gradient-to-r from-slate-900 via-slate-800 to-indigo-950 rounded-2xl p-5 text-white shadow-xl border border-slate-800">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4 pb-3 border-b border-slate-700/60">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-blue-500/20 rounded-lg text-blue-400 border border-blue-500/30">
              <Server className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-bold text-white tracking-wide">Live Fleet Monitoring Summary</h2>
              <p className="text-xs text-slate-400">Real-time equipment condition monitoring across active facility (10 Multi-Sensor Telemetry Signals)</p>
            </div>
          </div>
          <div className="flex items-center space-x-2 text-xs font-mono bg-slate-800/80 px-3 py-1.5 rounded-lg border border-slate-700">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span>Facility Status: Active</span>
          </div>
        </div>

        {/* Business Value Fleet KPI Cards */}
        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3 text-center">
          <div className="bg-slate-800/60 p-3 rounded-xl border border-slate-700/50">
            <div className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">Total Fleet</div>
            <div className="text-xl font-black text-white mt-1">{kpi.total_machines}</div>
            <div className="text-[10px] text-slate-500">Machines</div>
          </div>

          <div className="bg-emerald-950/40 p-3 rounded-xl border border-emerald-800/40">
            <div className="text-[10px] uppercase font-bold text-emerald-400 tracking-wider">Healthy</div>
            <div className="text-xl font-black text-emerald-300 mt-1">{kpi.healthy_count}</div>
            <div className="text-[10px] text-emerald-500">Normal</div>
          </div>

          <div className="bg-amber-950/40 p-3 rounded-xl border border-amber-800/40">
            <div className="text-[10px] uppercase font-bold text-amber-400 tracking-wider">Needs Attention</div>
            <div className="text-xl font-black text-amber-300 mt-1">{needsAttentionCount}</div>
            <div className="text-[10px] text-amber-500">Monitor</div>
          </div>

          <div className="bg-rose-950/40 p-3 rounded-xl border border-rose-800/40">
            <div className="text-[10px] uppercase font-bold text-rose-400 tracking-wider">Critical</div>
            <div className="text-xl font-black text-rose-300 mt-1">{kpi.critical_count}</div>
            <div className="text-[10px] text-rose-500">Imminent</div>
          </div>

          <div className="bg-indigo-950/40 p-3 rounded-xl border border-indigo-800/40">
            <div className="text-[10px] uppercase font-bold text-indigo-300 tracking-wider">Avg Fleet Health</div>
            <div className="text-xl font-black text-indigo-200 mt-1">{kpi.avg_health}%</div>
            <div className="text-[10px] text-indigo-400">Condition</div>
          </div>

          <div className="bg-cyan-950/40 p-3 rounded-xl border border-cyan-800/40">
            <div className="text-[10px] uppercase font-bold text-cyan-300 tracking-wider">Estimated Life</div>
            <div className="text-xl font-black text-cyan-200 mt-1">{kpi.avg_predicted_rul}d</div>
            <div className="text-[10px] text-cyan-400">Mean Remaining</div>
          </div>

          <div className="bg-red-950/50 p-3 rounded-xl border border-red-700/60 col-span-2 sm:col-span-1">
            <div className="text-[10px] uppercase font-bold text-red-400 tracking-wider">Maintenance Alerts</div>
            <div className="text-xl font-black text-red-300 mt-1">{kpi.critical_alerts_count}</div>
            <div className="text-[10px] text-red-500">Active Action</div>
          </div>
        </div>
      </div>

      {/* Row 1: Fleet Equipment Grid */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-2">
            <Cpu className="w-5 h-5 text-blue-600" />
            <h2 className="text-base font-bold text-gray-900">Industrial Equipment Fleet ({machines.length} Machines)</h2>
          </div>
          <span className="text-xs text-gray-500">Select any machine to view detailed multi-sensor diagnostic</span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-3">
          {machines.map((m) => {
            const isSelected = m.Machine_ID === selectedMachineId;
            const mHealth = m.Machine_Health;
            const mStatus = m.Machine_Status;
            const mRul = m.Predicted_RUL;
            const hasAnomaly = m.Active_Event && m.Active_Event !== 'None';

            return (
              <button
                key={m.Machine_ID}
                onClick={() => onSelectMachine && onSelectMachine(m.Machine_ID)}
                className={`flex flex-col justify-between p-3 rounded-xl border text-left transition-all duration-200 ${
                  isSelected
                    ? 'bg-blue-50/80 border-blue-500 shadow-md ring-2 ring-blue-400/30 scale-[1.02]'
                    : 'bg-white border-gray-200 hover:border-blue-300 hover:shadow-sm'
                }`}
              >
                <div className="flex items-center justify-between w-full mb-1">
                  <span className="text-xs font-bold text-gray-900">{m.Machine_ID}</span>
                  {hasAnomaly && (
                    <span className="w-2 h-2 rounded-full bg-red-500 animate-ping" title={m.Active_Event} />
                  )}
                </div>
                <div className="text-[10px] text-gray-500 truncate mb-2">{m.Machine_Name || m.Machine_ID}</div>

                <div className="space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] text-gray-500">Health</span>
                    <span className="text-xs font-semibold text-gray-800">{mHealth}%</span>
                  </div>
                  <div className="w-full bg-gray-100 rounded-full h-1.5 overflow-hidden">
                    <div
                      className="h-full transition-all duration-500 rounded-full"
                      style={{
                        width: `${Math.max(0, Math.min(100, mHealth))}%`,
                        backgroundColor: getGaugeColor(mStatus),
                      }}
                    />
                  </div>
                  <div className="flex items-center justify-between pt-1">
                    <span className="text-[10px] text-gray-400">Est. Life</span>
                    <span className="text-[11px] font-bold text-blue-600">{mRul}d</span>
                  </div>
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Selected Machine Health Overview & Risk Indicators */}
      <div className="bg-white rounded-xl p-5 border border-gray-200 shadow-sm space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-gray-100 pb-3">
          <div>
            <h3 className="text-base font-bold text-gray-900">
              Machine Health Overview: <span className="text-blue-600">{selectedMachineId}</span> ({machineName})
            </h3>
            <p className="text-xs text-gray-500 mt-0.5">Real-time multi-sensor condition diagnostic & recommended action</p>
          </div>
          <StatusBadge status={statusStr} />
        </div>

        {/* Why is this machine at risk? Section */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 bg-slate-50 p-4 rounded-xl border border-slate-200">
          <div className="md:col-span-2 space-y-2">
            <div className="flex items-center space-x-2 text-xs font-bold text-slate-800 uppercase tracking-wider">
              <AlertTriangle className="w-4 h-4 text-amber-500" />
              <span>Why is this machine at risk?</span>
            </div>
            
            <div className="space-y-1.5 pt-1">
              {abnormalIndicators.length > 0 ? (
                abnormalIndicators.map((ind, idx) => (
                  <div key={idx} className="flex items-center space-x-2 text-xs text-slate-700 font-medium bg-white px-3 py-1.5 rounded-lg border border-slate-200 shadow-2xs">
                    <span className="w-1.5 h-1.5 rounded-full bg-amber-500 shrink-0" />
                    <span>{ind}</span>
                  </div>
                ))
              ) : (
                <div className="flex items-center space-x-2 text-xs text-emerald-700 font-medium bg-emerald-50 px-3 py-1.5 rounded-lg border border-emerald-200">
                  <CheckCircle className="w-4 h-4 text-emerald-500 shrink-0" />
                  <span>All 10 operational telemetry parameters are functioning within normal baseline limits.</span>
                </div>
              )}
            </div>
          </div>

          {/* Recommended Action Callout */}
          <div className="flex flex-col justify-between bg-blue-50/90 p-4 rounded-xl border border-blue-200 text-blue-950">
            <div>
              <div className="flex items-center space-x-1.5 text-xs font-bold text-blue-900 uppercase tracking-wider mb-2">
                <ShieldCheck className="w-4 h-4 text-blue-600" />
                <span>Recommended Action</span>
              </div>
              <p className="text-xs font-bold text-blue-900 leading-snug">
                {recommendedAction}
              </p>
            </div>
            <div className="text-[11px] text-blue-700 font-medium mt-3 pt-2 border-t border-blue-200/60">
              Estimated Remaining Life: <span className="font-bold text-blue-900">{predictedRul} Days</span>
            </div>
          </div>
        </div>
      </div>

      {/* Row 2: Selected Machine Health Gauge + Key Status */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 items-stretch">
        <GaugeCard value={health} statusColor={statusColor} />

        <MetricCard
          title="Estimated Remaining Life"
          value={predictedRul}
          unit="Days"
          icon={Calendar}
          valueColor="text-blue-600"
        />

        <MetricCard
          title="Alert Priority Level"
          icon={ShieldAlert}
          badgeComponent={<StatusBadge status={selectedMachine?.Inspection_Priority || selectedMachine?.alert_status || 'Healthy'} />}
        />
      </div>

      {/* Row 3: 5 Client-Facing Operational Category Telemetry Cards */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <h3 className="text-xs font-bold text-gray-700 uppercase tracking-wider">
            Operational Telemetry Categories (10 Active Signals)
          </h3>
          <span className="text-[11px] text-gray-500 font-medium">Real-time condition indicators</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-3">
          {/* 1. THERMAL CATEGORY */}
          <div className="bg-white rounded-xl p-4 border border-slate-200 shadow-xs space-y-3">
            <div className="flex items-center justify-between pb-2 border-b border-slate-100">
              <div className="flex items-center space-x-2">
                <Thermometer className="w-4 h-4 text-orange-500" />
                <span className="text-xs font-bold text-slate-800 uppercase tracking-wider">Thermal</span>
              </div>
              <span className="text-[10px] font-semibold text-slate-400">2 Signals</span>
            </div>
            <div className="space-y-2.5">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-[11px] text-slate-500 font-medium">Operating Temp</div>
                  <div className="text-base font-bold text-slate-900">{tempVal} <span className="text-xs text-slate-500 font-normal">°C</span></div>
                </div>
                <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${getSensorCondition('temp', tempVal).color}`}>
                  {getSensorCondition('temp', tempVal).label}
                </span>
              </div>
              <div className="flex items-center justify-between pt-2 border-t border-slate-100">
                <div>
                  <div className="text-[11px] text-slate-500 font-medium">Lube Oil Temp</div>
                  <div className="text-base font-bold text-slate-900">{oilTempVal} <span className="text-xs text-slate-500 font-normal">°C</span></div>
                </div>
                <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${getSensorCondition('oil_temp', oilTempVal).color}`}>
                  {getSensorCondition('oil_temp', oilTempVal).label}
                </span>
              </div>
            </div>
          </div>

          {/* 2. MECHANICAL CATEGORY */}
          <div className="bg-white rounded-xl p-4 border border-slate-200 shadow-xs space-y-3">
            <div className="flex items-center justify-between pb-2 border-b border-slate-100">
              <div className="flex items-center space-x-2">
                <Activity className="w-4 h-4 text-blue-500" />
                <span className="text-xs font-bold text-slate-800 uppercase tracking-wider">Mechanical</span>
              </div>
              <span className="text-[10px] font-semibold text-slate-400">2 Signals</span>
            </div>
            <div className="space-y-2.5">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-[11px] text-slate-500 font-medium">Vibration RMS</div>
                  <div className="text-base font-bold text-slate-900">{vibVal} <span className="text-xs text-slate-500 font-normal">mm/s</span></div>
                </div>
                <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${getSensorCondition('vib', vibVal).color}`}>
                  {getSensorCondition('vib', vibVal).label}
                </span>
              </div>
              <div className="flex items-center justify-between pt-2 border-t border-slate-100">
                <div>
                  <div className="text-[11px] text-slate-500 font-medium">Shaft Speed</div>
                  <div className="text-base font-bold text-slate-900">{rpmVal} <span className="text-xs text-slate-500 font-normal">RPM</span></div>
                </div>
                <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${getSensorCondition('rpm', rpmVal).color}`}>
                  {getSensorCondition('rpm', rpmVal).label}
                </span>
              </div>
            </div>
          </div>

          {/* 3. PROCESS CATEGORY */}
          <div className="bg-white rounded-xl p-4 border border-slate-200 shadow-xs space-y-3">
            <div className="flex items-center justify-between pb-2 border-b border-slate-100">
              <div className="flex items-center space-x-2">
                <Wind className="w-4 h-4 text-purple-500" />
                <span className="text-xs font-bold text-slate-800 uppercase tracking-wider">Process</span>
              </div>
              <span className="text-[10px] font-semibold text-slate-400">2 Signals</span>
            </div>
            <div className="space-y-2.5">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-[11px] text-slate-500 font-medium">Operating Pressure</div>
                  <div className="text-base font-bold text-slate-900">{pressVal} <span className="text-xs text-slate-500 font-normal">bar</span></div>
                </div>
                <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${getSensorCondition('press', pressVal).color}`}>
                  {getSensorCondition('press', pressVal).label}
                </span>
              </div>
              <div className="flex items-center justify-between pt-2 border-t border-slate-100">
                <div>
                  <div className="text-[11px] text-slate-500 font-medium">Fluid Flow Rate</div>
                  <div className="text-base font-bold text-slate-900">{flowVal} <span className="text-xs text-slate-500 font-normal">L/min</span></div>
                </div>
                <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${getSensorCondition('flow', flowVal).color}`}>
                  {getSensorCondition('flow', flowVal).label}
                </span>
              </div>
            </div>
          </div>

          {/* 4. ELECTRICAL CATEGORY */}
          <div className="bg-white rounded-xl p-4 border border-slate-200 shadow-xs space-y-3">
            <div className="flex items-center justify-between pb-2 border-b border-slate-100">
              <div className="flex items-center space-x-2">
                <Zap className="w-4 h-4 text-amber-500" />
                <span className="text-xs font-bold text-slate-800 uppercase tracking-wider">Electrical</span>
              </div>
              <span className="text-[10px] font-semibold text-slate-400">2 Signals</span>
            </div>
            <div className="space-y-2.5">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-[11px] text-slate-500 font-medium">Motor Current</div>
                  <div className="text-base font-bold text-slate-900">{currVal} <span className="text-xs text-slate-500 font-normal">A</span></div>
                </div>
                <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${getSensorCondition('curr', currVal).color}`}>
                  {getSensorCondition('curr', currVal).label}
                </span>
              </div>
              <div className="flex items-center justify-between pt-2 border-t border-slate-100">
                <div>
                  <div className="text-[11px] text-slate-500 font-medium">Power Demand</div>
                  <div className="text-base font-bold text-slate-900">{pwrVal} <span className="text-xs text-slate-500 font-normal">kW</span></div>
                </div>
                <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${getSensorCondition('power', pwrVal).color}`}>
                  {getSensorCondition('power', pwrVal).label}
                </span>
              </div>
            </div>
          </div>

          {/* 5. TRIBOLOGY & LOAD CATEGORY */}
          <div className="bg-white rounded-xl p-4 border border-slate-200 shadow-xs space-y-3">
            <div className="flex items-center justify-between pb-2 border-b border-slate-100">
              <div className="flex items-center space-x-2">
                <Gauge className="w-4 h-4 text-emerald-600" />
                <span className="text-xs font-bold text-slate-800 uppercase tracking-wider">Load & Debris</span>
              </div>
              <span className="text-[10px] font-semibold text-slate-400">2 Signals</span>
            </div>
            <div className="space-y-2.5">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-[11px] text-slate-500 font-medium">Operating Load</div>
                  <div className="text-base font-bold text-slate-900">{loadVal} <span className="text-xs text-slate-500 font-normal">%</span></div>
                </div>
                <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${getSensorCondition('load', loadVal).color}`}>
                  {getSensorCondition('load', loadVal).label}
                </span>
              </div>
              <div className="flex items-center justify-between pt-2 border-t border-slate-100">
                <div>
                  <div className="text-[11px] text-slate-500 font-medium">Oil Debris Count</div>
                  <div className="text-base font-bold text-slate-900">{debVal} <span className="text-xs text-slate-500 font-normal">pt/mL</span></div>
                </div>
                <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${getSensorCondition('debris', debVal).color}`}>
                  {getSensorCondition('debris', debVal).label}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Row 4: Live Sensor Chart for Selected Machine */}
      <LiveChart historyData={historyData} />

      {/* Row 5: Recent Sensor Logs Table */}
      <SensorTable logs={logs} />
    </div>
  );
};

export default Dashboard;
