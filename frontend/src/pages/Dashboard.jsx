import React from 'react';
import { Thermometer, Activity, Zap, Calendar, ShieldAlert } from 'lucide-react';
import GaugeCard from '../components/GaugeCard';
import MetricCard from '../components/MetricCard';
import StatusBadge from '../components/StatusBadge';
import LiveChart from '../components/LiveChart';
import SensorTable from '../components/SensorTable';

const Dashboard = ({ currentData, historyData, logs }) => {
  const health = currentData?.machine_health ?? 100;
  const statusStr = currentData?.machine_status || 'Healthy';

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


  return (
    <div className="space-y-4">
      {/* Row 1: Single Horizontal Row with Health Gauge and equal-width Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3 items-stretch">
        {/* Health Gauge (~250px container) */}
        <div className="lg:col-span-1 min-w-[200px]">
          <GaugeCard value={health} statusColor={statusColor} />
        </div>

        {/* RUL Card */}
        <MetricCard
          title="Remaining Useful Life"
          value={currentData?.predicted_rul_days ?? '--'}
          unit="Days"
          icon={Calendar}
          valueColor="text-blue-600"
        />

        {/* Machine Stage Card */}
        <MetricCard
          title="Machine Stage"
          badgeComponent={<StatusBadge status={currentData?.machine_status || 'Healthy'} />}
        />

        {/* Alert Status Card */}
        <MetricCard
          title="Alert Status"
          icon={ShieldAlert}
          badgeComponent={<StatusBadge status={currentData?.alert_status || 'Healthy'} />}
        />

        {/* Temperature Card */}
        <MetricCard
          title="Temperature"
          value={currentData?.temperature ?? '--'}
          unit="°C"
          icon={Thermometer}
        />

        {/* Vibration RMS Card */}
        <MetricCard
          title="Vibration RMS"
          value={currentData?.vibration ?? '--'}
          unit="mm/s"
          icon={Activity}
        />

        {/* Motor Current Card */}
        <MetricCard
          title="Motor Current"
          value={currentData?.motor_current ?? '--'}
          unit="A"
          icon={Zap}
        />
      </div>

      {/* Row 2: ONE large Live Sensor Graph with dropdown */}
      <LiveChart historyData={historyData} />

      {/* Row 3: Recent Sensor Readings Table (Latest 10 rows) */}
      <SensorTable logs={logs} />
    </div>
  );
};

export default Dashboard;
