import React from 'react';
import StatusBadge from './StatusBadge';

const SensorTable = ({ logs }) => {
  const logList = logs || [];

  return (
    <div className="industrial-card p-4">
      <div className="flex items-center justify-between pb-3 border-b border-gray-200 mb-3">
        <div>
          <h2 className="text-sm font-bold text-gray-900">Recent Sensor Readings</h2>
          <p className="text-xs text-gray-500">Latest 10 telemetric records from edge monitoring unit</p>
        </div>
        <span className="text-xs font-semibold px-2 py-1 bg-gray-100 text-gray-600 rounded">
          Last {logList.length} logs
        </span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-xs text-left text-gray-600">
          <thead className="text-[11px] text-gray-500 uppercase bg-gray-50/70 border-b border-gray-200 font-bold tracking-wider">
            <tr>
              <th scope="col" className="px-3 py-2.5">Timestamp</th>
              <th scope="col" className="px-3 py-2.5">Temp (°C)</th>
              <th scope="col" className="px-3 py-2.5">Oil T (°C)</th>
              <th scope="col" className="px-3 py-2.5">Vib (mm/s)</th>
              <th scope="col" className="px-3 py-2.5">RPM</th>
              <th scope="col" className="px-3 py-2.5">Press (bar)</th>
              <th scope="col" className="px-3 py-2.5">Flow (L/m)</th>
              <th scope="col" className="px-3 py-2.5">Curr (A)</th>
              <th scope="col" className="px-3 py-2.5">Pwr (kW)</th>
              <th scope="col" className="px-3 py-2.5">Load (%)</th>
              <th scope="col" className="px-3 py-2.5">Debris</th>
              <th scope="col" className="px-3 py-2.5">Pred RUL</th>
              <th scope="col" className="px-3 py-2.5">Health</th>
              <th scope="col" className="px-3 py-2.5">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100 font-mono">
            {logList.length === 0 ? (
              <tr>
                <td colSpan="14" className="px-3 py-4 text-center text-gray-400 font-sans">
                  No log entries available
                </td>
              </tr>
            ) : (
              logList.map((log, idx) => (
                <tr key={idx} className="hover:bg-gray-50/80 transition-colors">
                  <td className="px-3 py-2 text-gray-900 font-sans whitespace-nowrap">{log.timestamp}</td>
                  <td className="px-3 py-2 font-medium text-gray-800">{log.temperature}</td>
                  <td className="px-3 py-2 font-medium text-gray-800">{log.oil_temperature ?? '--'}</td>
                  <td className="px-3 py-2 font-medium text-gray-800">{log.vibration}</td>
                  <td className="px-3 py-2 font-medium text-gray-800">{log.rpm ?? '--'}</td>
                  <td className="px-3 py-2 font-medium text-gray-800">{log.pressure ?? '--'}</td>
                  <td className="px-3 py-2 font-medium text-gray-800">{log.flow_rate ?? '--'}</td>
                  <td className="px-3 py-2 font-medium text-gray-800">{log.motor_current}</td>
                  <td className="px-3 py-2 font-medium text-gray-800">{log.power_consumption ?? '--'}</td>
                  <td className="px-3 py-2 font-medium text-gray-800">{log.load_percent ?? '--'}</td>
                  <td className="px-3 py-2 font-medium text-gray-800">{log.oil_debris_count ?? '--'}</td>
                  <td className="px-3 py-2 font-bold text-blue-700 font-sans">{log.predicted_rul}d</td>
                  <td className="px-3 py-2 font-semibold text-gray-900">{log.machine_health}%</td>
                  <td className="px-3 py-2 font-sans">
                    <StatusBadge status={log.status} />
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default SensorTable;
