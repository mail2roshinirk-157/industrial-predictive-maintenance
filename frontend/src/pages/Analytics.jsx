import React from 'react';
import AnalyticsCharts from '../components/AnalyticsCharts';

const Analytics = ({ historyData, selectedMachineId, onSelectMachine }) => {
  const machineList = ['M-001', 'M-002', 'M-003', 'M-004', 'M-005', 'M-006', 'M-007', 'M-008'];

  return (
    <div className="space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-2 border-b border-gray-200 gap-2">
        <div>
          <h1 className="text-base font-bold text-gray-900">Historical Trends & Fleet Comparative Analysis</h1>
          <p className="text-xs text-gray-500">Multi-parameter machine degradation trends & 8-machine fleet overlays</p>
        </div>

        <div className="flex items-center space-x-2">
          <label className="text-xs font-semibold text-gray-700">Machine Focus:</label>
          <select
            value={selectedMachineId || 'M-001'}
            onChange={(e) => onSelectMachine && onSelectMachine(e.target.value)}
            className="px-3 py-1.5 text-xs font-semibold bg-white border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {machineList.map((mId) => (
              <option key={mId} value={mId}>
                {mId}
              </option>
            ))}
          </select>
        </div>
      </div>

      <AnalyticsCharts
        historyData={historyData}
        selectedMachineId={selectedMachineId}
        onSelectMachine={onSelectMachine}
      />
    </div>
  );
};

export default Analytics;

