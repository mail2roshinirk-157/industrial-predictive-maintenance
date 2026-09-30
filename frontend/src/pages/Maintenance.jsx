import React from 'react';
import MaintenanceDashboard from '../components/MaintenanceDashboard';

const Maintenance = ({ maintenanceData, historyData, selectedMachineId, onSelectMachine }) => {
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between pb-2 border-b border-gray-200">
        <div>
          <h1 className="text-base font-bold text-gray-900">Engineer Maintenance Decision Support</h1>
          <p className="text-xs text-gray-500">RUL-driven intelligent maintenance policy decision engine & servicing recommendations for all 8 machines</p>
        </div>
      </div>
      <MaintenanceDashboard
        maintenanceData={maintenanceData}
        historyData={historyData}
        selectedMachineId={selectedMachineId}
        onSelectMachine={onSelectMachine}
      />
    </div>
  );
};

export default Maintenance;

