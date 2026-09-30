import React, { useRef } from 'react';
import { Calendar, Wrench, ShieldCheck, Clock } from 'lucide-react';
import MetricCard from './MetricCard';
import StatusBadge from './StatusBadge';
import Plot from 'react-plotly.js';
import { calculateAdaptiveYRange } from '../services/chartUtils';

const SingleTrendPlot = ({
  title,
  yLabel,
  trendData,
  actualColor = '#2563EB',
  isFullWidth = false,
  degradationDay = null,
  minClamp = 0,
  maxClamp = 100,
  minSpan = 1,
  metricType = 'general',
  dtick = null,
  minorDtick = null
}) => {
  const actual = trendData?.actual || [];
  const predicted = trendData?.predicted_future || [];
  const rangeRef = useRef(null);

  const xHist = actual.map((_, i) => i + 1);
  const latestDay = xHist.length > 0 ? xHist[xHist.length - 1] : 1;
  const xFuture = actual.length > 0 ? predicted.map((_, i) => latestDay + i) : [];

  const visibleMin = Math.max(1, latestDay - 100);
  const visibleMax = latestDay + 20;

  const adaptiveRange = metricType === 'health'
    ? [0, 100]
    : calculateAdaptiveYRange(actual, minClamp, maxClamp, minSpan, metricType, rangeRef);

  const shapes = degradationDay ? [
    {
      type: 'line',
      x0: degradationDay,
      x1: degradationDay,
      y0: 0,
      y1: 1,
      yref: 'paper',
      line: { color: '#EF4444', width: 1.5, dash: 'dash' },
    },
  ] : [];

  const annotations = degradationDay ? [
    {
      x: degradationDay,
      y: 1.05,
      yref: 'paper',
      text: `Degradation (Day ${degradationDay})`,
      showarrow: false,
      font: { size: 9, color: '#DC2626' },
      bgcolor: '#FEE2E2',
      bordercolor: '#FCA5A5',
      borderwidth: 1,
      borderpad: 2,
    },
  ] : [];

  const yAxisConfig = {
    title: { text: yLabel, font: { size: 10, color: '#6B7280' } },
    gridcolor: '#E5E7EB',
    zeroline: false,
    range: adaptiveRange,
    autorange: false,
    ...(dtick ? { dtick: dtick } : {}),
    minor: { showgrid: true, gridcolor: '#F3F4F6', ...(minorDtick ? { dtick: minorDtick } : {}) },
  };

  return (
    <div className={`industrial-card p-4 ${isFullWidth ? 'w-full' : ''}`}>
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-xs font-bold text-gray-900 uppercase tracking-wider">{title}</h3>
        {actual.length > 0 && (
          <span className="text-[11px] text-gray-500 font-mono">
            Latest: {actual[actual.length - 1]} {yLabel}
          </span>
        )}
      </div>

      <div className="w-full h-64">
        <Plot
          data={[
            {
              x: xHist,
              y: actual,
              type: 'scatter',
              mode: 'lines+markers',
              name: 'Historical Reading',
              line: { color: actualColor, width: 3, shape: 'spline', smoothing: 0.45 },
              marker: { size: 4, color: actualColor },
              hovertemplate: `%{y:.2f} ${yLabel}<extra></extra>`,
            },
            {
              x: xFuture,
              y: predicted,
              type: 'scatter',
              mode: 'lines',
              name: '20-Step Trend Extrapolation',
              line: { color: '#F59E0B', width: 2.5, dash: 'dash', shape: 'spline', smoothing: 0.45 },
              hovertemplate: `%{y:.2f} ${yLabel}<extra></extra>`,
            },
          ]}
          layout={{
            autosize: true,
            height: 240,
            uirevision: title,
            transition: { duration: 300, easing: 'cubic-in-out' },
            margin: { l: 45, r: 15, t: 25, b: 35 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: '#FAFAFA',
            xaxis: {
              title: { text: 'Day', font: { size: 10, color: '#6B7280' } },
              gridcolor: '#E5E7EB',
              zeroline: false,
              range: [visibleMin, visibleMax],
              autorange: false,
            },
            yaxis: yAxisConfig,
            shapes: shapes,
            annotations: annotations,
            legend: {
              orientation: 'h',
              y: 1.2,
              x: 1,
              xanchor: 'right',
              font: { size: 10, color: '#4B5563' },
            },
          }}
          useResizeHandler={true}
          style={{ width: '100%', height: '100%' }}
          config={{ displayModeBar: false, responsive: true }}
        />
      </div>
    </div>
  );
};


const MaintenanceDashboard = ({ maintenanceData, historyData, selectedMachineId, onSelectMachine }) => {
  const rawList = maintenanceData?.maintenance_list || [];

  // Enforce Priority conceptual ordering: Critical -> Urgent -> High -> Moderate -> Low
  const prioOrder = { Critical: 0, Urgent: 1, High: 2, Moderate: 3, Low: 4 };
  const sortedMaintenanceList = [...rawList].sort(
    (a, b) => (prioOrder[a.inspection_priority] ?? 5) - (prioOrder[b.inspection_priority] ?? 5)
  );

  const selectedOrTop = sortedMaintenanceList.find((m) => m.machine_id === selectedMachineId) || sortedMaintenanceList[0] || {};

  const predictedRul = selectedOrTop?.predicted_rul_days ?? maintenanceData?.predicted_rul_days ?? '--';
  const status = selectedOrTop?.maintenance_status || maintenanceData?.maintenance_status || 'Healthy';
  const action = selectedOrTop?.recommended_action || maintenanceData?.recommended_action || 'Continue Normal Operation';
  const priority = selectedOrTop?.inspection_priority || maintenanceData?.inspection_priority || 'Low';
  const window = selectedOrTop?.next_inspection_window || maintenanceData?.next_inspection_window || 'Routine inspection within 90–120 days';

  const healthTrend = historyData?.machine_health_trend?.actual || [];
  let degradationDay = null;
  for (let i = 0; i < healthTrend.length; i++) {
    if (healthTrend[i] < 98.0) {
      degradationDay = i + 1;
      break;
    }
  }

  const getPriorityBadgeClass = (p) => {
    switch (p) {
      case 'Critical':
        return 'bg-red-100 text-red-800 border-red-300';
      case 'Urgent':
        return 'bg-orange-100 text-orange-800 border-orange-300';
      case 'High':
        return 'bg-amber-100 text-amber-800 border-amber-300';
      case 'Moderate':
        return 'bg-blue-100 text-blue-800 border-blue-300';
      default:
        return 'bg-emerald-100 text-emerald-800 border-emerald-300';
    }
  };

  return (
    <div className="space-y-6">
      {/* Fleet Maintenance Priority Schedule Table */}
      <div className="industrial-card p-4">
        <div className="flex items-center justify-between mb-3">
          <div>
            <h3 className="text-xs font-bold text-gray-900 uppercase tracking-wider">
              Fleet Servicing & Priority Action Schedule ({sortedMaintenanceList.length} Machines)
            </h3>
            <p className="text-[11px] text-gray-500">
              Determines which machines the maintenance team should service first based on equipment condition & remaining life
            </p>
          </div>
          <span className="text-xs font-semibold text-gray-600 bg-gray-100 px-2.5 py-1 rounded border border-gray-200">
            Total Fleet: {sortedMaintenanceList.length}
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-gray-200 bg-gray-50/80 text-[11px] font-bold text-gray-600 uppercase">
                <th className="py-2.5 px-3">Rank</th>
                <th className="py-2.5 px-3">Machine ID</th>
                <th className="py-2.5 px-3">Machine Name</th>
                <th className="py-2.5 px-3">Health %</th>
                <th className="py-2.5 px-3">Predicted RUL</th>
                <th className="py-2.5 px-3">Status</th>
                <th className="py-2.5 px-3">Priority</th>
                <th className="py-2.5 px-3">Recommended Action</th>
                <th className="py-2.5 px-3">Servicing Window</th>
                <th className="py-2.5 px-3">Abnormal Indicators</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100 text-xs">
              {sortedMaintenanceList.map((m, idx) => {
                const isSelected = m.machine_id === selectedMachineId;
                const abnormal = m.abnormal_indicators || [];
                return (
                  <tr
                    key={m.machine_id}
                    onClick={() => onSelectMachine && onSelectMachine(m.machine_id)}
                    className={`cursor-pointer transition-colors ${
                      isSelected ? 'bg-blue-50/90 font-semibold' : 'hover:bg-gray-50'
                    }`}
                  >
                    <td className="py-2.5 px-3 font-bold text-gray-400">#{idx + 1}</td>
                    <td className="py-2.5 px-3 font-mono font-bold text-blue-600">{m.machine_id}</td>
                    <td className="py-2.5 px-3 text-gray-800">{m.machine_name}</td>
                    <td className="py-2.5 px-3 font-semibold">{m.machine_health}%</td>
                    <td className="py-2.5 px-3 font-bold text-blue-700">{m.predicted_rul_days} Days</td>
                    <td className="py-2.5 px-3">
                      <StatusBadge status={m.maintenance_status} />
                    </td>
                    <td className="py-2.5 px-3">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${getPriorityBadgeClass(m.inspection_priority)}`}>
                        {m.inspection_priority}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-gray-700 font-medium truncate max-w-[180px]" title={m.recommended_action}>
                      {m.recommended_action}
                    </td>
                    <td className="py-2.5 px-3 text-gray-500 text-[11px] truncate max-w-[160px]" title={m.next_inspection_window}>
                      {m.next_inspection_window}
                    </td>
                    <td className="py-2.5 px-3">
                      {abnormal.length > 0 ? (
                        <div className="flex flex-wrap gap-1">
                          {abnormal.map((ind, i) => (
                            <span key={i} className="px-1.5 py-0.5 text-[10px] font-medium rounded bg-amber-100 text-amber-800 border border-amber-200" title={ind}>
                              {ind.split(':')[0]}
                            </span>
                          ))}
                        </div>
                      ) : (
                        <span className="text-[10px] text-emerald-600 font-medium">Nominal</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Key Maintenance Decision Cards for Selected / Top Machine */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Predicted RUL */}
        <MetricCard
          title="Predicted Remaining Useful Life"
          value={predictedRul}
          unit="Days"
          icon={Calendar}
          valueColor="text-blue-600"
        />

        {/* Maintenance Status */}
        <MetricCard
          title="Maintenance Status"
          icon={Wrench}
          badgeComponent={<StatusBadge status={status} />}
        />

        {/* Recommended Action */}
        <div className="industrial-card p-4 flex flex-col justify-between h-full flex-1">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-gray-500 uppercase tracking-wider">Recommended Action</span>
            <ShieldCheck className="w-4 h-4 text-blue-500 shrink-0 ml-1" />
          </div>
          <div className="mt-2 mb-1">
            <span className="text-xs font-semibold text-gray-800 bg-blue-50 px-2.5 py-1.5 rounded border border-blue-200 block truncate" title={action}>
              {action}
            </span>
          </div>
        </div>

        {/* Inspection Priority & Next Inspection Window */}
        <div className="industrial-card p-4 flex flex-col justify-between h-full flex-1">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-gray-500 uppercase tracking-wider">Inspection Priority</span>
            <Clock className="w-4 h-4 text-amber-500 shrink-0 ml-1" />
          </div>
          <div className="mt-2 space-y-1">
            <div className="flex items-center space-x-2">
              <span className={`text-xs font-bold px-2 py-0.5 rounded border ${getPriorityBadgeClass(priority)}`}>
                {priority} Priority
              </span>
            </div>
            <p className="text-[11px] font-medium text-gray-600 truncate" title={window}>
              {window}
            </p>
          </div>
        </div>
      </div>

      {/* Main Full-Width Predicted RUL Trend Chart */}
      <SingleTrendPlot
        title="Predicted Remaining Useful Life Trend & Projection"
        yLabel="Days"
        trendData={historyData?.rul_trend}
        actualColor="#2563EB"
        isFullWidth={true}
        degradationDay={degradationDay}
        minClamp={0}
        maxClamp={250}
        minSpan={80}
        metricType="rul"
      />

      {/* 2x2 Grid for Machine Health & Physical Sensor Trends */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <SingleTrendPlot
          title="Machine Health Trend"
          yLabel="%"
          trendData={historyData?.machine_health_trend}
          actualColor="#22C55E"
          degradationDay={degradationDay}
          minClamp={0}
          maxClamp={100}
          minSpan={100}
          metricType="health"
        />
        <SingleTrendPlot
          title="Temperature Trend"
          yLabel="°C"
          trendData={historyData?.temperature_trend}
          actualColor="#2563EB"
          degradationDay={degradationDay}
          minClamp={55}
          maxClamp={85}
          minSpan={6}
          metricType="temperature"
          dtick={2}
          minorDtick={0.5}
        />
        <SingleTrendPlot
          title="Vibration RMS Trend"
          yLabel="mm/s"
          trendData={historyData?.vibration_trend}
          actualColor="#2563EB"
          degradationDay={degradationDay}
          minClamp={0}
          maxClamp={6}
          minSpan={1}
          metricType="vibration"
          dtick={0.5}
          minorDtick={0.1}
        />
        <SingleTrendPlot
          title="Motor Current Trend"
          yLabel="A"
          trendData={historyData?.motor_current_trend}
          actualColor="#2563EB"
          degradationDay={degradationDay}
          minClamp={5}
          maxClamp={25}
          minSpan={3}
          metricType="current"
          dtick={2}
          minorDtick={0.5}
        />
      </div>
    </div>
  );
};

export default MaintenanceDashboard;
