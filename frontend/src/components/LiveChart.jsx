import React, { useState, useRef } from 'react';
import Plot from 'react-plotly.js';
import { calculateAdaptiveYRange } from '../services/chartUtils';


const LiveChart = ({ historyData }) => {
  const [selectedMetric, setSelectedMetric] = useState('Temperature');
  const [isDeviationView, setIsDeviationView] = useState(false);
  const currentRangeRef = useRef(null);

  const metricConfigs = {
    Temperature: { label: 'Operating Temp (°C)', key: 'temperature_trend', color: '#2563EB', unit: '°C', minClamp: 55, maxClamp: 85, minSpan: 6, type: 'temperature', dtick: 2, minorDtick: 0.5 },
    Vibration: { label: 'Vibration RMS (mm/s)', key: 'vibration_trend', color: '#2563EB', unit: 'mm/s', minClamp: 0, maxClamp: 6, minSpan: 1, type: 'vibration', dtick: 0.5, minorDtick: 0.1 },
    Motor_Current: { label: 'Motor Current (A)', key: 'motor_current_trend', color: '#2563EB', unit: 'A', minClamp: 5, maxClamp: 25, minSpan: 3, type: 'current', dtick: 2, minorDtick: 0.5 },
    Pressure: { label: 'Operating Pressure (bar)', key: 'pressure_trend', color: '#2563EB', unit: 'bar', minClamp: 2.0, maxClamp: 8.0, minSpan: 1.0, type: 'pressure', dtick: 1.0, minorDtick: 0.2 },
    RPM: { label: 'Shaft Speed (RPM)', key: 'rpm_trend', color: '#2563EB', unit: 'RPM', minClamp: 1500, maxClamp: 1900, minSpan: 60, type: 'rpm', dtick: 50, minorDtick: 10 },
    Flow_Rate: { label: 'Fluid Flow Rate (L/min)', key: 'flow_rate_trend', color: '#2563EB', unit: 'L/min', minClamp: 90, maxClamp: 180, minSpan: 15, type: 'flow', dtick: 15, minorDtick: 5 },
    Oil_Temperature: { label: 'Lube Oil Temp (°C)', key: 'oil_temperature_trend', color: '#2563EB', unit: '°C', minClamp: 35, maxClamp: 75, minSpan: 6, type: 'oil_temperature', dtick: 5, minorDtick: 1 },
    Load_Percent: { label: 'Operating Load (%)', key: 'load_percent_trend', color: '#8B5CF6', unit: '%', minClamp: 30, maxClamp: 110, minSpan: 10, type: 'load', dtick: 10, minorDtick: 2 },
    Oil_Debris_Count: { label: 'Oil Debris (particles/mL)', key: 'oil_debris_count_trend', color: '#D97706', unit: 'pt/mL', minClamp: 0, maxClamp: 500, minSpan: 50, type: 'debris', dtick: 50, minorDtick: 10 },
  };

  const config = metricConfigs[selectedMetric] || metricConfigs.Temperature;
  const trend = historyData?.[config.key] || { actual: [], predicted_future: [] };
  const actualList = Array.isArray(trend?.actual) ? trend.actual : [];
  const predictedList = Array.isArray(trend?.predicted_future) ? trend.predicted_future : [];
  const healthTrend = Array.isArray(historyData?.machine_health_trend?.actual) ? historyData.machine_health_trend.actual : [];

  const xHist = actualList.map((_, i) => i + 1);
  const latestDay = xHist.length > 0 ? xHist[xHist.length - 1] : 1;
  const xFuture = (actualList.length > 0 && predictedList.length > 0)
    ? predictedList.map((_, i) => latestDay + i)
    : [];

  const visibleMin = Math.max(1, latestDay - 100);
  const visibleMax = latestDay + 20;

  // Purely local in-memory transformation (zero backend network calls)
  const baseline = actualList.length > 0 ? actualList[0] : 0;
  const yActual = isDeviationView ? actualList.map((v) => roundDec(v - baseline, 2)) : actualList;
  const yPredicted = isDeviationView ? predictedList.map((v) => roundDec(v - baseline, 2)) : predictedList;
  
  const adaptiveYRange = calculateAdaptiveYRange(
    yActual,
    isDeviationView ? -10 : config.minClamp,
    isDeviationView ? 30 : config.maxClamp,
    isDeviationView ? 4 : config.minSpan,
    config.type,
    currentRangeRef
  );

  const yAxisTitle = isDeviationView ? `Δ ${config.label} (vs Baseline ${baseline} ${config.unit})` : config.label;
  const hoverFmt = `%{y:.2f} ${config.unit}<extra></extra>`;

  function roundDec(val, dec = 2) {
    return Math.round(val * Math.pow(10, dec)) / Math.pow(10, dec);
  }

  // Telemetry-driven degradation detection (first day health drops below 98.0%)
  let degradationDay = null;
  for (let i = 0; i < healthTrend.length; i++) {
    if (healthTrend[i] < 98.0) {
      degradationDay = i + 1;
      break;
    }
  }

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
      text: `Condition Degradation Detected (Day ${degradationDay})`,
      showarrow: false,
      font: { size: 10, color: '#DC2626' },
      bgcolor: '#FEE2E2',
      bordercolor: '#FCA5A5',
      borderwidth: 1,
      borderpad: 2,
    },
  ] : [];

  return (
    <div className="industrial-card p-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-3 border-b border-gray-200 mb-2 gap-2">
        <div>
          <h2 className="text-sm font-bold text-gray-900">Live Sensor Stream & Predictive Projection</h2>
          <p className="text-xs text-gray-500">Real-time signal analysis across digital twin lifecycle (10 Telemetry Parameters)</p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          {/* Purely Local Deviation View Toggle */}
          <div className="flex items-center space-x-1 bg-gray-100 p-0.5 rounded-lg border border-gray-200">
            <button
              onClick={() => {
                currentRangeRef.current = null;
                setIsDeviationView(false);
              }}
              className={`px-2.5 py-1 text-xs font-semibold rounded-md transition-colors ${!isDeviationView ? 'bg-white text-gray-900 shadow-sm' : 'text-gray-600 hover:text-gray-900'}`}
            >
              Absolute
            </button>
            <button
              onClick={() => {
                currentRangeRef.current = null;
                setIsDeviationView(true);
              }}
              className={`px-2.5 py-1 text-xs font-semibold rounded-md transition-colors ${isDeviationView ? 'bg-blue-600 text-white shadow-sm' : 'text-gray-600 hover:text-gray-900'}`}
            >
              Δ Deviation
            </button>
          </div>

          <div className="flex items-center space-x-2">
            <label className="text-xs font-semibold text-gray-600">Select Parameter:</label>
            <select
              value={selectedMetric}
              onChange={(e) => {
                currentRangeRef.current = null;
                setSelectedMetric(e.target.value);
              }}
              className="bg-gray-50 border border-gray-300 text-gray-900 text-xs rounded-lg focus:ring-blue-500 focus:border-blue-500 block px-3 py-1.5 font-medium cursor-pointer"
            >
              <option value="Temperature">Operating Temp (°C)</option>
              <option value="Oil_Temperature">Lube Oil Temp (°C)</option>
              <option value="Vibration">Vibration RMS (mm/s)</option>
              <option value="RPM">Shaft Speed (RPM)</option>
              <option value="Pressure">Operating Pressure (bar)</option>
              <option value="Flow_Rate">Flow Rate (L/min)</option>
              <option value="Motor_Current">Motor Current (A)</option>
              <option value="Power_Consumption">Power Demand (kW)</option>
              <option value="Load_Percent">Operating Load (%)</option>
              <option value="Oil_Debris_Count">Oil Debris (particles/mL)</option>
            </select>
          </div>
        </div>
      </div>

      <div className="w-full h-[360px]">
        <Plot
          data={[
            {
              x: xHist,
              y: yActual,
              type: 'scatter',
              mode: 'lines+markers',
              name: isDeviationView ? 'Δ Actual Reading' : 'Actual Reading',
              line: { color: config.color, width: 3, shape: 'spline', smoothing: 0.45 },
              marker: { size: 5, color: config.color },
              hovertemplate: hoverFmt,
            },
            {
              x: xFuture,
              y: yPredicted,
              type: 'scatter',
              mode: 'lines',
              name: isDeviationView ? 'Δ Predicted Future' : 'Predicted Future',
              line: { color: '#F59E0B', width: 2.5, dash: 'dash', shape: 'spline', smoothing: 0.45 },
              hovertemplate: hoverFmt,
            },
          ]}
          layout={{
            autosize: true,
            height: 350,
            uirevision: `${selectedMetric}_${isDeviationView}`,
            transition: { duration: 300, easing: 'cubic-in-out' },
            margin: { l: 50, r: 20, t: 40, b: 45 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: '#FAFAFA',
            xaxis: {
              title: { text: 'Day', font: { size: 11, color: '#6B7280' } },
              gridcolor: '#E5E7EB',
              zeroline: false,
              range: [visibleMin, visibleMax],
              autorange: false,
            },
            yaxis: {
              title: { text: yAxisTitle, font: { size: 11, color: '#6B7280' } },
              gridcolor: '#E5E7EB',
              zeroline: isDeviationView,
              zerolinecolor: '#9CA3AF',
              zerolinewidth: 1.5,
              range: adaptiveYRange,
              dtick: config.dtick,
              minor: { showgrid: true, gridcolor: '#F3F4F6', dtick: config.minorDtick },
              autorange: false,
            },
            shapes: shapes,
            annotations: annotations,
            legend: {
              orientation: 'h',
              y: 1.15,
              x: 1,
              xanchor: 'right',
              font: { size: 11, color: '#374151' },
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

export default LiveChart;



