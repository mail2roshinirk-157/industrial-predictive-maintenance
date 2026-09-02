import React, { useState, useRef } from 'react';
import Plot from './Plot';

export function calculateAdaptiveYRange(dataArray, minClamp, maxClamp, minSpan, type, rangeRef) {
  if (!dataArray || dataArray.length === 0) {
    return [minClamp, maxClamp];
  }

  const rawMin = Math.min(...dataArray);
  const rawMax = Math.max(...dataArray);

  let targetMin, targetMax;

  if (type === 'temperature') {
    targetMin = Math.floor(rawMin - 1.0);
    targetMax = Math.ceil(rawMax + 1.0);
  } else if (type === 'vibration') {
    targetMin = rawMin - 0.15;
    targetMax = rawMax + 0.15;
  } else if (type === 'current') {
    targetMin = rawMin - 0.5;
    targetMax = rawMax + 0.5;
  } else if (type === 'rul') {
    targetMin = rawMin - 40.0;
    targetMax = rawMax + 40.0;
  } else {
    targetMin = rawMin - 1.0;
    targetMax = rawMax + 1.0;
  }

  if (targetMax - targetMin < minSpan) {
    const mid = (targetMax + targetMin) / 2;
    targetMin = mid - minSpan / 2;
    targetMax = mid + minSpan / 2;
  }

  targetMin = Math.max(minClamp, targetMin);
  targetMax = Math.min(maxClamp, targetMax);

  if (targetMax - targetMin < minSpan) {
    if (targetMin === minClamp) {
      targetMax = Math.min(maxClamp, targetMin + minSpan);
    } else if (targetMax === maxClamp) {
      targetMin = Math.max(minClamp, targetMax - minSpan);
    }
  }

  // 15% Hysteresis check
  if (rangeRef && rangeRef.current) {
    const [cMin, cMax] = rangeRef.current;
    const cSpan = cMax - cMin;
    const thresh = 0.15 * cSpan;

    if (rawMin > cMin + thresh && rawMax < cMax - thresh) {
      return rangeRef.current;
    }
  }

  const res = [Math.round(targetMin * 100) / 100, Math.round(targetMax * 100) / 100];
  if (rangeRef) {
    rangeRef.current = res;
  }
  return res;
}

const LiveChart = ({ historyData }) => {
  const [selectedMetric, setSelectedMetric] = useState('Temperature');
  const [isDeviationView, setIsDeviationView] = useState(false);
  const currentRangeRef = useRef(null);

  const metricConfigs = {
    Temperature: { label: 'Temperature (°C)', key: 'temperature_trend', color: '#2563EB', unit: '°C', minClamp: 55, maxClamp: 85, minSpan: 6, type: 'temperature', dtick: 2, minorDtick: 0.5 },
    Vibration: { label: 'Vibration RMS (mm/s)', key: 'vibration_trend', color: '#2563EB', unit: 'mm/s', minClamp: 0, maxClamp: 6, minSpan: 1, type: 'vibration', dtick: 0.5, minorDtick: 0.1 },
    Motor_Current: { label: 'Motor Current (A)', key: 'motor_current_trend', color: '#2563EB', unit: 'A', minClamp: 5, maxClamp: 25, minSpan: 3, type: 'current', dtick: 2, minorDtick: 0.5 },
  };

  const config = metricConfigs[selectedMetric] || metricConfigs.Temperature;
  const rawTrend = historyData?.[config.key];
  const actualArr = Array.isArray(rawTrend?.actual) ? rawTrend.actual : [];
  const predictedArr = Array.isArray(rawTrend?.predicted_future) ? rawTrend.predicted_future : [];
  const healthTrend = Array.isArray(historyData?.machine_health_trend?.actual) ? historyData.machine_health_trend.actual : [];

  const xHist = actualArr.map((_, i) => i + 1);
  const latestDay = xHist.length > 0 ? xHist[xHist.length - 1] : 1;
  const xFuture = actualArr.length > 0
    ? predictedArr.map((_, i) => latestDay + i)
    : [];

  const visibleMin = Math.max(1, latestDay - 100);
  const visibleMax = latestDay + 20;

  // Purely local in-memory transformation (zero backend network calls)
  const baseline = actualArr.length > 0 ? actualArr[0] : 0;
  const yActual = isDeviationView ? actualArr.map((v) => roundDec(v - baseline, 2)) : actualArr;
  const yPredicted = isDeviationView ? predictedArr.map((v) => roundDec(v - baseline, 2)) : predictedArr;
  
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
          <p className="text-xs text-gray-500">Real-time signal analysis across digital twin lifecycle</p>
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
              <option value="Temperature">Temperature (°C)</option>
              <option value="Vibration">Vibration RMS (mm/s)</option>
              <option value="Motor_Current">Motor Current (A)</option>
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



