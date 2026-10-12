import React, { useRef } from 'react';
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
  const actual = Array.isArray(trendData?.actual) ? trendData.actual : [];
  const predicted = Array.isArray(trendData?.predicted_future) ? trendData.predicted_future : [];
  const rangeRef = useRef(null);

  const xHist = actual.map((_, i) => i + 1);
  const latestDay = xHist.length > 0 ? xHist[xHist.length - 1] : 1;
  const xFuture = (actual.length > 0 && predicted.length > 0) ? predicted.map((_, i) => latestDay + i) : [];

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
              name: 'Actual',
              line: { color: actualColor, width: 3, shape: 'spline', smoothing: 0.45 },
              marker: { size: 4, color: actualColor },
              hovertemplate: `%{y:.2f} ${yLabel}<extra></extra>`,
            },
            {
              x: xFuture,
              y: predicted,
              type: 'scatter',
              mode: 'lines',
              name: 'Predicted Future',
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


const AnalyticsCharts = ({ historyData }) => {
  const healthTrend = historyData?.machine_health_trend?.actual || [];
  let degradationDay = null;
  for (let i = 0; i < healthTrend.length; i++) {
    if (healthTrend[i] < 98.0) {
      degradationDay = i + 1;
      break;
    }
  }

  return (
    <div className="space-y-6">
      {/* Category 1: Thermal (Operating Temp & Lube Oil Temp) */}
      <div>
        <h4 className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-2">Thermal Monitoring (2 Signals)</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <SingleTrendPlot
            title="Operating Temperature Trend"
            yLabel="°C"
            trendData={historyData?.temperature_trend}
            actualColor="#EA580C"
            degradationDay={degradationDay}
            minClamp={55}
            maxClamp={85}
            minSpan={6}
            metricType="temperature"
            dtick={2}
            minorDtick={0.5}
          />
          <SingleTrendPlot
            title="Lube Oil Temperature Trend"
            yLabel="°C"
            trendData={historyData?.oil_temperature_trend}
            actualColor="#C2410C"
            degradationDay={degradationDay}
            minClamp={35}
            maxClamp={75}
            minSpan={6}
            metricType="oil_temperature"
            dtick={5}
            minorDtick={1}
          />
        </div>
      </div>

      {/* Category 2: Mechanical (Vibration RMS & Shaft Speed RPM) */}
      <div>
        <h4 className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-2">Mechanical Health (2 Signals)</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
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
            title="Shaft Speed (RPM) Trend"
            yLabel="RPM"
            trendData={historyData?.rpm_trend}
            actualColor="#1D4ED8"
            degradationDay={degradationDay}
            minClamp={1500}
            maxClamp={1900}
            minSpan={60}
            metricType="rpm"
            dtick={50}
            minorDtick={10}
          />
        </div>
      </div>

      {/* Category 3: Process Fluid Dynamics (Pressure & Flow Rate) */}
      <div>
        <h4 className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-2">Process & Fluid Dynamics (2 Signals)</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <SingleTrendPlot
            title="Operating Pressure Trend"
            yLabel="bar"
            trendData={historyData?.pressure_trend}
            actualColor="#7C3AED"
            degradationDay={degradationDay}
            minClamp={2.0}
            maxClamp={8.0}
            minSpan={1.0}
            metricType="pressure"
            dtick={1.0}
            minorDtick={0.2}
          />
          <SingleTrendPlot
            title="Fluid Flow Rate Trend"
            yLabel="L/min"
            trendData={historyData?.flow_rate_trend}
            actualColor="#6D28D9"
            degradationDay={degradationDay}
            minClamp={90}
            maxClamp={180}
            minSpan={15}
            metricType="flow_rate"
            dtick={15}
            minorDtick={5}
          />
        </div>
      </div>

      {/* Category 4: Electrical Power (Motor Current & Power Demand) */}
      <div>
        <h4 className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-2">Electrical Power (2 Signals)</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <SingleTrendPlot
            title="Motor Current Trend"
            yLabel="A"
            trendData={historyData?.motor_current_trend}
            actualColor="#D97706"
            degradationDay={degradationDay}
            minClamp={5}
            maxClamp={25}
            minSpan={3}
            metricType="current"
            dtick={2}
            minorDtick={0.5}
          />
          <SingleTrendPlot
            title="Power Demand Trend"
            yLabel="kW"
            trendData={historyData?.power_consumption_trend}
            actualColor="#B45309"
            degradationDay={degradationDay}
            minClamp={10}
            maxClamp={30}
            minSpan={4}
            metricType="power_consumption"
            dtick={3}
            minorDtick={1}
          />
        </div>
      </div>

      {/* Category 5: Operating Context & Tribology (Load & Oil Debris) */}
      <div>
        <h4 className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-2">Operating Context & Tribology (2 Signals)</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <SingleTrendPlot
            title="Operating Load Trend"
            yLabel="%"
            trendData={historyData?.load_percent_trend}
            actualColor="#8B5CF6"
            degradationDay={degradationDay}
            minClamp={30}
            maxClamp={110}
            minSpan={10}
            metricType="load_percent"
            dtick={10}
            minorDtick={2}
          />
          <SingleTrendPlot
            title="Oil Debris Count Trend"
            yLabel="particles/mL"
            trendData={historyData?.oil_debris_count_trend}
            actualColor="#D97706"
            degradationDay={degradationDay}
            minClamp={0}
            maxClamp={500}
            minSpan={50}
            metricType="oil_debris_count"
            dtick={50}
            minorDtick={10}
          />
        </div>
      </div>

      {/* Machine Health & Estimated RUL Overall System Degradation */}
      <div>
        <h4 className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-2">System Degradation & Lifecycle Projections</h4>
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
            title="Estimated Remaining Useful Life (RUL) Trend"
            yLabel="Days"
            trendData={historyData?.rul_trend}
            actualColor="#0284C7"
            degradationDay={degradationDay}
            minClamp={0}
            maxClamp={250}
            minSpan={80}
            metricType="rul"
          />
        </div>
      </div>
    </div>
  );
};


export default AnalyticsCharts;

