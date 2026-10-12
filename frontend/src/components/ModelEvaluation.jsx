import React from 'react';
import Plot from 'react-plotly.js';
import { Cpu, CheckCircle2, BarChart2, TrendingUp, Layers, FlaskConical, Award } from 'lucide-react';

const ModelEvaluation = ({ modelData }) => {
  if (!modelData) {
    return (
      <div className="industrial-card p-8 text-center text-gray-500">
        Loading model evaluation metrics...
      </div>
    );
  }

  const {
    dataset_size,
    train_size,
    test_size,
    mae,
    rmse,
    r2_score,
    feature_importance = [],
    scatter_plot = { actual: [], predicted: [] },
    residuals = [],
    xgboost_experiment = null
  } = modelData;

  const xgbExp = xgboost_experiment;

  const rfR2 = xgbExp?.random_forest?.r2_score ?? r2_score;
  const rfMae = xgbExp?.random_forest?.mae ?? mae;
  const rfRmse = xgbExp?.random_forest?.rmse ?? rmse;

  const xgbR2 = xgbExp?.xgboost?.r2_score;
  const xgbMae = xgbExp?.xgboost?.mae;
  const xgbRmse = xgbExp?.xgboost?.rmse;

  const baseline3f = xgbExp?.rf_3feature_baseline || xgbExp?.baseline_3feature;
  const baseR2 = baseline3f?.r2_score;
  const baseMae = baseline3f?.mae;
  const baseRmse = baseline3f?.rmse;

  // Dynamically compute Upgrade Delta: (10-feature Random Forest) minus (3-feature Baseline)
  const upgradeDeltaR2 = (rfR2 != null && baseR2 != null) ? Number((rfR2 - baseR2).toFixed(4)) : (modelData?.upgrade_delta?.r2_score_delta ?? null);
  const upgradeDeltaMae = (rfMae != null && baseMae != null) ? Number((rfMae - baseMae).toFixed(2)) : (modelData?.upgrade_delta?.mae_days_delta ?? null);
  const upgradeDeltaRmse = (rfRmse != null && baseRmse != null) ? Number((rfRmse - baseRmse).toFixed(2)) : (modelData?.upgrade_delta?.rmse_days_delta ?? null);

  const formatDelta = (val, decimals, unit = '') => {
    if (val == null || isNaN(val)) return '—';
    const sign = val >= 0 ? '+' : '';
    return `${sign}${val.toFixed(decimals)}${unit}`;
  };

  const lifecycleRows = xgbExp?.lifecycle_evaluation || [];

  const featureCount = xgbExp?.feature_count || (feature_importance.length > 0 ? feature_importance.length : (xgbExp?.features?.length || 10));

  return (
    <div className="space-y-6">
      {/* Model Summary Card */}
      <div className="industrial-card p-6 bg-white">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-gray-200 mb-6 gap-3">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-blue-50 text-blue-600 rounded-lg">
              <Cpu className="w-6 h-6" />
            </div>
            <div>
              <h2 className="text-base font-bold text-gray-900">Random Forest Production Model & XGBoost Benchmark Evaluation</h2>
              <p className="text-xs text-gray-500">
                Holdout validation metrics on {test_size ? test_size.toLocaleString() : '—'} test samples ({train_size && test_size ? `${Math.round(train_size / (train_size + test_size) * 100)}/${Math.round(test_size / (train_size + test_size) * 100)}` : '—'} train/test split)
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <span className="inline-flex items-center px-3 py-1 bg-emerald-50 text-emerald-700 rounded-full text-xs font-semibold border border-emerald-200">
              <CheckCircle2 className="w-3.5 h-3.5 mr-1" /> Random Forest (Production)
            </span>
            {xgbExp && (
              <span className="inline-flex items-center px-3 py-1 bg-purple-50 text-purple-700 rounded-full text-xs font-semibold border border-purple-200">
                <FlaskConical className="w-3.5 h-3.5 mr-1" /> XGBoost (Offline Experiment)
              </span>
            )}
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
          <div className="bg-gray-50 p-3.5 rounded-lg border border-gray-200">
            <span className="text-[11px] font-bold text-gray-500 uppercase tracking-wider block">Production Model</span>
            <span className="text-xs font-bold text-gray-900 mt-1 block truncate">Random Forest ({featureCount}-Feature)</span>
          </div>

          <div className="bg-gray-50 p-3.5 rounded-lg border border-gray-200">
            <span className="text-[11px] font-bold text-gray-500 uppercase tracking-wider block">Dataset Size</span>
            <span className="text-base font-bold text-gray-900 mt-1 block">{dataset_size ? `${dataset_size.toLocaleString()} rows` : '—'}</span>
          </div>

          <div className="bg-gray-50 p-3.5 rounded-lg border border-gray-200">
            <span className="text-[11px] font-bold text-gray-500 uppercase tracking-wider block">Train/Test Split</span>
            <span className="text-base font-bold text-gray-900 mt-1 block">{(train_size && test_size) ? `${train_size.toLocaleString()} / ${test_size.toLocaleString()}` : '—'}</span>
          </div>

          <div className="bg-blue-50/60 p-3.5 rounded-lg border border-blue-200">
            <span className="text-[11px] font-bold text-blue-700 uppercase tracking-wider block">Production MAE</span>
            <span className="text-xl font-bold text-blue-900 mt-1 block">{rfMae != null ? rfMae : '—'} <span className="text-xs font-normal text-blue-600">Days</span></span>
          </div>

          <div className="bg-blue-50/60 p-3.5 rounded-lg border border-blue-200">
            <span className="text-[11px] font-bold text-blue-700 uppercase tracking-wider block">Production RMSE</span>
            <span className="text-xl font-bold text-blue-900 mt-1 block">{rfRmse != null ? rfRmse : '—'} <span className="text-xs font-normal text-blue-600">Days</span></span>
          </div>

          <div className="bg-emerald-50/60 p-3.5 rounded-lg border border-emerald-200">
            <span className="text-[11px] font-bold text-emerald-700 uppercase tracking-wider block">Production R² Score</span>
            <span className="text-xl font-bold text-emerald-900 mt-1 block">{rfR2 != null ? rfR2 : '—'}</span>
          </div>
        </div>
      </div>

      {/* Model Benchmark Comparison Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Model Performance Comparison Table */}
        <div className="industrial-card p-5 bg-white">
          <div className="flex items-center space-x-2 pb-3 border-b border-gray-200 mb-3">
            <Award className="w-5 h-5 text-indigo-600" />
            <h3 className="text-xs font-bold text-gray-900 uppercase tracking-wider">
              Random Forest (Production) vs XGBoost (Offline Experiment)
            </h3>
          </div>
          <p className="text-xs text-gray-500 mb-3">
            Random Forest is the active production model for live SCADA telemetry and fleet batch inference.
          </p>

          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="border-b border-gray-200 bg-gray-50 text-gray-600 font-bold uppercase text-[10px]">
                  <th className="py-2 px-3">Model Architecture</th>
                  <th className="py-2 px-3">Features</th>
                  <th className="py-2 px-3">Role</th>
                  <th className="py-2 px-3">R² Score</th>
                  <th className="py-2 px-3">MAE (Days)</th>
                  <th className="py-2 px-3">RMSE (Days)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {/* 3-Feature RF Baseline */}
                <tr className="bg-slate-50 font-medium">
                  <td className="py-2.5 px-3 font-bold text-gray-800">Random Forest (Baseline)</td>
                  <td className="py-2.5 px-3 font-mono text-xs">3 Features</td>
                  <td className="py-2.5 px-3">
                    <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-slate-200 text-slate-700 border border-slate-300">
                      Preserved Baseline
                    </span>
                  </td>
                  <td className="py-2.5 px-3 font-mono font-bold text-slate-700">
                    {baseR2 != null ? baseR2 : '—'}
                  </td>
                  <td className="py-2.5 px-3 font-mono font-bold text-slate-700">
                    {baseMae != null ? `${baseMae}d` : '—'}
                  </td>
                  <td className="py-2.5 px-3 font-mono text-slate-600">
                    {baseRmse != null ? `${baseRmse}d` : '—'}
                  </td>
                </tr>
                {/* 10-Feature RF Production */}
                <tr className="bg-blue-50/60 font-medium border-l-4 border-l-blue-600">
                  <td className="py-2.5 px-3 font-bold text-gray-900">Random Forest ({featureCount}-Feature)</td>
                  <td className="py-2.5 px-3 font-mono text-xs font-semibold text-blue-700">{featureCount} Features</td>
                  <td className="py-2.5 px-3">
                    <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-100 text-emerald-800 border border-emerald-300">
                      Active Production
                    </span>
                  </td>
                  <td className="py-2.5 px-3 font-mono font-bold text-emerald-700">{rfR2 != null ? rfR2 : '—'}</td>
                  <td className="py-2.5 px-3 font-mono font-bold text-blue-700">{rfMae != null ? `${rfMae}d` : '—'}</td>
                  <td className="py-2.5 px-3 font-mono text-gray-700">{rfRmse != null ? `${rfRmse}d` : '—'}</td>
                </tr>
                {/* 10-Feature XGBoost Experiment */}
                <tr>
                  <td className="py-2.5 px-3 font-bold text-gray-900">XGBoost Regressor ({featureCount}-Feature)</td>
                  <td className="py-2.5 px-3 font-mono text-xs">{featureCount} Features</td>
                  <td className="py-2.5 px-3">
                    <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-purple-100 text-purple-800 border border-purple-300">
                      Offline Benchmark
                    </span>
                  </td>
                  <td className="py-2.5 px-3 font-mono font-bold text-purple-700">{xgbR2 != null ? xgbR2 : '—'}</td>
                  <td className="py-2.5 px-3 font-mono font-bold text-purple-700">{xgbMae != null ? `${xgbMae}d` : '—'}</td>
                  <td className="py-2.5 px-3 font-mono text-gray-700">{xgbRmse != null ? `${xgbRmse}d` : '—'}</td>
                </tr>
                {upgradeDeltaR2 != null && (
                  <tr className="bg-gray-50 text-[11px] font-semibold text-gray-600">
                    <td className="py-2 px-3 font-bold text-slate-700" colSpan={2}>Upgrade Delta ({featureCount}-Feat RF vs 3-Feat Base)</td>
                    <td className="py-2 px-3 text-gray-400">—</td>
                    <td className="py-2 px-3 font-mono text-emerald-600">{formatDelta(upgradeDeltaR2, 4)}</td>
                    <td className="py-2 px-3 font-mono text-blue-600">{formatDelta(upgradeDeltaMae, 2, 'd')}</td>
                    <td className="py-2 px-3 font-mono text-blue-600">{formatDelta(upgradeDeltaRmse, 2, 'd')}</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Degradation Lifecycle Comparison Table */}
        <div className="industrial-card p-5 bg-white">
          <div className="flex items-center space-x-2 pb-3 border-b border-gray-200 mb-3">
            <FlaskConical className="w-5 h-5 text-purple-600" />
            <h3 className="text-xs font-bold text-gray-900 uppercase tracking-wider">
              Degradation Lifecycle Performance (MAE by Region)
            </h3>
          </div>
          <p className="text-xs text-gray-500 mb-3">
            Comparative Mean Absolute Error (MAE in days) across standard machine health wear regions.
          </p>

          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="border-b border-gray-200 bg-gray-50 text-gray-600 font-bold uppercase text-[10px]">
                  <th className="py-2 px-3">Degradation Lifecycle Region</th>
                  <th className="py-2 px-3">RF MAE</th>
                  <th className="py-2 px-3">XGB MAE</th>
                  <th className="py-2 px-3">MAE Delta</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {lifecycleRows.map((row, idx) => {
                  const rfM = row.random_forest?.mae;
                  const xgbM = row.xgboost?.mae;
                  const d = row.mae_delta;
                  return (
                    <tr key={idx} className="hover:bg-gray-50">
                      <td className="py-2.5 px-3 font-semibold text-gray-800">{row.region}</td>
                      <td className="py-2.5 px-3 font-mono font-bold text-blue-600">{rfM != null ? `${rfM}d` : 'N/A'}</td>
                      <td className="py-2.5 px-3 font-mono font-bold text-purple-600">{xgbM != null ? `${xgbM}d` : 'N/A'}</td>
                      <td className="py-2.5 px-3 font-mono font-medium">
                        {d != null ? (
                          <span className={d <= 0 ? 'text-emerald-600' : 'text-amber-600'}>
                            {d > 0 ? `+${d}d` : `${d}d`}
                          </span>
                        ) : (
                          'N/A'
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Diagnostic Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Feature Importance Bar Chart */}
        <div className="industrial-card p-4">
          <div className="flex items-center space-x-2 pb-3 border-b border-gray-200 mb-2">
            <BarChart2 className="w-4 h-4 text-blue-600" />
            <h3 className="text-xs font-bold text-gray-900 uppercase tracking-wider">Production Model Feature Weights</h3>
          </div>
          <div className="w-full h-64">
            <Plot
              data={[
                {
                  x: feature_importance.map((f) => f.importance),
                  y: feature_importance.map((f) => f.feature === 'Vibration' ? 'Vibration RMS' : f.feature.replace('_', ' ')),
                  type: 'bar',
                  orientation: 'h',
                  marker: { color: '#2563EB' },
                },
              ]}
              layout={{
                autosize: true,
                height: 240,
                margin: { l: 95, r: 20, t: 10, b: 35 },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: '#FAFAFA',
                xaxis: { title: { text: 'Gini Feature Weight', font: { size: 10, color: '#6B7280' } }, gridcolor: '#F3F4F6' },
                yaxis: { automargin: true, font: { size: 11, color: '#111827' } },
              }}
              useResizeHandler={true}
              style={{ width: '100%', height: '100%' }}
              config={{ displayModeBar: false, responsive: true }}
            />
          </div>
        </div>

        {/* Prediction vs Actual Scatter Plot */}
        <div className="industrial-card p-4">
          <div className="flex items-center space-x-2 pb-3 border-b border-gray-200 mb-2">
            <TrendingUp className="w-4 h-4 text-blue-600" />
            <h3 className="text-xs font-bold text-gray-900 uppercase tracking-wider">Prediction vs Actual</h3>
          </div>
          <div className="w-full h-64">
            <Plot
              data={[
                {
                  x: scatter_plot.actual,
                  y: scatter_plot.predicted,
                  mode: 'markers',
                  type: 'scatter',
                  marker: { color: '#2563EB', size: 6, opacity: 0.7 },
                  name: 'Test Point',
                },
                {
                  x: [0, 1000],
                  y: [0, 1000],
                  mode: 'lines',
                  type: 'scatter',
                  line: { color: '#EF4444', dash: 'dash', width: 1.5 },
                  name: 'Ideal Fit (1:1)',
                },
              ]}
              layout={{
                autosize: true,
                height: 240,
                margin: { l: 40, r: 15, t: 10, b: 35 },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: '#FAFAFA',
                xaxis: { title: { text: 'Actual RUL (Days)', font: { size: 10, color: '#6B7280' } }, gridcolor: '#F3F4F6' },
                yaxis: { title: { text: 'Predicted RUL (Days)', font: { size: 10, color: '#6B7280' } }, gridcolor: '#F3F4F6' },
                legend: { orientation: 'h', y: 1.15, x: 1, xanchor: 'right', font: { size: 10 } },
              }}
              useResizeHandler={true}
              style={{ width: '100%', height: '100%' }}
              config={{ displayModeBar: false, responsive: true }}
            />
          </div>
        </div>

        {/* Residual Error Histogram */}
        <div className="industrial-card p-4">
          <div className="flex items-center space-x-2 pb-3 border-b border-gray-200 mb-2">
            <Layers className="w-4 h-4 text-blue-600" />
            <h3 className="text-xs font-bold text-gray-900 uppercase tracking-wider">Residual Error Histogram</h3>
          </div>
          <div className="w-full h-64">
            <Plot
              data={[
                {
                  x: residuals,
                  type: 'histogram',
                  marker: { color: '#22C55E' },
                  opacity: 0.85,
                },
              ]}
              layout={{
                autosize: true,
                height: 240,
                margin: { l: 40, r: 15, t: 10, b: 35 },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: '#FAFAFA',
                xaxis: { title: { text: 'Residual Error (Actual - Pred)', font: { size: 10, color: '#6B7280' } }, gridcolor: '#F3F4F6' },
                yaxis: { title: { text: 'Frequency', font: { size: 10, color: '#6B7280' } }, gridcolor: '#F3F4F6' },
              }}
              useResizeHandler={true}
              style={{ width: '100%', height: '100%' }}
              config={{ displayModeBar: false, responsive: true }}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelEvaluation;
