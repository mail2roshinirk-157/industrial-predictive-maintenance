import React from 'react';
import Plot from './Plot';
import { Cpu, CheckCircle2, BarChart2, TrendingUp, Layers } from 'lucide-react';

const ModelEvaluation = ({ modelData }) => {
  if (!modelData) {
    return (
      <div className="industrial-card p-8 text-center text-gray-500">
        Loading model evaluation metrics...
      </div>
    );
  }

  const {
    algorithm = 'Random Forest Regressor',
    dataset_size = 1000,
    train_size = 800,
    test_size = 200,
    mae = 0.0,
    rmse = 0.0,
    r2_score = 0.0,
    feature_importance = [],
    scatter_plot = { actual: [], predicted: [] },
    residuals = [],
    models = null,
  } = modelData || {};

  const rf = models?.random_forest;
  const xgb = models?.xgboost;

  // Defensive array normalization
  const featList = Array.isArray(feature_importance)
    ? feature_importance
    : typeof feature_importance === 'object' && feature_importance !== null
      ? Object.entries(feature_importance).map(([feature, importance]) => ({ feature, importance }))
      : [];

  const actualPoints = Array.isArray(scatter_plot?.actual) ? scatter_plot.actual : [];
  const predictedPoints = Array.isArray(scatter_plot?.predicted) ? scatter_plot.predicted : [];
  const resList = Array.isArray(residuals) ? residuals : [];

  return (
    <div className="space-y-6">
      {/* Side-by-Side Model Benchmark Comparison Table if available */}
      {rf && xgb && (
        <div className="industrial-card p-6 bg-white border border-gray-200">
          <div className="flex items-center justify-between pb-4 border-b border-gray-200 mb-4">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-purple-50 text-purple-600 rounded-lg">
                <Layers className="w-6 h-6" />
              </div>
              <div>
                <h2 className="text-base font-bold text-gray-900">Side-by-Side Model Benchmark Comparison</h2>
                <p className="text-xs text-gray-500">Trained on identical 80/20 train-test split ({(dataset_size || 91250).toLocaleString()} samples)</p>
              </div>
            </div>
            <span className="inline-flex items-center px-3 py-1 bg-purple-50 text-purple-700 rounded-full text-xs font-bold border border-purple-200">
              ⭐ Top Performer: XGBoost Regressor
            </span>
          </div>

          <div className="overflow-x-auto">
            <table className="min-w-full text-left text-xs">
              <thead className="bg-gray-50 text-gray-700 uppercase font-semibold border-y border-gray-200">
                <tr>
                  <th className="py-2.5 px-4">Evaluation Metric</th>
                  <th className="py-2.5 px-4 text-blue-700">🌲 Random Forest Regressor</th>
                  <th className="py-2.5 px-4 text-purple-700">🚀 XGBoost Regressor</th>
                  <th className="py-2.5 px-4 text-emerald-700">Superior Model</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100 font-medium text-gray-900">
                <tr className="hover:bg-gray-50/50">
                  <td className="py-2.5 px-4 font-semibold text-gray-700">Training R² Score</td>
                  <td className="py-2.5 px-4">{rf.train_r2}</td>
                  <td className="py-2.5 px-4">{xgb.train_r2}</td>
                  <td className="py-2.5 px-4 text-gray-500">RF (Tighter train fit)</td>
                </tr>
                <tr className="hover:bg-gray-50/50 bg-purple-50/20">
                  <td className="py-2.5 px-4 font-semibold text-gray-700">Testing R² Score</td>
                  <td className="py-2.5 px-4 font-bold text-blue-900">{rf.test_r2}</td>
                  <td className="py-2.5 px-4 font-bold text-purple-900">{xgb.test_r2}</td>
                  <td className="py-2.5 px-4 font-bold text-emerald-600">⭐ XGBoost (+{(xgb.test_r2 - rf.test_r2).toFixed(4)})</td>
                </tr>
                <tr className="hover:bg-gray-50/50 bg-purple-50/20">
                  <td className="py-2.5 px-4 font-semibold text-gray-700">Mean Absolute Error (MAE)</td>
                  <td className="py-2.5 px-4">{rf.mae} Days</td>
                  <td className="py-2.5 px-4 font-bold text-purple-900">{xgb.mae} Days</td>
                  <td className="py-2.5 px-4 font-bold text-emerald-600">⭐ XGBoost (-{(rf.mae - xgb.mae).toFixed(2)} Days)</td>
                </tr>
                <tr className="hover:bg-gray-50/50 bg-purple-50/20">
                  <td className="py-2.5 px-4 font-semibold text-gray-700">Root Mean Squared Error (RMSE)</td>
                  <td className="py-2.5 px-4">{rf.rmse} Days</td>
                  <td className="py-2.5 px-4 font-bold text-purple-900">{xgb.rmse} Days</td>
                  <td className="py-2.5 px-4 font-bold text-emerald-600">⭐ XGBoost (-{(rf.rmse - xgb.rmse).toFixed(2)} Days)</td>
                </tr>
                <tr className="hover:bg-gray-50/50">
                  <td className="py-2.5 px-4 font-semibold text-gray-700">Ensemble Architecture</td>
                  <td className="py-2.5 px-4 text-gray-600">Bagging (Parallel 100 Trees)</td>
                  <td className="py-2.5 px-4 text-gray-600">Boosting (Sequential 100 Trees)</td>
                  <td className="py-2.5 px-4 text-gray-500">Gradient Optimization</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Model Summary Card */}
      <div className="industrial-card p-6 bg-white">
        <div className="flex items-center justify-between pb-4 border-b border-gray-200 mb-6">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-blue-50 text-blue-600 rounded-lg">
              <Cpu className="w-6 h-6" />
            </div>
            <div>
              <h2 className="text-base font-bold text-gray-900">{algorithm} Diagnostics</h2>
              <p className="text-xs text-gray-500">Validation metrics on test dataset holdout (80/20 split)</p>
            </div>
          </div>
          <span className="inline-flex items-center px-3 py-1 bg-emerald-50 text-emerald-700 rounded-full text-xs font-semibold border border-emerald-200">
            <CheckCircle2 className="w-3.5 h-3.5 mr-1" /> Models Trained & Available
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
          <div className="bg-gray-50 p-3.5 rounded-lg border border-gray-200">
            <span className="text-[11px] font-bold text-gray-500 uppercase tracking-wider block">Top Model</span>
            <span className="text-xs font-bold text-gray-900 mt-1 block truncate">{algorithm}</span>
          </div>

          <div className="bg-gray-50 p-3.5 rounded-lg border border-gray-200">
            <span className="text-[11px] font-bold text-gray-500 uppercase tracking-wider block">Dataset Size</span>
            <span className="text-base font-bold text-gray-900 mt-1 block">{(dataset_size || 91250).toLocaleString()} rows</span>
          </div>

          <div className="bg-gray-50 p-3.5 rounded-lg border border-gray-200">
            <span className="text-[11px] font-bold text-gray-500 uppercase tracking-wider block">Train/Test Split</span>
            <span className="text-base font-bold text-gray-900 mt-1 block">{(train_size || 73000).toLocaleString()} / {(test_size || 18250).toLocaleString()}</span>
          </div>

          <div className="bg-blue-50/60 p-3.5 rounded-lg border border-blue-200">
            <span className="text-[11px] font-bold text-blue-700 uppercase tracking-wider block">MAE</span>
            <span className="text-xl font-bold text-blue-900 mt-1 block">{mae} <span className="text-xs font-normal text-blue-600">Days</span></span>
          </div>

          <div className="bg-blue-50/60 p-3.5 rounded-lg border border-blue-200">
            <span className="text-[11px] font-bold text-blue-700 uppercase tracking-wider block">RMSE</span>
            <span className="text-xl font-bold text-blue-900 mt-1 block">{rmse} <span className="text-xs font-normal text-blue-600">Days</span></span>
          </div>

          <div className="bg-emerald-50/60 p-3.5 rounded-lg border border-emerald-200">
            <span className="text-[11px] font-bold text-emerald-700 uppercase tracking-wider block">R² Score</span>
            <span className="text-xl font-bold text-emerald-900 mt-1 block">{r2_score}</span>
          </div>
        </div>
      </div>

      {/* Diagnostic Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Feature Importance Bar Chart */}
        <div className="industrial-card p-4">
          <div className="flex items-center space-x-2 pb-3 border-b border-gray-200 mb-2">
            <BarChart2 className="w-4 h-4 text-blue-600" />
            <h3 className="text-xs font-bold text-gray-900 uppercase tracking-wider">Feature Importance</h3>
          </div>
          <div className="w-full h-64">
            <Plot
              data={[
                {
                  x: featList.map((f) => f.importance),
                  y: featList.map((f) => f.feature),
                  type: 'bar',
                  orientation: 'h',
                  marker: { color: '#2563EB' },
                },
              ]}
              layout={{
                autosize: true,
                height: 240,
                margin: { l: 90, r: 20, t: 10, b: 35 },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: '#FAFAFA',
                xaxis: { title: { text: 'Importance', font: { size: 10, color: '#6B7280' } }, gridcolor: '#F3F4F6' },
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
                  x: actualPoints,
                  y: predictedPoints,
                  mode: 'markers',
                  type: 'scatter',
                  marker: { color: '#2563EB', size: 6, opacity: 0.7 },
                  name: 'Test Point',
                },
                {
                  x: [0, 300],
                  y: [0, 300],
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
                  x: resList,
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

