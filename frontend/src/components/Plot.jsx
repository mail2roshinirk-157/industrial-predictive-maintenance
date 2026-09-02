import React from 'react';
import Plotly from 'plotly.js-dist-min';
import createPlotlyComponent from 'react-plotly.js/factory';

// Create a stable Plot component using the minified Plotly distribution
const PlotlyComponent = createPlotlyComponent(Plotly);

class PlotErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.warn('Plotly rendering error caught by boundary:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="w-full h-full min-h-[160px] flex flex-col items-center justify-center p-4 bg-gray-50/80 border border-gray-200 rounded-lg text-gray-500 text-xs">
          <span className="font-medium text-gray-600">Chart rendering standby</span>
          <button
            onClick={() => this.setState({ hasError: false })}
            className="mt-2 px-2.5 py-1 bg-blue-50 text-blue-600 font-semibold rounded border border-blue-200 hover:bg-blue-100 transition-colors"
          >
            Refresh Chart
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

const Plot = (props) => {
  return (
    <PlotErrorBoundary>
      <PlotlyComponent {...props} />
    </PlotErrorBoundary>
  );
};

export default Plot;
