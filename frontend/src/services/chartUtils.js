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
    targetMin = Math.max(0, rawMin - 0.15);
    targetMax = rawMax + 0.15;
  } else if (type === 'current') {
    targetMin = Math.max(0, rawMin - 0.5);
    targetMax = rawMax + 0.5;
  } else if (type === 'pressure') {
    targetMin = Math.max(0, rawMin - 0.5);
    targetMax = rawMax + 0.5;
  } else if (type === 'rpm') {
    targetMin = Math.floor((rawMin - 25.0) / 10) * 10;
    targetMax = Math.ceil((rawMax + 25.0) / 10) * 10;
  } else if (type === 'flow_rate') {
    targetMin = Math.floor(rawMin - 5.0);
    targetMax = Math.ceil(rawMax + 5.0);
  } else if (type === 'oil_temperature') {
    targetMin = Math.floor(rawMin - 2.0);
    targetMax = Math.ceil(rawMax + 2.0);
  } else if (type === 'power_consumption' || type === 'power') {
    targetMin = Math.max(0, rawMin - 1.0);
    targetMax = rawMax + 1.0;
  } else if (type === 'rul') {
    targetMin = Math.max(0, rawMin - 20.0);
    targetMax = rawMax + 20.0;
  } else {
    targetMin = rawMin - 1.0;
    targetMax = rawMax + 1.0;
  }

  // Minimum span enforcement
  if (targetMax - targetMin < minSpan) {
    const center = (targetMin + targetMax) / 2;
    targetMin = center - minSpan / 2;
    targetMax = center + minSpan / 2;
  }

  // Soft clamping
  targetMin = Math.max(minClamp, targetMin);
  targetMax = Math.min(maxClamp, targetMax);

  if (rangeRef && rangeRef.current) {
    const [prevMin, prevMax] = rangeRef.current;
    const smoothMin = prevMin + (targetMin - prevMin) * 0.15;
    const smoothMax = prevMax + (targetMax - prevMax) * 0.15;
    rangeRef.current = [smoothMin, smoothMax];
    return [smoothMin, smoothMax];
  }

  if (rangeRef) {
    rangeRef.current = [targetMin, targetMax];
  }

  return [targetMin, targetMax];
}
