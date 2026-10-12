"""
========================================================================================
INDUSTRIAL PREDICTIVE MAINTENANCE PLATFORM
Unsupervised Anomaly Detection Module: Isolation Forest
========================================================================================

Theoretical Foundations & Mathematical Logic of Isolation Forest:
-----------------------------------------------------------------
1. Isolation via Random Recursive Partitioning:
   Traditional anomaly detection methods (like One-Class SVM, LOF, or Elliptic Envelope)
   attempt to build a profile of 'normal' data density and flag points outside that envelope.
   In contrast, Isolation Forest (Liu, Ting, Zhou 2008) explicitly isolates anomalies
   instead of profiling normal instances.

2. Shorter Path Length Principle:
   An ensemble of Isolation Trees (iTrees) is constructed. At each tree node, an algorithm:
     a) Randomly chooses a sensor feature among the available 10 dimensions.
     b) Randomly chooses a split value uniformly distributed between the minimum and
        maximum values of that feature in the current subset.
   Because anomalies are 'few and different', they reside in sparse, low-density regions
   of the multi-dimensional sensor space. Consequently, anomalous points are isolated
   near the root of the tree with very few splits (short average path length).
   Conversely, normal steady-state readings reside in dense clusters and require many
   subsequent random splits to isolate into single leaf nodes (long average path length).

3. Anomaly Score Formulation:
   Let h(x) be the path length (number of edges traversed from root to termination leaf)
   for sample x in an iTree. The anomaly score s(x, n) across an ensemble of trees is:
       s(x, n) = 2^(- E[h(x)] / c(n))
   where:
     - E[h(x)] is the average path length across all isolation trees in the ensemble.
     - c(n) = 2 * (ln(n - 1) + 0.5772156649) - (2 * (n - 1) / n) is the average path length
       of an unsuccessful search in a Binary Search Tree (BST) for sample size n.
   Interpretation:
     - If E[h(x)] -> 0 (very short path length): s(x, n) -> 1.0 (Definite Anomaly).
     - If E[h(x)] -> n - 1 (very long path length): s(x, n) -> 0.0 (Completely Normal).
     - If E[h(x)] -> c(n): s(x, n) -> 0.5 (Boundary / Typical Cluster Edge).

Contamination Parameter Rationale (default = 0.05):
---------------------------------------------------
The `contamination` parameter defines the expected proportion of outliers/anomalies in the
training distribution. In heavy industrial SCADA environments:
- Critical industrial equipment (turbines, pumps, compressors) is engineered with safety
  tolerances to run stably in healthy steady-state ~95% of its operating life.
- A contamination rate of 5% (0.05) corresponds to the expected frequency of abnormal
  operational excursions (e.g., thermal micro-bursts, temporary cavitations, load spikes,
  bearing wear onsets).
- Setting contamination too high (e.g., 15-20%) leads to 'alarm fatigue' for operators,
  causing false positives during benign fluctuations.
- Setting contamination too low (e.g., 0.5%) risks missing subtle precursor degradation
  signals before catastrophic failures occur.
- 0.05 provides the optimal trade-off between sensitivity and operational specificity.
========================================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.ensemble import IsolationForest

import sys
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from config import FEATURE_COLUMNS, EXCLUDED_COLUMNS, MODELS_DIR

# Canonical 10-Sensor Feature Schema (v2)
SENSOR_FEATURES = FEATURE_COLUMNS


def extract_sensor_features(df_or_records: Union[pd.DataFrame, List[dict], dict]) -> pd.DataFrame:
    """
    Extracts strictly the 10 sensor feature columns from input data.
    Automatically excludes timestamps, machine IDs, and target/label columns.
    Imputes missing values with reasonable physical defaults if columns are omitted.
    """
    if isinstance(df_or_records, dict):
        df = pd.DataFrame([df_or_records])
    elif isinstance(df_or_records, list):
        df = pd.DataFrame(df_or_records)
    else:
        df = df_or_records.copy()

    # Derived physics attributes if missing
    if "Load_Percent" not in df.columns:
        df["Load_Percent"] = 70.0

    if "Oil_Debris_Count" not in df.columns:
        if "Machine_Health" in df.columns:
            p_val = np.clip(1.0 - df["Machine_Health"] / 100.0, 0.0, 1.0)
            df["Oil_Debris_Count"] = np.random.poisson(5.0 + 400.0 * (p_val ** 3))
        else:
            df["Oil_Debris_Count"] = 5

    # Ensure all 10 features exist with forward fill and fallback
    for feat in SENSOR_FEATURES:
        if feat not in df.columns:
            df[feat] = 0.0

    X = df[SENSOR_FEATURES].ffill().bfill().fillna(0.0)
    return X


class IsolationForestDetector:
    """
    Standalone Unsupervised Anomaly Detector using scikit-learn's IsolationForest.
    
    Operates strictly independently of any supervised RUL or failure labels.
    Outputs binary anomaly labels and calibrated continuous anomaly scores (0.0 to 1.0).
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 150,
        max_samples: Union[int, float, str] = "auto",
        random_state: int = 42
    ):
        """
        Parameters:
        -----------
        contamination : float, default=0.05
            The proportion of outliers in the data set. Used to define the decision threshold.
        n_estimators : int, default=150
            Number of isolation trees in the ensemble. 150 trees provide smooth path-length estimates.
        max_samples : int, float, or 'auto', default='auto'
            Number of samples to draw from X to train each tree (min(256, n_samples)).
        random_state : int, default=42
            Seed for deterministic reproducibility.
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state

        # Instantiate scikit-learn IsolationForest
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.is_fitted = False
        self.feature_names = SENSOR_FEATURES

    def fit(self, df_or_records: Union[pd.DataFrame, List[dict]]) -> "IsolationForestDetector":
        """
        Fits the Isolation Forest on the 10 sensor feature columns.
        Ignores all metadata, timestamps, and target labels.
        """
        X = extract_sensor_features(df_or_records)
        self.model.fit(X)
        self.is_fitted = True
        return self

    def score_samples(self, df_or_records: Union[pd.DataFrame, List[dict], dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes anomaly predictions and continuous anomaly scores.

        Returns:
        --------
        is_anomaly : np.ndarray (bool)
            True if the sample is classified as an anomaly, False if normal.
        anomaly_scores : np.ndarray (float in [0.0, 1.0])
            Normalized continuous score where 0.0 = completely healthy/normal,
            and 1.0 = severe anomaly. The decision threshold sits at 0.50.
        raw_scores : np.ndarray (float)
            Raw decision_function output (negative for anomalies, positive for normal).
        """
        if not self.is_fitted:
            raise RuntimeError("IsolationForestDetector must be fitted before running inference.")

        X = extract_sensor_features(df_or_records)
        raw_decision = self.model.decision_function(X) # > 0 is normal, < 0 is anomaly
        raw_preds = self.model.predict(X)              # 1 is normal, -1 is anomaly

        is_anomaly = (raw_preds == -1)

        # Calibrate continuous anomaly score into intuitive [0.0, 1.0] range
        # When decision_function == 0.0, score = 0.50 (exact contamination decision boundary)
        # When decision_function > 0 (normal), score scales between 0.0 and 0.50
        # When decision_function < 0 (anomalous), score scales between 0.50 and 1.0
        # Formula: sigmoid-like mapping centered at decision boundary
        norm_scores = 1.0 / (1.0 + np.exp(raw_decision * 10.0))
        norm_scores = np.clip(norm_scores, 0.0, 1.0).round(4)

        return is_anomaly, norm_scores, raw_decision

    def predict(self, df_or_records: Union[pd.DataFrame, List[dict], dict]) -> pd.DataFrame:
        """
        Generates enriched output DataFrame containing:
          - anomaly_label: 'anomaly' or 'normal'
          - is_anomaly: bool
          - anomaly_score: float in [0.0, 1.0] (higher = more abnormal)
          - raw_score: float (decision function distance)
          - anomaly_severity: 'Low', 'Moderate', 'High', or 'Severe'
        """
        is_anomaly, norm_scores, raw_scores = self.score_samples(df_or_records)

        labels = ["anomaly" if a else "normal" for a in is_anomaly]

        # Severity categorization based on normalized anomaly score
        severities = []
        for s in norm_scores:
            if s >= 0.80:
                severities.append("Severe")
            elif s >= 0.65:
                severities.append("High")
            elif s >= 0.50:
                severities.append("Moderate")
            else:
                severities.append("Normal")

        result = pd.DataFrame({
            "anomaly_label": labels,
            "is_anomaly": is_anomaly,
            "anomaly_score": norm_scores,
            "raw_score": raw_scores.round(4),
            "anomaly_severity": severities
        })

        return result

    def save(self, filepath: str):
        """Serializes trained model artifact to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump({
            "model": self.model,
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted
        }, filepath)

    @classmethod
    def load(cls, filepath: str) -> "IsolationForestDetector":
        """Loads serialized model artifact from disk."""
        data = joblib.load(filepath)
        detector = cls(
            contamination=data.get("contamination", 0.05),
            n_estimators=data.get("n_estimators", 150),
            random_state=data.get("random_state", 42)
        )
        detector.model = data["model"]
        detector.feature_names = data.get("feature_names", SENSOR_FEATURES)
        detector.is_fitted = data.get("is_fitted", True)
        return detector


class FleetAnomalyDetector:
    """
    Fleet-Level Anomaly Detection Manager.
    
    Supports both:
    1. Per-Machine Models: Individual Isolation Forest trained on each machine's baseline data.
    2. Fleet-Wide Model: Unified fallback model trained across all fleet sensor distributions.
    """

    def __init__(self, contamination: float = 0.05, n_estimators: int = 150, random_state: int = 42):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state

        self.machine_detectors: Dict[str, IsolationForestDetector] = {}
        self.fleet_detector: Optional[IsolationForestDetector] = None

    def fit_fleet(self, df_or_records: Union[pd.DataFrame, List[dict]], machine_col: str = "Machine_ID"):
        """
        Trains both per-machine models (if machine_col is present) and a unified fleet-wide model.
        """
        if isinstance(df_or_records, list):
            df = pd.DataFrame(df_or_records)
        else:
            df = df_or_records.copy()

        # 1. Train Fleet-Wide Model (Fallback / Fleet Aggregate)
        print(f"Training Fleet-Wide Isolation Forest (contamination={self.contamination})...")
        self.fleet_detector = IsolationForestDetector(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state
        )
        self.fleet_detector.fit(df)

        # 2. Train Per-Machine Models if machine_col exists
        if machine_col in df.columns:
            unique_machines = df[machine_col].unique()
            print(f"Training Per-Machine Isolation Forests for {len(unique_machines)} machines...")
            for m_id in unique_machines:
                m_df = df[df[machine_col] == m_id]
                # Require at least 20 historical readings to fit machine-specific trees
                if len(m_df) >= 20:
                    det = IsolationForestDetector(
                        contamination=self.contamination,
                        n_estimators=self.n_estimators,
                        random_state=self.random_state + int(str(m_id).replace("M-", "").replace("-", "") or 0)
                    )
                    det.fit(m_df)
                    self.machine_detectors[str(m_id)] = det
                    print(f"  - Fitted per-machine model for {m_id} ({len(m_df)} samples)")

        return self

    def predict_reading(self, record: dict, machine_id: Optional[str] = None) -> dict:
        """
        Evaluates a single telemetry reading.
        Prioritizes per-machine model if available; falls back to fleet-wide model.
        """
        m_id = str(machine_id or record.get("Machine_ID", ""))
        detector = self.machine_detectors.get(m_id, self.fleet_detector)

        if detector is None:
            raise RuntimeError("FleetAnomalyDetector is not fitted. Call fit_fleet() first.")

        res_df = detector.predict(record)
        res = res_df.iloc[0].to_dict()
        res["model_type"] = "per_machine" if m_id in self.machine_detectors else "fleet_wide"
        res["machine_id"] = m_id
        return res

    def predict_fleet_batch(self, fleet_records: List[dict]) -> List[dict]:
        """
        Evaluates a batch of synchronized fleet records in one pass.
        Returns a list of enriched anomaly dictionaries.
        """
        results = []
        for rec in fleet_records:
            m_id = rec.get("Machine_ID")
            results.append(self.predict_reading(rec, machine_id=m_id))
        return results

    def save(self, directory: str):
        """Saves all fleet detectors to a directory."""
        os.makedirs(directory, exist_ok=True)
        if self.fleet_detector:
            self.fleet_detector.save(os.path.join(directory, "isolation_forest_fleet.pkl"))
        for m_id, det in self.machine_detectors.items():
            det.save(os.path.join(directory, f"isolation_forest_{m_id}.pkl"))

    @classmethod
    def load(cls, directory: str, contamination: float = 0.05) -> "FleetAnomalyDetector":
        """Loads fleet detectors from a directory."""
        fleet = cls(contamination=contamination)
        fleet_path = os.path.join(directory, "isolation_forest_fleet.pkl")
        if os.path.exists(fleet_path):
            fleet.fleet_detector = IsolationForestDetector.load(fleet_path)

        if os.path.exists(directory):
            for fname in os.listdir(directory):
                if fname.startswith("isolation_forest_M-") and fname.endswith(".pkl"):
                    m_id = fname.replace("isolation_forest_", "").replace(".pkl", "")
                    fleet.machine_detectors[m_id] = IsolationForestDetector.load(os.path.join(directory, fname))
        return fleet
