"""
P-SVM Attack (AdaBoost Implementation)

Paper: "I Can Hear Your Alexa: Voice Command Fingerprinting on Smart Home Speakers"

The paper authors initially tried SVM but it failed (1.2% accuracy on page 235).
They switched to AdaBoost with decision stumps as weak learners and achieved
33.4% accuracy. This implementation follows the AdaBoost approach.

Algorithm:
    1. Extract 15 statistical + burst-based features from each traffic trace
    2. Train AdaBoost classifier with decision stumps (max_depth=1)
    3. No feature scaling needed (tree-based ensemble)
"""

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from typing import List, Tuple


class PSVM:
    """
    Panchenko-style SVM Attack using AdaBoost with Decision Stumps.

    Extracts 15 features from traffic traces and classifies using AdaBoost.
    Decision stumps (single-split trees) serve as weak learners.

    Features extracted (15 total):
        0  total_packets      - Number of packets in trace
        1  total_bytes        - Sum of all packet lengths
        2  incoming_bytes     - Sum of incoming packet lengths
        3  outgoing_bytes     - Sum of outgoing packet lengths
        4  incoming_packets   - Count of incoming packets
        5  outgoing_packets   - Count of outgoing packets
        6  pct_incoming       - Fraction of incoming packets
        7  num_bursts         - Number of traffic bursts
        8  duration           - Time span of trace (seconds)
        9  avg_packet_size    - Mean packet length
        10 std_packet_size    - Std dev of packet lengths
        11 max_packet_size    - Maximum packet length
        12 min_packet_size    - Minimum packet length
        13 avg_burst_size     - Mean burst size
        14 std_burst_size     - Std dev of burst sizes

    Attributes:
        model: AdaBoostClassifier instance
        trained: Whether the model has been fitted
        n_estimators: Number of boosting rounds
    """

    NUM_FEATURES = 15

    FEATURE_NAMES = [
        'total_packets', 'total_bytes', 'incoming_bytes', 'outgoing_bytes',
        'incoming_packets', 'outgoing_packets', 'pct_incoming', 'num_bursts',
        'duration', 'avg_packet_size', 'std_packet_size', 'max_packet_size',
        'min_packet_size', 'avg_burst_size', 'std_burst_size'
    ]

    def __init__(self, n_estimators: int = 50, random_state: int = 42):
        """
        Initialize P-SVM attack with AdaBoost classifier.

        Args:
            n_estimators: Number of boosting rounds (weak learners).
            random_state: Random seed for reproducibility.
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=n_estimators,
            random_state=random_state,
            algorithm='SAMME'
        )
        self.trained = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X_train: List, y_train: np.ndarray) -> None:
        """
        Train the AdaBoost model on traffic traces.

        Args:
            X_train: List of traffic traces. Each trace is a list of
                     (timestamp, packet_length, direction) tuples.
            y_train: Numpy array of integer class labels.
        """
        print(f"Training P-SVM (AdaBoost, n_estimators={self.n_estimators})...")

        total = len(X_train)
        features_list: List[np.ndarray] = []

        for i, trace in enumerate(X_train):
            features_list.append(self._extract_features(trace))
            if (i + 1) % 100 == 0 or (i + 1) == total:
                print(f"  Extracted features for {i + 1}/{total} traces")

        X_features = np.array(features_list)
        print(f"  Feature matrix shape: {X_features.shape}")

        self.model.fit(X_features, y_train)
        self.trained = True
        print("  Training complete")

    def predict(self, X_test: List) -> np.ndarray:
        """
        Predict class labels for test traces.

        Args:
            X_test: List of traffic traces.

        Returns:
            Numpy array of predicted class labels.

        Raises:
            ValueError: If model has not been trained.
        """
        if not self.trained:
            raise ValueError("Model not trained. Call fit() first.")

        print(f"Predicting {len(X_test)} test samples...")

        features_list: List[np.ndarray] = []
        for trace in X_test:
            features_list.append(self._extract_features(trace))

        X_features = np.array(features_list)
        return self.model.predict(X_features)

    def score(self, X_test: List, y_test: np.ndarray) -> float:
        """
        Calculate classification accuracy on test data.

        Args:
            X_test: List of test traces.
            y_test: True class labels.

        Returns:
            Accuracy as a float in [0, 1].
        """
        predictions = self.predict(X_test)
        return float(np.mean(predictions == y_test))

    def get_feature_importance(self) -> np.ndarray:
        """
        Return per-feature importance scores from the trained model.

        Returns:
            Numpy array of shape (15,) with importance values.

        Raises:
            ValueError: If model has not been trained.
        """
        if not self.trained:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.feature_importances_

    # ------------------------------------------------------------------
    # Feature Extraction (private)
    # ------------------------------------------------------------------

    def _extract_features(self, trace: List) -> np.ndarray:
        """
        Extract 15 features from a single traffic trace.

        Args:
            trace: List of (timestamp, packet_length, direction) tuples.
                   direction is +1 (outgoing) or -1 (incoming).

        Returns:
            Numpy array of shape (15,).
        """
        if not trace:
            return np.zeros(self.NUM_FEATURES)

        timestamps = [pkt[0] for pkt in trace]
        sizes = [pkt[1] for pkt in trace]
        directions = [pkt[2] for pkt in trace]

        total_packets = len(trace)
        total_bytes = sum(sizes)

        incoming_bytes = sum(s for s, d in zip(sizes, directions) if d == -1)
        outgoing_bytes = sum(s for s, d in zip(sizes, directions) if d == 1)
        incoming_packets = sum(1 for d in directions if d == -1)
        outgoing_packets = sum(1 for d in directions if d == 1)

        pct_incoming = incoming_packets / total_packets if total_packets > 0 else 0.0

        # Burst extraction
        bursts = self._extract_bursts(trace)
        num_bursts = len(bursts)

        # Duration
        duration = timestamps[-1] - timestamps[0] if total_packets > 1 else 0.0

        # Packet size statistics
        sizes_arr = np.array(sizes, dtype=float)
        avg_packet_size = float(np.mean(sizes_arr))
        std_packet_size = float(np.std(sizes_arr)) if total_packets > 1 else 0.0
        max_packet_size = int(np.max(sizes_arr))
        min_packet_size = int(np.min(sizes_arr))

        # Burst size statistics
        if num_bursts > 0:
            burst_sizes = np.array([b[0] for b in bursts], dtype=float)
            avg_burst_size = float(np.mean(burst_sizes))
            std_burst_size = float(np.std(burst_sizes)) if num_bursts > 1 else 0.0
        else:
            avg_burst_size = 0.0
            std_burst_size = 0.0

        return np.array([
            total_packets,
            total_bytes,
            incoming_bytes,
            outgoing_bytes,
            incoming_packets,
            outgoing_packets,
            pct_incoming,
            num_bursts,
            duration,
            avg_packet_size,
            std_packet_size,
            max_packet_size,
            min_packet_size,
            avg_burst_size,
            std_burst_size,
        ], dtype=float)

    @staticmethod
    def _extract_bursts(trace: List) -> List[Tuple[int, int]]:
        """
        Extract traffic bursts from a trace.

        A burst is a maximal sequence of consecutive packets travelling in the
        same direction.  Each burst is represented as (total_bytes, direction).

        Args:
            trace: List of (timestamp, packet_length, direction) tuples.

        Returns:
            List of (burst_size, direction) tuples.
        """
        if not trace:
            return []

        bursts: List[Tuple[int, int]] = []
        current_size = trace[0][1]
        current_dir = trace[0][2]

        for pkt in trace[1:]:
            _, size, direction = pkt
            if direction == current_dir:
                current_size += size
            else:
                bursts.append((current_size, current_dir))
                current_size = size
                current_dir = direction

        # Save final burst
        bursts.append((current_size, current_dir))
        return bursts


# ======================================================================
# Standalone test
# ======================================================================

if __name__ == "__main__":
    from src.data_loader import TrafficDataLoader

    # Load data
    loader = TrafficDataLoader('data/trace_csv')
    X, y, command_names = loader.load_all_traces(max_commands=10)
    X_train, X_test, y_train, y_test = loader.split_train_test(X, y)

    # Train model
    model = PSVM(n_estimators=50)
    model.fit(X_train, y_train)

    # Test accuracy
    accuracy = model.score(X_test, y_test)
    print(f"\nP-SVM (AdaBoost) Accuracy: {accuracy:.1%}")

    # Show feature importance
    importance = model.get_feature_importance()
    feature_names = [
        'total_packets', 'total_bytes', 'incoming_bytes', 'outgoing_bytes',
        'incoming_packets', 'outgoing_packets', 'pct_incoming', 'num_bursts',
        'duration', 'avg_packet_size', 'std_packet_size', 'max_packet_size',
        'min_packet_size', 'avg_burst_size', 'std_burst_size'
    ]

    print("\nTop 5 Most Important Features:")
    indices = np.argsort(importance)[::-1][:5]
    for idx in indices:
        print(f"  {feature_names[idx]}: {importance[idx]:.3f}")
