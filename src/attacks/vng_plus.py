"""
VNG++ Attack
Matching reference code: trainByVNGpp.py + testByVNGpp.py

Key algorithm:
1. Calculate bursts (signed burst sizes = sum of size*direction per burst)
2. Bin bursts into histogram [-400000, 400001, interval]
3. Prepend [TotalTraceTime, UpStreamTotal, DownStreamTotal]
4. Classifier: sklearn GaussianNB
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB
from typing import List


class VNGPlus:
    """
    Variable N-Gram++ Attack

    Uses sklearn GaussianNB on burst histogram features + statistics
    """

    def __init__(self, interval: int = 3000, var_smoothing: float = 1e-9):
        """
        Initialize VNG++ attack

        Args:
            interval: Burst histogram bin width (optimal=3000 from sweep)
            var_smoothing: GaussianNB variance smoothing parameter
        """
        self.interval = interval
        self.model = GaussianNB(var_smoothing=var_smoothing)
        self.trained = False

    def fit(self, X_train: List, y_train: np.ndarray):
        """
        Train GaussianNB on VNG++ features

        Args:
            X_train: List of training traces
            y_train: Training labels
        """
        from src.feature_extraction import FeatureExtractor

        print(f"Training VNG++ (GaussianNB, interval={self.interval})...")

        X_features = []
        for trace in X_train:
            features = FeatureExtractor.extract_vng_features(trace, self.interval)
            X_features.append(features)

        X_features = np.array(X_features, dtype=float)

        print(f"  Feature matrix shape: {X_features.shape}")

        self.model.fit(X_features, y_train)

        self.trained = True
        print(f"  Training complete")

    def predict(self, X_test: List) -> np.ndarray:
        """
        Predict using trained GaussianNB

        Args:
            X_test: List of test traces

        Returns:
            Predicted labels
        """
        if not self.trained:
            raise ValueError("Model not trained. Call fit() first.")

        from src.feature_extraction import FeatureExtractor

        print(f"Predicting {len(X_test)} test samples...")

        X_features = []
        for trace in X_test:
            features = FeatureExtractor.extract_vng_features(trace, self.interval)
            X_features.append(features)

        X_features = np.array(X_features, dtype=float)

        predictions = self.model.predict(X_features)
        return predictions

    def score(self, X_test: List, y_test: np.ndarray) -> float:
        """Calculate accuracy"""
        predictions = self.predict(X_test)
        return np.mean(predictions == y_test)


if __name__ == "__main__":
    from src.data_loader import TrafficDataLoader

    loader = TrafficDataLoader('data/trace_csv')
    X, y, command_names = loader.load_all_traces(max_commands=10)
    X_train, X_test, y_train, y_test = loader.split_train_test(X, y)

    model = VNGPlus(interval=5000)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"\nVNG++ Accuracy: {accuracy:.1%}")
