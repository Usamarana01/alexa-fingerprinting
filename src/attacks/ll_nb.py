"""
LL-NB (Bayes) Attack
Matching reference code: trainByBayes.py + testByBayes.py

Key algorithm:
1. Features: size * direction for each packet -> histogram bins [-1500, 1501, interval]
2. Classifier: sklearn GaussianNB on histogram feature vectors
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB
from typing import List


class LLNB:
    """
    Liberatore & Levine - Naive Bayes Attack

    Uses sklearn GaussianNB on histogram-binned signed packet sizes
    """

    def __init__(self, interval: int = 100):
        """
        Initialize LL-NB attack

        Args:
            interval: Histogram bin width (paper tests 10-100, default 100)
        """
        self.interval = interval
        self.model = GaussianNB()
        self.trained = False

    def fit(self, X_train: List, y_train: np.ndarray):
        """
        Train GaussianNB on histogram features

        Args:
            X_train: List of training traces
            y_train: Training labels
        """
        from src.feature_extraction import FeatureExtractor

        print(f"Training LL-NB (GaussianNB, interval={self.interval})...")

        # Extract histogram features for all training traces
        X_features = []
        for trace in X_train:
            features = FeatureExtractor.extract_bayes_features(trace, self.interval)
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
            features = FeatureExtractor.extract_bayes_features(trace, self.interval)
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

    model = LLNB(interval=100)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"\nLL-NB Accuracy: {accuracy:.1%}")
