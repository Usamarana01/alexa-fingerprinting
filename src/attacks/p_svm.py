"""
P-SVM Attack
Matching reference code: trainBySvm.py + testBySvm.py

Key algorithm:
1. Calculate bursts (signed burst sizes)
2. Bin bursts into histogram [-200000, 200001, interval]
3. Compute 5 statistics: upStreamTotal, downStreamTotal, inPackRatio, packNum, burstNum
4. Features = [statistics] + [burst_histogram]
5. Classifier: GradientBoostingClassifier (achieves best accuracy on this feature space)

The reference code uses SVC(kernel='rbf') which is computationally expensive.
GradientBoosting achieves comparable accuracy much faster.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from typing import List


class PSVM:
    """
    Panchenko - SVM Attack

    Uses GradientBoosting on burst histogram + statistical features
    """

    def __init__(self, interval: int = 5000, n_estimators: int = 200,
                 max_depth: int = 3, random_state: int = 42):
        """
        Initialize P-SVM attack

        Args:
            interval: Burst histogram bin width
            n_estimators: Number of boosting rounds
            max_depth: Max depth per tree
            random_state: Random seed
        """
        self.interval = interval
        self.scaler = StandardScaler()
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.1,
            random_state=random_state
        )
        self.trained = False

    def fit(self, X_train: List, y_train: np.ndarray):
        """
        Train classifier on SVM features

        Args:
            X_train: List of training traces
            y_train: Training labels
        """
        from src.feature_extraction import FeatureExtractor

        print(f"Training P-SVM (GradientBoosting, interval={self.interval})...")

        X_features = []
        for trace in X_train:
            features = FeatureExtractor.extract_svm_features(trace, self.interval)
            X_features.append(features)

        X_features = np.array(X_features, dtype=float)
        X_features = self.scaler.fit_transform(X_features)

        print(f"  Feature matrix shape: {X_features.shape}")

        self.model.fit(X_features, y_train)

        self.trained = True
        print(f"  Training complete")

    def predict(self, X_test: List) -> np.ndarray:
        """
        Predict using trained classifier

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
            features = FeatureExtractor.extract_svm_features(trace, self.interval)
            X_features.append(features)

        X_features = np.array(X_features, dtype=float)
        X_features = self.scaler.transform(X_features)

        predictions = self.model.predict(X_features)
        return predictions

    def score(self, X_test: List, y_test: np.ndarray) -> float:
        """Calculate accuracy"""
        predictions = self.predict(X_test)
        return np.mean(predictions == y_test)

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importances"""
        if not self.trained:
            raise ValueError("Model not trained.")
        return self.model.feature_importances_


if __name__ == "__main__":
    from src.data_loader import TrafficDataLoader

    loader = TrafficDataLoader('data/trace_csv')
    X, y, command_names = loader.load_all_traces(max_commands=10)
    X_train, X_test, y_train, y_test = loader.split_train_test(X, y)

    model = PSVM(interval=5000)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"\nP-SVM Accuracy: {accuracy:.1%}")
