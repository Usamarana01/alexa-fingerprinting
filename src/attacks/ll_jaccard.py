"""
LL-Jaccard Attack
Matching reference code: trainByJaccard.py + testByJaccard.py

Key algorithm:
1. Training: For each class, create a PROTOTYPE set by majority vote
   (a feature must appear in >= 50% of training traces to be included)
2. Testing: Compare test trace set against each class prototype using
   Jaccard similarity. Predict class with highest Jaccard score.
"""

import numpy as np
from collections import defaultdict
import math
from typing import List


class LLJaccard:
    """
    Liberatore & Levine - Jaccard Similarity Attack

    Uses majority-vote prototype sets per class (from reference code)
    """

    def __init__(self):
        self.class_prototypes = {}  # class_label -> prototype set
        self.trained = False

    def fit(self, X_train: List, y_train: np.ndarray):
        """
        Train by creating majority-vote prototype sets per class

        Reference: trainByJaccard.trainFromList()
        For each class, union all training trace sets, then keep only
        features that appear in >= ceil(N/2) of the training traces.

        Args:
            X_train: List of training traces
            y_train: Training labels
        """
        from src.feature_extraction import FeatureExtractor

        print(f"Training LL-Jaccard (majority-vote prototypes)...")

        # Group traces by class
        class_traces = defaultdict(list)
        for trace, label in zip(X_train, y_train):
            features = FeatureExtractor.extract_ll_features(trace)
            class_traces[label].append(features)

        # Create prototype for each class
        self.class_prototypes = {}

        for label, feature_sets in class_traces.items():
            if len(feature_sets) == 1:
                # Single trace: use it directly
                self.class_prototypes[label] = feature_sets[0]
            else:
                # Majority vote: keep features appearing in >= ceil(N/2) traces
                threshold = math.ceil(len(feature_sets) / 2)

                # Union of all features
                all_features = set()
                for fs in feature_sets:
                    all_features = all_features.union(fs)

                # Keep only majority features
                prototype = set()
                for feature in all_features:
                    count = sum(1 for fs in feature_sets if feature in fs)
                    if count >= threshold:
                        prototype.add(feature)

                self.class_prototypes[label] = prototype

        self.trained = True
        print(f"  Created {len(self.class_prototypes)} class prototypes")

    @staticmethod
    def jaccard_similarity(set1: set, set2: set) -> float:
        """Jaccard distance: |A ∩ B| / |A ∪ B|"""
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        if union == 0:
            return 0.0
        return intersection / union

    def predict(self, X_test: List) -> np.ndarray:
        """
        Predict by comparing test trace set to each class prototype

        Reference: testByJaccard.computeLabel()

        Args:
            X_test: List of test traces

        Returns:
            Predicted labels
        """
        if not self.trained:
            raise ValueError("Model not trained. Call fit() first.")

        from src.feature_extraction import FeatureExtractor

        print(f"Predicting {len(X_test)} test samples...")

        predictions = []

        for i, test_trace in enumerate(X_test):
            test_features = FeatureExtractor.extract_ll_features(test_trace)

            max_sim = -1
            best_label = 0

            for label, prototype in self.class_prototypes.items():
                sim = self.jaccard_similarity(test_features, prototype)
                if sim > max_sim:
                    max_sim = sim
                    best_label = label

            predictions.append(best_label)

            if (i + 1) % 50 == 0:
                print(f"  Predicted {i+1}/{len(X_test)} samples")

        return np.array(predictions)

    def score(self, X_test: List, y_test: np.ndarray) -> float:
        """Calculate accuracy"""
        predictions = self.predict(X_test)
        return np.mean(predictions == y_test)


if __name__ == "__main__":
    from src.data_loader import TrafficDataLoader

    loader = TrafficDataLoader('data/trace_csv')
    X, y, command_names = loader.load_all_traces(max_commands=10)
    X_train, X_test, y_train, y_test = loader.split_train_test(X, y)

    model = LLJaccard()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"\nLL-Jaccard Accuracy: {accuracy:.1%}")
