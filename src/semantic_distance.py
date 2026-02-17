"""
Semantic Distance Calculator
Uses doc2vec to measure semantic similarity between voice commands
"""

import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from typing import List
import os


class SemanticDistanceCalculator:
    """
    Calculate semantic distance between voice commands using doc2vec
    """

    def __init__(self, model_path: str = None):
        """
        Initialize calculator

        Args:
            model_path: Path to pre-trained model (optional)
        """
        self.model = None
        self.model_path = model_path

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def train_doc2vec(self, command_texts: List[str],
                     vector_size: int = 300,
                     epochs: int = 100,
                     save_path: str = None):
        """
        Train doc2vec model on voice commands

        Args:
            command_texts: List of voice command strings
            vector_size: Dimension of semantic vectors
            epochs: Number of training epochs
            save_path: Path to save trained model
        """
        print(f"Training doc2vec model...")
        print(f"  Commands: {len(command_texts)}")
        print(f"  Vector size: {vector_size}")
        print(f"  Epochs: {epochs}")

        # Prepare documents — convert underscores to spaces for tokenization
        documents = []
        for idx, text in enumerate(command_texts):
            # Replace underscores with spaces and tokenize
            cleaned = text.replace('_', ' ').replace("'", '')
            tokens = cleaned.lower().split()
            documents.append(TaggedDocument(tokens, [idx]))

        print(f"  Documents prepared: {len(documents)}")

        # Train model
        self.model = Doc2Vec(
            documents,
            vector_size=vector_size,
            window=5,
            min_count=1,
            workers=4,
            epochs=epochs
        )

        print(f"  Training complete!")

        if save_path:
            self.save_model(save_path)
            print(f"  Model saved to: {save_path}")

    def get_vector(self, command_text: str) -> np.ndarray:
        """
        Get semantic vector for a command

        Args:
            command_text: Voice command string

        Returns:
            Semantic vector
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_doc2vec() first.")

        cleaned = command_text.replace('_', ' ').replace("'", '')
        tokens = cleaned.lower().split()
        return self.model.infer_vector(tokens)

    def semantic_similarity(self, command1: str, command2: str) -> float:
        """
        Calculate cosine similarity between two commands

        Args:
            command1: First command
            command2: Second command

        Returns:
            Similarity in [-1, 1]
        """
        vec1 = self.get_vector(command1)
        vec2 = self.get_vector(command2)

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        return float(similarity)

    def normalized_semantic_distance(self, true_command: str,
                                     predicted_command: str,
                                     all_commands: List[str]) -> int:
        """
        Calculate normalized semantic distance (rank-based)

        Args:
            true_command: Ground truth command
            predicted_command: Predicted command
            all_commands: All possible commands

        Returns:
            Rank (0 = perfect, higher = worse)
        """
        similarities = []
        for cmd in all_commands:
            sim = self.semantic_similarity(true_command, cmd)
            similarities.append((sim, cmd))

        # Sort descending by similarity
        similarities.sort(key=lambda x: x[0], reverse=True)

        # Find rank of predicted command
        for rank, (sim, cmd) in enumerate(similarities):
            if cmd == predicted_command:
                return rank

        return len(all_commands) - 1

    def save_model(self, path: str):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)

    def load_model(self, path: str):
        """Load trained model"""
        self.model = Doc2Vec.load(path)
        print(f"Model loaded from: {path}")


if __name__ == "__main__":
    sample_commands = [
        "what_is_the_weather",
        "whats_the_weather_for_sunday",
        "tell_me_a_joke",
        "what_time_is_it",
        "set_a_timer_for_thirty_seconds",
        "play_npr",
        "how_old_are_you",
        "tell_me_a_fun_fact",
        "what_is_in_the_news",
        "good_morning"
    ]

    print("Training doc2vec on sample commands...")
    calc = SemanticDistanceCalculator()
    calc.train_doc2vec(sample_commands, vector_size=50, epochs=50)

    print("\nSemantic Similarities:")
    pairs = [
        ("what_is_the_weather", "whats_the_weather_for_sunday"),
        ("what_is_the_weather", "tell_me_a_joke"),
        ("what_is_in_the_news", "tell_me_a_fun_fact"),
        ("what_time_is_it", "set_a_timer_for_thirty_seconds")
    ]

    for cmd1, cmd2 in pairs:
        sim = calc.semantic_similarity(cmd1, cmd2)
        print(f"  '{cmd1}' <-> '{cmd2}': {sim:.3f}")
