"""
Evaluation Module
Comprehensive evaluation metrics and visualization
"""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import pandas as pd


class Evaluator:
    """
    Evaluate fingerprinting attacks
    """

    def __init__(self, command_names: List[str],
                 semantic_calc=None):
        """
        Initialize evaluator

        Args:
            command_names: List of command names
            semantic_calc: SemanticDistanceCalculator (optional)
        """
        self.command_names = command_names
        self.semantic_calc = semantic_calc

    def evaluate_attack(self, y_true: np.ndarray,
                       y_pred: np.ndarray,
                       attack_name: str) -> Dict:
        """
        Comprehensive evaluation of an attack

        Args:
            y_true: True labels
            y_pred: Predicted labels
            attack_name: Name of attack

        Returns:
            Dictionary with metrics
        """
        results = {
            'attack_name': attack_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'num_samples': len(y_true),
            'num_correct': int(np.sum(y_true == y_pred)),
            'num_incorrect': int(np.sum(y_true != y_pred))
        }

        # Add semantic metrics if available
        if self.semantic_calc is not None:
            try:
                avg_sim = self._calculate_avg_semantic_similarity(y_true, y_pred)
                avg_norm_dist = self._calculate_avg_normalized_distance(y_true, y_pred)
                results['avg_semantic_similarity'] = avg_sim
                results['avg_normalized_distance'] = avg_norm_dist
            except Exception as e:
                print(f"  Warning: Could not compute semantic metrics: {e}")

        return results

    def _calculate_avg_semantic_similarity(self, y_true: np.ndarray,
                                           y_pred: np.ndarray) -> float:
        """Calculate average semantic similarity"""
        similarities = []

        for true_label, pred_label in zip(y_true, y_pred):
            if true_label < len(self.command_names) and pred_label < len(self.command_names):
                true_cmd = self.command_names[true_label]
                pred_cmd = self.command_names[pred_label]
                sim = self.semantic_calc.semantic_similarity(true_cmd, pred_cmd)
                similarities.append(sim)

        return np.mean(similarities) if similarities else 0

    def _calculate_avg_normalized_distance(self, y_true: np.ndarray,
                                           y_pred: np.ndarray) -> float:
        """Calculate average normalized semantic distance"""
        norm_distances = []

        for true_label, pred_label in zip(y_true, y_pred):
            if true_label < len(self.command_names) and pred_label < len(self.command_names):
                true_cmd = self.command_names[true_label]
                pred_cmd = self.command_names[pred_label]
                norm_dist = self.semantic_calc.normalized_semantic_distance(
                    true_cmd, pred_cmd, self.command_names
                )
                norm_distances.append(norm_dist)

        return np.mean(norm_distances) if norm_distances else 0

    def plot_confusion_matrix(self, y_true: np.ndarray,
                             y_pred: np.ndarray,
                             attack_name: str,
                             save_path: str = None,
                             top_n: int = 20):
        """
        Plot confusion matrix

        Args:
            y_true: True labels
            y_pred: Predicted labels
            attack_name: Attack name
            save_path: Path to save figure
            top_n: Show top N classes
        """
        if len(self.command_names) > top_n:
            unique, counts = np.unique(y_true, return_counts=True)
            top_indices = unique[np.argsort(counts)[-top_n:]]

            mask = np.isin(y_true, top_indices)
            y_true_filtered = y_true[mask]
            y_pred_filtered = y_pred[mask]

            cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=top_indices)
            labels = [self.command_names[i][:20] if i < len(self.command_names) else str(i)
                     for i in top_indices]
        else:
            cm = confusion_matrix(y_true, y_pred)
            labels = [name[:20] for name in self.command_names]

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix - {attack_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right', fontsize=7)
        plt.yticks(rotation=0, fontsize=7)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"  Saved confusion matrix to: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_comparison(self, results_dict: Dict[str, Dict],
                       save_path: str = None):
        """
        Plot comparison of all attacks vs paper results

        Args:
            results_dict: Dictionary mapping attack names to results
            save_path: Path to save figure
        """
        attacks = list(results_dict.keys())
        accuracies = [results_dict[a]['accuracy'] for a in attacks]

        paper_results = {
            'LL-Jaccard': 0.174,
            'LL-NB': 0.338,
            'VNG++': 0.249,
            'P-SVM': 0.334
        }

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(attacks))
        width = 0.35

        bars1 = ax.bar(x - width/2, accuracies, width, label='My Results',
                      color='#2196F3')

        paper_acc = [paper_results.get(a, 0) for a in attacks]
        bars2 = ax.bar(x + width/2, paper_acc, width, label='Paper Results',
                      color='#FF9800')

        ax.set_xlabel('Attack')
        ax.set_ylabel('Accuracy')
        ax.set_title('Attack Performance Comparison: My Results vs Paper')
        ax.set_xticks(x)
        ax.set_xticklabels(attacks)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1%}',
                       ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"  Saved comparison plot to: {save_path}")
        else:
            plt.show()

        plt.close()

    def generate_report(self, results_dict: Dict[str, Dict],
                       save_path: str = None) -> pd.DataFrame:
        """
        Generate comparison table

        Args:
            results_dict: Dictionary mapping attack names to results
            save_path: Path to save CSV

        Returns:
            DataFrame with results
        """
        rows = []

        paper_results = {
            'LL-Jaccard': {'accuracy': 0.174, 'norm_dist': 46.99},
            'LL-NB': {'accuracy': 0.338, 'norm_dist': 34.11},
            'VNG++': {'accuracy': 0.249, 'norm_dist': 43.80},
            'P-SVM': {'accuracy': 0.334, 'norm_dist': 37.68}
        }

        for attack_name, results in results_dict.items():
            paper_acc = paper_results.get(attack_name, {}).get('accuracy', 0)
            row = {
                'Attack': attack_name,
                'My Accuracy': f"{results['accuracy']:.1%}",
                'Paper Accuracy': f"{paper_acc:.1%}",
                'Accuracy Diff': f"{abs(results['accuracy'] - paper_acc):.1%}"
            }

            if 'avg_normalized_distance' in results:
                paper_nd = paper_results.get(attack_name, {}).get('norm_dist', 0)
                row['My Norm. Dist.'] = f"{results['avg_normalized_distance']:.1f}"
                row['Paper Norm. Dist.'] = f"{paper_nd:.1f}"

            rows.append(row)

        df = pd.DataFrame(rows)

        if save_path:
            df.to_csv(save_path, index=False)
            print(f"  Saved report to: {save_path}")

        return df

    def print_results(self, results: Dict):
        """Print formatted results"""
        print(f"\n{'='*60}")
        print(f"Attack: {results['attack_name']}")
        print(f"{'='*60}")
        print(f"Accuracy: {results['accuracy']:.1%}")
        print(f"Correct: {results['num_correct']}/{results['num_samples']}")

        if 'avg_semantic_similarity' in results:
            print(f"Avg Semantic Similarity: {results['avg_semantic_similarity']:.3f}")
            print(f"Avg Normalized Distance: {results['avg_normalized_distance']:.1f}")

        print(f"{'='*60}\n")


if __name__ == "__main__":
    command_names = [f"Command_{i}" for i in range(10)]
    evaluator = Evaluator(command_names)

    np.random.seed(42)
    y_true = np.random.randint(0, 10, 100)
    y_pred = y_true.copy()
    y_pred[np.random.choice(100, 30, replace=False)] = np.random.randint(0, 10, 30)

    results = evaluator.evaluate_attack(y_true, y_pred, "Test Attack")
    evaluator.print_results(results)
