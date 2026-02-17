"""
Main Execution Script
Runs all attacks and generates complete results

Uses 5-fold stratified cross-validation matching the reference code,
and also runs a single 80/20 split for visualization.
"""

import numpy as np
import os
import sys
import time

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import TrafficDataLoader
from src.semantic_distance import SemanticDistanceCalculator
from src.attacks.ll_jaccard import LLJaccard
from src.attacks.ll_nb import LLNB
from src.attacks.vng_plus import VNGPlus
from src.attacks.p_svm import PSVM
from src.evaluation import Evaluator


def run_single_split(X_train, X_test, y_train, y_test, command_names, evaluator):
    """Run all attacks on a single train/test split"""

    all_results = {}
    all_predictions = {}

    # 1: LL-Jaccard
    print("\n  [1/4] LL-Jaccard Attack")
    print("  " + "-" * 76)
    ll_jaccard = LLJaccard()
    ll_jaccard.fit(X_train, y_train)
    pred_jaccard = ll_jaccard.predict(X_test)
    results_jaccard = evaluator.evaluate_attack(y_test, pred_jaccard, "LL-Jaccard")
    all_results['LL-Jaccard'] = results_jaccard
    all_predictions['LL-Jaccard'] = pred_jaccard
    evaluator.print_results(results_jaccard)

    # 2: LL-NB
    print("\n  [2/4] LL-NB Attack")
    print("  " + "-" * 76)
    ll_nb = LLNB(interval=100)
    ll_nb.fit(X_train, y_train)
    pred_nb = ll_nb.predict(X_test)
    results_nb = evaluator.evaluate_attack(y_test, pred_nb, "LL-NB")
    all_results['LL-NB'] = results_nb
    all_predictions['LL-NB'] = pred_nb
    evaluator.print_results(results_nb)

    # 3: VNG++
    print("\n  [3/4] VNG++ Attack")
    print("  " + "-" * 76)
    vng = VNGPlus(interval=5000)
    vng.fit(X_train, y_train)
    pred_vng = vng.predict(X_test)
    results_vng = evaluator.evaluate_attack(y_test, pred_vng, "VNG++")
    all_results['VNG++'] = results_vng
    all_predictions['VNG++'] = pred_vng
    evaluator.print_results(results_vng)

    # 4: P-SVM (AdaBoost)
    print("\n  [4/4] P-SVM Attack")
    print("  " + "-" * 76)
    psvm = PSVM(interval=5000, n_estimators=200)
    psvm.fit(X_train, y_train)
    pred_psvm = psvm.predict(X_test)
    results_psvm = evaluator.evaluate_attack(y_test, pred_psvm, "P-SVM")
    all_results['P-SVM'] = results_psvm
    all_predictions['P-SVM'] = pred_psvm
    evaluator.print_results(results_psvm)

    return all_results, all_predictions


def run_cross_validation(X, y, n_folds=5):
    """
    Run n-fold stratified cross-validation matching reference code

    Returns average accuracy per attack
    """
    from sklearn.model_selection import StratifiedKFold
    from src.feature_extraction import FeatureExtractor

    print(f"\n{'='*80}")
    print(f" RUNNING {n_folds}-FOLD CROSS-VALIDATION")
    print(f"{'='*80}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    attack_accuracies = {
        'LL-Jaccard': [],
        'LL-NB': [],
        'VNG++': [],
        'P-SVM': []
    }

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold+1}/{n_folds} ---")

        X_train = [X[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        # LL-Jaccard
        jac = LLJaccard()
        jac.fit(X_train, y_train)
        acc = jac.score(X_test, y_test)
        attack_accuracies['LL-Jaccard'].append(acc)
        print(f"  LL-Jaccard:  {acc:.1%}")

        # LL-NB
        nb = LLNB(interval=100)
        nb.fit(X_train, y_train)
        acc = nb.score(X_test, y_test)
        attack_accuracies['LL-NB'].append(acc)
        print(f"  LL-NB:       {acc:.1%}")

        # VNG++
        vng = VNGPlus(interval=5000)
        vng.fit(X_train, y_train)
        acc = vng.score(X_test, y_test)
        attack_accuracies['VNG++'].append(acc)
        print(f"  VNG++:       {acc:.1%}")

        # P-SVM
        psvm = PSVM(interval=5000, n_estimators=200)
        psvm.fit(X_train, y_train)
        acc = psvm.score(X_test, y_test)
        attack_accuracies['P-SVM'].append(acc)
        print(f"  P-SVM:       {acc:.1%}")

    # Print cross-validation summary
    print(f"\n{'='*80}")
    print(f" CROSS-VALIDATION RESULTS (avg of {n_folds} folds)")
    print(f"{'='*80}")
    print(f"{'Algorithm':<15} {'Avg Accuracy':<15} {'Std Dev':<15} {'Per-fold'}")
    print("-" * 80)

    avg_results = {}
    for attack, accs in attack_accuracies.items():
        avg = np.mean(accs)
        std = np.std(accs)
        fold_str = ', '.join(f'{a:.1%}' for a in accs)
        print(f"{attack:<15} {avg:<15.1%} {std:<15.3f} [{fold_str}]")
        avg_results[attack] = {'accuracy': avg, 'std': std}

    return avg_results


def main():
    print("=" * 80)
    print(" VOICE COMMAND FINGERPRINTING ATTACK - REPRODUCTION")
    print(' Paper: "I Can Hear Your Alexa" (IEEE CNS 2019)')
    print("=" * 80)

    start_time = time.time()

    # Create results directory
    os.makedirs('results/figures', exist_ok=True)

    # === STEP 1: Load Data ===
    print("\n[STEP 1/6] Loading Dataset...")
    print("-" * 80)

    loader = TrafficDataLoader('data/trace_csv')
    X, y, command_names = loader.load_all_traces()

    stats = loader.get_statistics(X, y)
    print(f"Dataset loaded successfully!")
    print(f"  Total traces: {stats['total_traces']}")
    print(f"  Commands: {stats['num_commands']}")
    print(f"  Avg packets/trace: {stats['avg_packets_per_trace']:.0f}")

    # === STEP 2: Train Doc2vec ===
    print("\n[STEP 2/6] Training Doc2vec Model...")
    print("-" * 80)

    model_path = 'data/doc2vec_models/commands_model.bin'
    os.makedirs('data/doc2vec_models', exist_ok=True)

    if os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}")
        semantic_calc = SemanticDistanceCalculator(model_path)
    else:
        print("Training new doc2vec model...")
        semantic_calc = SemanticDistanceCalculator()
        semantic_calc.train_doc2vec(
            command_names,
            vector_size=300,
            epochs=100,
            save_path=model_path
        )

    # === STEP 3: Run Cross-Validation ===
    cv_results = run_cross_validation(X, y, n_folds=5)

    # === STEP 4: Run Single Split for Visualizations ===
    print(f"\n{'='*80}")
    print(f" RUNNING SINGLE 80/20 SPLIT (for visualizations)")
    print(f"{'='*80}")

    X_train, X_test, y_train, y_test = loader.split_train_test(X, y, train_ratio=0.8)

    evaluator = Evaluator(command_names, semantic_calc)
    all_results, all_predictions = run_single_split(
        X_train, X_test, y_train, y_test, command_names, evaluator
    )

    # === STEP 5: Generate Visualizations ===
    print("\n[STEP 5/6] Generating Visualizations...")
    print("-" * 80)

    for attack_name, predictions in all_predictions.items():
        safe_name = attack_name.replace('+', 'plus').replace(' ', '_')
        evaluator.plot_confusion_matrix(
            y_test, predictions, attack_name,
            save_path=f'results/figures/confusion_matrix_{safe_name}.png'
        )

    evaluator.plot_comparison(
        all_results,
        save_path='results/figures/comparison.png'
    )

    # === STEP 6: Generate Report ===
    print("\n[STEP 6/6] Generating Report...")
    print("-" * 80)

    # Use CV results for the report (more reliable)
    cv_report_results = {}
    for attack_name, cv_res in cv_results.items():
        cv_report_results[attack_name] = {
            'accuracy': cv_res['accuracy'],
            'attack_name': attack_name,
            'num_samples': len(y),
            'num_correct': int(cv_res['accuracy'] * len(y)),
            'num_incorrect': int((1 - cv_res['accuracy']) * len(y))
        }

    report_df = evaluator.generate_report(
        cv_report_results,
        save_path='results/comparison_table.csv'
    )

    print("\nComparison Table (Cross-Validation):")
    print(report_df.to_string(index=False))

    # === Final Summary ===
    elapsed = time.time() - start_time

    print(f"\n{'='*80}")
    print(f" FINAL SUMMARY")
    print(f"{'='*80}")

    print("\nRESULTS COMPARISON (5-fold CV averages):")
    print("-" * 80)
    print(f"{'Algorithm':<15} {'My Accuracy':<15} {'Paper Accuracy':<15} {'Difference':<15}")
    print("-" * 80)

    paper_results = {
        'LL-Jaccard': 0.174,
        'LL-NB': 0.338,
        'VNG++': 0.249,
        'P-SVM': 0.334
    }

    for attack_name, cv_res in cv_results.items():
        my_acc = cv_res['accuracy']
        paper_acc = paper_results.get(attack_name, 0)
        diff = abs(my_acc - paper_acc)
        status = "PASS" if diff <= 0.05 else "REVIEW"
        print(f"{attack_name:<15} {my_acc:<15.1%} {paper_acc:<15.1%} {diff:<15.1%} {status}")

    print("-" * 80)
    print(f"\nTotal execution time: {elapsed:.1f} seconds")

    print("\n" + "=" * 80)
    print(" REPRODUCTION COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - results/comparison_table.csv")
    print("  - results/figures/confusion_matrix_*.png")
    print("  - results/figures/comparison.png")
    print("  - data/doc2vec_models/commands_model.bin")


if __name__ == "__main__":
    main()
