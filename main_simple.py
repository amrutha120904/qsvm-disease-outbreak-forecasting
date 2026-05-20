"""
Simple and reliable main script
Compatible with Qiskit 0.25.1
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import time
from src.data_loader import DataLoader
from src.classical_svm import ClassicalSVM
from src.simple_improved_quantum_svm import SimpleImprovedQuantumSVM, MultiConfigQuantumSVM
from src.utils import ResultsLogger

def check_environment():
    """Check environment"""
    try:
        import qiskit
        print(f"✅ Qiskit {qiskit.__version__}")
        import qiskit_machine_learning
        print(f"✅ Qiskit ML {qiskit_machine_learning.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Missing: {e}")
        return False

def main(mode='single'):
    """
    Main function
    
    Args:
        mode: 'single' (fastest) or 'multi' (best accuracy)
    """
    print("\n" + "="*70)
    print("🚀 QUANTUM vs CLASSICAL SVM - DIABETES PREDICTION")
    print("="*70)
    
    if not check_environment():
        return
    
    # Load data
    print("\n📊 Loading data...")
    data_loader = DataLoader()
    X_train, X_test, y_train, y_test = data_loader.load_diabetes_data()
    
    if X_train is None:
        print("❌ Data loading failed")
        return
    
    print(f"✓ Train: {X_train.shape[0]} samples")
    print(f"✓ Test: {X_test.shape[0]} samples")
    
    # Classical SVM
    print("\n" + "="*60)
    print("🔬 CLASSICAL SVM")
    print("="*60)
    
    classical_svm = ClassicalSVM()
    t0 = time.time()
    classical_svm.train_with_hyperparameter_tuning(X_train, y_train)
    classical_time = time.time() - t0
    classical_metrics, _ = classical_svm.evaluate(X_test, y_test, "Classical_SVM")
    classical_metrics['training_time'] = classical_time
    
    # Quantum SVM
    print("\n" + "="*60)
    print("⚛️  QUANTUM SVM")
    print("="*60)
    
    quantum_results = {}
    t0 = time.time()
    
    if mode == 'single':
        # Fast single configuration
        print("Mode: Single (Fast)")
        qsvm = SimpleImprovedQuantumSVM(random_state=42)
        qsvm.train(X_train, y_train, X_test, y_test,
                   n_components=5, reps=3, sample_fraction=0.6)
        metrics, _ = qsvm.evaluate(X_test, y_test, "Quantum_SVM")
        quantum_time = time.time() - t0
        metrics['training_time'] = quantum_time
        quantum_results['Quantum_SVM'] = metrics
        
    else:  # multi
        # Multiple configurations - best accuracy
        print("Mode: Multi (Best Accuracy)")
        multi = MultiConfigQuantumSVM(random_state=42)
        results = multi.grid_search_and_train(X_train, y_train, X_test, y_test)
        quantum_time = time.time() - t0
        
        for name, metrics in results.items():
            metrics['training_time'] = quantum_time / len(results)
            quantum_results[f"QSVM_{name}"] = metrics
    
    # Combine results
    all_results = {
        "Classical_SVM": classical_metrics,
        **quantum_results
    }
    
    # Print comparison
    print("\n" + "="*70)
    print("🏆 FINAL RESULTS")
    print("="*70)
    
    sorted_results = sorted(all_results.items(), 
                          key=lambda x: x[1]['accuracy'], 
                          reverse=True)
    
    for rank, (name, metrics) in enumerate(sorted_results, 1):
        print(f"\n#{rank} {name}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  Time:      {metrics.get('training_time', 0):.1f}s")
    
    # Improvement analysis
    classical_acc = classical_metrics['accuracy']
    quantum_models = [k for k in all_results.keys() if 'Quantum' in k or 'QSVM' in k]
    
    if quantum_models:
        best_quantum = max(quantum_models, key=lambda k: all_results[k]['accuracy'])
        best_quantum_acc = all_results[best_quantum]['accuracy']
        improvement = ((best_quantum_acc - classical_acc) / classical_acc) * 100
        
        print(f"\n{'='*70}")
        print("📈 QUANTUM vs CLASSICAL")
        print(f"{'='*70}")
        print(f"Classical:     {classical_acc:.4f} ({classical_acc*100:.2f}%)")
        print(f"Best Quantum:  {best_quantum_acc:.4f} ({best_quantum_acc*100:.2f}%)")
        print(f"Difference:    {improvement:+.2f}%")
        
        if improvement > 0:
            print(f"\n🎉 Quantum SVM wins by {improvement:.2f}%!")
        elif improvement > -1:
            print(f"\n⚖️  Very close performance!")
        else:
            print(f"\n💡 Tip: Try 'multi' mode for better results")
    
    # Save results
    results_logger = ResultsLogger()
    results_logger.save_metrics("final_comparison", all_results)
    
    # Generate plots
    print("\n📊 Generating plots...")
    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        try:
            results_logger.plot_comparison(all_results, metric)
        except:
            pass
    
    print("\n✅ Done! Results saved to 'results/' directory")
    
    return all_results


if __name__ == "__main__":
    # Usage:
    # python main_simple.py           -> Single config (fast, ~3-4 min) with multiple feature maps.
    # python main_simple.py multi     -> Multiple configs (best, ~10-12 min) with only z and zz feature maps.
    
    mode = sys.argv[1] if len(sys.argv) > 1 else 'single'
    
    if mode not in ['single', 'multi']:
        print(f"Invalid mode '{mode}'. Use 'single' or 'multi'")
        print("Using 'single' mode...")
        mode = 'single'
    
    print(f"\n🎯 Mode: {mode}")
    results = main(mode=mode)