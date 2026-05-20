import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import time
from src.data_loader import DataLoader
from src.classical_svm import ClassicalSVM
from src.quantum_svm import QuantumSVM
from src.utils import ResultsLogger

def check_environment():
    """Check if all required packages are installed"""
    try:
        import qiskit
        import qiskit_machine_learning
        from qiskit import Aer
        print(f"✅ Qiskit {qiskit.__version__} installed")
        print(f"✅ Qiskit ML {qiskit_machine_learning.__version__} installed")
        
        # Test critical imports
        from qiskit.circuit.library import ZZFeatureMap
        from qiskit_machine_learning.kernels import QuantumKernel
        from qiskit_machine_learning.algorithms import QSVC
        print("✅ All critical imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("\nPlease run: python clean_install.py")
        return False
    except Exception as e:
        print(f"❌ Import error: {e}")
        print("\nPlease run: python clean_install.py")
        return False

def run_classical_svm(X_train, X_test, y_train, y_test):
    """Run classical SVM"""
    print("\n🔬 Training Classical SVM...")
    classical_svm = ClassicalSVM()
    
    # Use simple training for faster results
    start_time = time.time()
    classical_svm.train_simple(X_train, y_train, kernel='rbf', C=1.0, gamma='scale')
    classical_time = time.time() - start_time
    
    classical_metrics, classical_pred = classical_svm.evaluate(X_test, y_test, "Classical_SVM_RBF")
    classical_metrics['training_time'] = classical_time
    
    return classical_metrics

def run_quantum_svm(X_train, X_test, y_train, y_test):
    """Run quantum SVM"""
    print("\n⚛️ Training Quantum SVM...")
    quantum_svm = QuantumSVM()
    
    # Train with different feature maps (using smaller sample for speed)
    start_time = time.time()
    quantum_results = quantum_svm.experiment_with_feature_maps(
        X_train, y_train, X_test, y_test, sample_fraction=0.3
    )
    quantum_time = time.time() - start_time
    
    # Add training time to quantum results
    for model_name in quantum_results:
        quantum_results[model_name]['training_time'] = quantum_time / len(quantum_results) if quantum_results else quantum_time
    
    return quantum_results

def main():
    print("🚀 Quantum vs Classical SVM for Disease Prediction")
    print("=" * 60)
    
    # Check environment first
    if not check_environment():
        return
    
    # Initialize components
    data_loader = DataLoader()
    results_logger = ResultsLogger()
    
    # Load and preprocess data
    print("\n📊 Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = data_loader.load_diabetes_data()
    
    if X_train is None:
        print("❌ Failed to load data. Exiting.")
        return
    
    # Run Classical SVM
    classical_metrics = run_classical_svm(X_train, X_test, y_train, y_test)
    
    # Run Quantum SVM
    quantum_results = run_quantum_svm(X_train, X_test, y_train, y_test)
    
    # Combine all results
    all_results = {
        "Classical_SVM_RBF": classical_metrics,
        **quantum_results
    }
    
    # Plot comparison
    print("\n📈 Generating comparison plots...")
    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        results_logger.plot_comparison(all_results, metric)
    
    # Print final comparison
    print("\n🏆 FINAL COMPARISON")
    print("=" * 60)
    for model_name, metrics in all_results.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  Time:      {metrics.get('training_time', 'N/A'):.2f}s")
    
    # Save all results
    results_logger.save_metrics("all_models_comparison", all_results)
    print(f"\n✅ Results saved to 'results/' directory")

if __name__ == "__main__":
    main()