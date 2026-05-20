"""
model_saver.py
Utility to save the best QSVM model after training with enhanced debugging
"""

import joblib
import os
import sys
import numpy as np
import pandas as pd
from sklearn.utils import resample

# Add the current directory to path to fix imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.data_loader import DataLoader
    from src.simple_improved_quantum_svm import MultiConfigQuantumSVM
except ImportError as e:
    print(f"Import error: {e}")
    print("Trying alternative import...")
    # Try direct import
    from data_loader import DataLoader
    from simple_improved_quantum_svm import MultiConfigQuantumSVM

def balance_training_data(X_train, y_train):
    """Balance the training data by oversampling the minority class"""
    # Combine features and labels
    train_data = pd.DataFrame(X_train)
    train_data['target'] = y_train
    
    # Separate classes
    majority_class = train_data[train_data['target'] == 0]
    minority_class = train_data[train_data['target'] == 1]
    
    print(f"📊 Before balancing:")
    print(f"   Non-Diabetic (0): {len(majority_class)} samples")
    print(f"   Diabetic (1): {len(minority_class)} samples")
    
    # Oversample minority class
    minority_oversampled = resample(
        minority_class,
        replace=True,  # sample with replacement
        n_samples=len(majority_class),  # match majority class size
        random_state=42
    )
    
    # Combine majority class with oversampled minority class
    balanced_data = pd.concat([majority_class, minority_oversampled])
    
    # Shuffle the data
    balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split back into features and labels
    X_balanced = balanced_data.drop('target', axis=1).values
    y_balanced = balanced_data['target'].values
    
    print(f"📊 After balancing:")
    print(f"   Non-Diabetic (0): {np.sum(y_balanced == 0)} samples")
    print(f"   Diabetic (1): {np.sum(y_balanced == 1)} samples")
    
    return X_balanced, y_balanced

def save_best_model(use_balanced_data=True):
    """Train and save the best QSVM model with optional data balancing"""
    print("🔮 Training and saving the best QSVM model...")
    
    # Load data
    data_loader = DataLoader()
    X_train, X_test, y_train, y_test = data_loader.load_diabetes_data()
    
    if X_train is None:
        print("❌ Failed to load data")
        return None
    
    print(f"✓ Original training data: {X_train.shape}")
    print(f"✓ Test data: {X_test.shape}")
    
    # Balance the training data if requested
    if use_balanced_data:
        print("\n⚖️  Balancing training data...")
        X_train_balanced, y_train_balanced = balance_training_data(X_train, y_train)
    else:
        X_train_balanced, y_train_balanced = X_train, y_train
    
    # Train multiple configurations and get the best one
    multi = MultiConfigQuantumSVM(random_state=42)
    results = multi.grid_search_and_train(X_train_balanced, y_train_balanced, X_test, y_test)
    
    if multi.best_model_instance:
        # Create results directory
        os.makedirs('results', exist_ok=True)
        
        # Save the best model
        model_path = 'results/best_qsvm_model.pkl'
        joblib.dump(multi.best_model_instance, model_path)
        
        # Save model info with additional metadata
        info = {
            'best_config_name': multi.best_config_name,
            'best_f1_score': float(multi.best_f1_score),
            'used_balanced_data': use_balanced_data,
            'all_results': results
        }
        joblib.dump(info, 'results/model_info.pkl')
        
        print(f"\n✅ Best model saved: {model_path}")
        print(f"✅ Best configuration: {multi.best_config_name}")
        print(f"✅ Best F1-Score: {multi.best_f1_score:.4f}")
        print(f"✅ Used balanced data: {use_balanced_data}")
        
        # Test the model on the original test set to verify it works
        print(f"\n🧪 Testing saved model on test set...")
        metrics, y_pred = multi.best_model_instance.evaluate(X_test, y_test, "Final_Model_Test", verbose=False)
        
        print(f"📊 Final Model Performance:")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        print(f"   F1-Score: {metrics['f1_score']:.4f}")
        
        # Check prediction distribution
        unique, counts = np.unique(y_pred, return_counts=True)
        print(f"📈 Prediction distribution on test set:")
        for cls, count in zip(unique, counts):
            class_name = "Non-Diabetic" if cls == 0 else "Diabetic"
            print(f"   {class_name}: {count} predictions")
        
        return multi.best_model_instance
    else:
        print("❌ No best model found!")
        return None

if __name__ == "__main__":
    # Ask user if they want to use balanced data
    print("QSVM Model Trainer")
    print("1. Use balanced training data (recommended for biased models)")
    print("2. Use original training data")
    
    choice = input("Enter your choice (1 or 2): ").strip()
    use_balanced = choice == "1"
    
    if use_balanced:
        print("🎯 Using balanced training data to fix prediction bias...")
    else:
        print("🎯 Using original training data...")
    
    save_best_model(use_balanced_data=use_balanced)