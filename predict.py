"""
predict.py
Loads the best QSVM model and provides interactive predictions with bias detection
"""

import numpy as np
import pandas as pd
import joblib
import os
import sys

# Add the current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from src.data_loader import DataLoader
    from src.simple_improved_quantum_svm import SimpleImprovedQuantumSVM
except ImportError:
    # Fallback to direct imports
    from data_loader import DataLoader
    from simple_improved_quantum_svm import SimpleImprovedQuantumSVM

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

class QSVMPredictor:
    """
    Loads and uses the best QSVM model for predictions with bias detection
    """
    
    def __init__(self, results_dir='results'):
        self.results_dir = results_dir
        self.best_model = None
        self.best_config = None
        self.feature_names = None
        self.scaler = None
        self.pca = None
        self.n_components = None
        self.model_info = None
        self.prediction_history = []
        
    def load_best_model(self, model_file='best_qsvm_model.pkl'):
        """Load the best model from file"""
        try:
            model_path = os.path.join(self.results_dir, model_file)
            if os.path.exists(model_path):
                self.best_model = joblib.load(model_path)
                
                # Load model info
                info_path = os.path.join(self.results_dir, 'model_info.pkl')
                if os.path.exists(info_path):
                    self.model_info = joblib.load(info_path)
                    print(f"✅ Model info: {self.model_info.get('best_config_name', 'Unknown')}")
                    print(f"✅ Best F1-Score: {self.model_info.get('best_f1_score', 0):.4f}")
                    if self.model_info.get('used_balanced_data'):
                        print("✅ Model trained with balanced data")
                
                # Extract preprocessing objects from the model
                if hasattr(self.best_model, 'pca'):
                    self.pca = self.best_model.pca
                    self.n_components = self.pca.n_components_
                    print(f"✅ Loaded PCA with {self.n_components} components")
                
                if hasattr(self.best_model, 'scaler'):
                    self.scaler = self.best_model.scaler
                    print("✅ Loaded scaler")
                
                print("✅ Best QSVM model loaded successfully!")
                return True
            else:
                print(f"❌ Best model file not found at: {model_path}")
                print("💡 Please run model_saver.py first to train and save the model")
                return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def load_training_config(self):
        """Load training configuration and feature names"""
        try:
            # Load dataset to get feature names
            data_loader = DataLoader()
            X_train, X_test, y_train, y_test = data_loader.load_diabetes_data()
            
            if X_train is None:
                print("❌ Failed to load training data")
                return False
            
            self.feature_names = [
                'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
            ]
            
            print(f"✅ Training configuration loaded! Expected features: {len(self.feature_names)}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading training config: {e}")
            return False
    
    def preprocess_input(self, input_features):
        """Preprocess user input same as training data"""
        if self.pca is None or self.scaler is None:
            raise ValueError("PCA or Scaler not loaded. Please load model first.")
        
        # Convert to numpy array and ensure correct shape
        input_array = np.array(input_features).reshape(1, -1)
        
        # Apply PCA
        input_pca = self.pca.transform(input_array)
        
        # Apply scaling
        input_scaled = self.scaler.transform(input_pca)
        
        return input_scaled
    
    def predict_single(self, input_features):
        """Make prediction on single input"""
        if self.best_model is None:
            print("❌ Model not loaded. Please load model first.")
            return None
        
        try:
            # Validate input length
            expected_features = self.pca.n_features_in_ if self.pca else len(self.feature_names)
            if len(input_features) != expected_features:
                raise ValueError(f"Expected {expected_features} features, but got {len(input_features)}")
            
            # Preprocess input
            processed_input = self.preprocess_input(input_features)
            
            # Make prediction
            prediction = self.best_model.model.predict(processed_input)[0]
            prediction_proba = self.get_prediction_confidence(processed_input)
            
            # Track prediction history for bias detection
            self.prediction_history.append(prediction)
            
            result = {
                'prediction': int(prediction),
                'class_name': 'Diabetic' if prediction == 1 else 'Non-Diabetic',
                'confidence': prediction_proba,
                'input_features': input_features
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None
    
    def get_prediction_confidence(self, processed_input):
        """
        Get prediction confidence (approximate for QSVM)
        """
        try:
            # For QSVM, we can use decision function values as confidence proxy
            if hasattr(self.best_model.model, 'decision_function'):
                decision_values = self.best_model.model.decision_function(processed_input)
                confidence = 1 / (1 + np.exp(-np.abs(decision_values[0])))
                return min(confidence, 0.99)  # Cap at 0.99
            else:
                return 0.75  # Default medium confidence
        except:
            return 0.75  # Default fallback
    
    def check_prediction_bias(self):
        """Check if the model is biased in its predictions"""
        if len(self.prediction_history) < 3:
            return None
        
        diabetic_count = sum(1 for p in self.prediction_history if p == 1)
        total_predictions = len(self.prediction_history)
        diabetic_ratio = diabetic_count / total_predictions
        
        if diabetic_ratio > 0.8:  # If more than 80% predictions are diabetic
            return {
                'biased': True,
                'message': f"⚠️  Warning: Model shows bias toward 'Diabetic' predictions ({diabetic_count}/{total_predictions} = {diabetic_ratio:.1%})",
                'diabetic_ratio': diabetic_ratio
            }
        elif diabetic_ratio < 0.2:  # If less than 20% predictions are diabetic
            return {
                'biased': True,
                'message': f"⚠️  Warning: Model shows bias toward 'Non-Diabetic' predictions ({diabetic_count}/{total_predictions} = {diabetic_ratio:.1%})",
                'diabetic_ratio': diabetic_ratio
            }
        else:
            return {
                'biased': False,
                'message': f"✅ Model predictions are balanced ({diabetic_count}/{total_predictions} = {diabetic_ratio:.1%} diabetic)",
                'diabetic_ratio': diabetic_ratio
            }
    
    def display_prediction(self, prediction_result):
        """Display prediction results in user-friendly format"""
        if prediction_result is None:
            return
        
        print("\n" + "="*50)
        print("🎯 DIABETES PREDICTION RESULT")
        print("="*50)
        
        # Display input features
        print("\n📊 Input Features:")
        for i, (name, value) in enumerate(zip(self.feature_names, prediction_result['input_features'])):
            print(f"   {name}: {value}")
        
        # Display prediction
        print(f"\n🔮 Prediction: {prediction_result['class_name']}")
        print(f"📈 Confidence: {prediction_result['confidence']:.2%}")
        
        # Display bias warning if applicable
        bias_info = self.check_prediction_bias()
        if bias_info and bias_info['biased']:
            print(f"\n{bias_info['message']}")
            print("💡 Consider retraining with balanced data using model_saver.py")
        
        # Display interpretation
        if prediction_result['prediction'] == 1:
            print("💡 Recommendation: Please consult with a healthcare professional for further evaluation.")
        else:
            print("💡 Recommendation: Maintain healthy lifestyle with regular checkups.")
        
        print("="*50)

def get_user_input(feature_names, expected_count=8):
    """Get feature values from user with validation"""
    features = []
    
    print(f"\n📝 Please enter {expected_count} health metrics:")
    print("   (Enter numeric values only)")
    
    # Use the actual feature names from the dataset
    diabetes_features = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]
    
    for i in range(expected_count):
        feature_name = diabetes_features[i]
        
        while True:
            try:
                # Get the appropriate prompt based on feature
                if feature_name == 'Pregnancies':
                    value = float(input(f"   {feature_name} (0-20): "))
                    if not (0 <= value <= 20):
                        print("     Please enter a value between 0-20")
                        continue
                        
                elif feature_name == 'Glucose':
                    value = float(input(f"   {feature_name} level (mg/dL) (0-200): "))
                    if not (0 <= value <= 200):
                        print("     Please enter a value between 0-200")
                        continue
                        
                elif feature_name == 'BloodPressure':
                    value = float(input(f"   {feature_name} (mm Hg) (0-150): "))
                    if not (0 <= value <= 150):
                        print("     Please enter a value between 0-150")
                        continue
                        
                elif feature_name == 'SkinThickness':
                    value = float(input(f"   {feature_name} (mm) (0-100): "))
                    if not (0 <= value <= 100):
                        print("     Please enter a value between 0-100")
                        continue
                        
                elif feature_name == 'Insulin':
                    value = float(input(f"   {feature_name} (mu U/ml) (0-900): "))
                    if not (0 <= value <= 900):
                        print("     Please enter a value between 0-900")
                        continue
                        
                elif feature_name == 'BMI':
                    value = float(input(f"   {feature_name} (kg/m²) (0-70): "))
                    if not (0 <= value <= 70):
                        print("     Please enter a value between 0-70")
                        continue
                        
                elif feature_name == 'DiabetesPedigreeFunction':
                    value = float(input(f"   {feature_name} (0-2.5): "))
                    if not (0 <= value <= 2.5):
                        print("     Please enter a value between 0-2.5")
                        continue
                        
                elif feature_name == 'Age':
                    value = float(input(f"   {feature_name} (years) (20-100): "))
                    if not (20 <= value <= 100):
                        print("     Please enter a value between 20-100")
                        continue
                
                # If we get here, the input is valid
                features.append(value)
                break
                
            except ValueError:
                print("     ❌ Please enter a valid number!")
            except KeyboardInterrupt:
                print("\n\n👋 Exiting...")
                return None
    
    return features

def interactive_prediction_mode():
    """Run interactive prediction mode"""
    predictor = QSVMPredictor()
    
    print("\n" + "="*60)
    print("🔮 QUANTUM SVM DIABETES PREDICTION SYSTEM")
    print("="*60)
    
    # Load model and configuration
    if not predictor.load_training_config():
        return
    
    if not predictor.load_best_model():
        print("💡 Please run model_saver.py first to train and save the best model!")
        return
    
    # Determine expected feature count
    expected_features = predictor.pca.n_features_in_ if predictor.pca else len(predictor.feature_names)
    print(f"✅ System ready! Expected features: {expected_features}")
    
    while True:
        try:
            print("\n" + "-"*50)
            user_input = input("\nPress Enter to make a prediction or 'quit' to exit: ")
            
            if user_input.lower() == 'quit':
                # Show final bias analysis
                bias_info = predictor.check_prediction_bias()
                if bias_info:
                    print(f"\n📊 Final Prediction Analysis: {bias_info['message']}")
                print("👋 Thank you for using Quantum SVM Prediction System!")
                break
            
            # Get user input
            input_features = get_user_input(predictor.feature_names, expected_features)
            
            if input_features is None:  # User pressed Ctrl+C
                break
            
            # Validate input length
            if len(input_features) != expected_features:
                print(f"❌ Error: Expected {expected_features} features, but got {len(input_features)}")
                print("💡 Please check your input and try again.")
                continue
            
            # Make prediction
            print("\n⚛️  Quantum SVM processing...")
            prediction_result = predictor.predict_single(input_features)
            
            if prediction_result:
                predictor.display_prediction(prediction_result)
            else:
                print("❌ Prediction failed. Please try again.")
            
        except KeyboardInterrupt:
            print("\n\n👋 Thank you for using Quantum SVM Prediction System!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    interactive_prediction_mode()