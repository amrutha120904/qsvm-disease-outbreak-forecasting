import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
import os

def download_diabetes_data():
    """Download and save Pima Indians Diabetes Dataset"""
    # Using a direct download link for the diabetes dataset
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    
    # Column names for the dataset
    column_names = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
    ]
    
    try:
        df = pd.read_csv(url, names=column_names)
        os.makedirs('../data', exist_ok=True)
        df.to_csv('../data/diabetes.csv', index=False)
        print(f"Diabetes dataset downloaded with {len(df)} samples")
        return df
    except Exception as e:
        print(f"Error downloading data: {e}")
        # Create a sample dataset if download fails
        return create_sample_data()

def create_sample_data():
    """Create sample diabetes-like data if download fails"""
    np.random.seed(42)
    n_samples = 768
    
    data = {
        'Pregnancies': np.random.randint(0, 17, n_samples),
        'Glucose': np.random.normal(120, 30, n_samples).astype(int),
        'BloodPressure': np.random.normal(70, 12, n_samples).astype(int),
        'SkinThickness': np.random.normal(20, 10, n_samples).astype(int),
        'Insulin': np.random.normal(80, 100, n_samples).astype(int),
        'BMI': np.random.normal(32, 8, n_samples),
        'DiabetesPedigreeFunction': np.random.uniform(0.08, 2.42, n_samples),
        'Age': np.random.randint(21, 81, n_samples),
        'Outcome': np.random.randint(0, 2, n_samples)
    }
    
    df = pd.DataFrame(data)
    # Clean negative values
    df['Glucose'] = df['Glucose'].clip(lower=0)
    df['BloodPressure'] = df['BloodPressure'].clip(lower=0)
    df['SkinThickness'] = df['SkinThickness'].clip(lower=0)
    df['Insulin'] = df['Insulin'].clip(lower=0)
    df['BMI'] = df['BMI'].clip(lower=0)
    
    os.makedirs('../data', exist_ok=True)
    df.to_csv('../data/diabetes.csv', index=False)
    print(f"Sample diabetes dataset created with {len(df)} samples")
    return df

if __name__ == "__main__":
    download_diabetes_data()