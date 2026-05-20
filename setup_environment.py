import subprocess
import sys
import os

def run_command(command):
    """Run a shell command and print output"""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    else:
        print(f"Success: {result.stdout}")
    return result.returncode

def main():
    print("🚀 Performing clean installation with compatible Qiskit versions...")
    
    # First, let's clean up the current environment
    print("\n1. Cleaning current Qiskit installations...")
    packages_to_remove = [
        "qiskit", "qiskit-aer", "qiskit-machine-learning", 
        "qiskit-ibm-runtime", "qiskit-terra", "qiskit-aqua"
    ]
    
    for package in packages_to_remove:
        run_command(f"pip uninstall -y {package}")
    
    # Install compatible versions in correct order
    print("\n2. Installing compatible Qiskit versions...")
    compatible_packages = [
        "numpy==1.23.5",
        "pandas==1.5.3",
        "scikit-learn==1.2.2", 
        "matplotlib==3.6.3",
        "seaborn==0.12.2",
        "qiskit==0.44.1",
        "qiskit-aer==0.12.2",
        "qiskit-machine-learning==0.6.1",
        "jupyter==1.0.0"
    ]
    
    for package in compatible_packages:
        run_command(f"pip install {package}")
    
    print("\n3. Testing installation...")
    test_code = """
try:
    import qiskit
    import qiskit_machine_learning
    from qiskit import Aer
    print(f"✅ Qiskit version: {qiskit.__version__}")
    print(f"✅ Qiskit ML version: {qiskit_machine_learning.__version__}")
    
    # Test key imports
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit_machine_learning.kernels import QuantumKernel
    from qiskit_machine_learning.algorithms import QSVC
    print("✅ All imports successful!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
"""
    
    with open("test_imports.py", "w") as f:
        f.write(test_code)
    
    result = run_command("python test_imports.py")
    os.remove("test_imports.py")
    
    if result == 0:
        print("\n🎉 Installation completed successfully!")
    else:
        print("\n❌ Installation failed!")

if __name__ == "__main__":
    main()