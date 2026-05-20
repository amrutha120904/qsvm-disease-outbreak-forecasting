import qiskit
import qiskit_machine_learning
from qiskit_aer import AerSimulator

print(f"Qiskit version: {qiskit.__version__}")
print(f"Qiskit ML version: {qiskit_machine_learning.__version__}")

# Test basic functionality
from qiskit.circuit.library import ZZFeatureMap
feature_map = ZZFeatureMap(feature_dimension=2, reps=1)
print("✅ ZZFeatureMap created successfully!")

backend = AerSimulator()
print("✅ AerSimulator created successfully!")

print("🎉 All tests passed! Your environment is ready.")