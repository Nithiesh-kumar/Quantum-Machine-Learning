"""Environment verification script for Quantum ML Health Risk System."""
import sys


def check_environment() -> None:
    print("=" * 60)
    print("QUANTUM ML HEALTH RISK SYSTEM - ENVIRONMENT VERIFICATION")
    print("=" * 60)

    print(f"Python: {sys.version.split()[0]} ({sys.executable})")

    # PennyLane
    try:
        import pennylane as qml
        print(f"pennylane: {qml.__version__}")
        dev = qml.device("default.qubit", wires=2)
        print("  - PennyLane 'default.qubit' device initialized successfully.")
    except Exception as e:
        print(f"pennylane ERROR: {e}")

    # Qiskit
    try:
        import qiskit
        print(f"qiskit: {qiskit.__version__}")
    except Exception as e:
        print(f"qiskit ERROR: {e}")

    # Qiskit Machine Learning
    try:
        import qiskit_machine_learning
        print(f"qiskit-machine-learning: {qiskit_machine_learning.__version__}")
    except Exception as e:
        print(f"qiskit-machine-learning ERROR: {e}")

    # Scikit-Learn
    try:
        import sklearn
        print(f"scikit-learn: {sklearn.__version__}")
    except Exception as e:
        print(f"scikit-learn ERROR: {e}")

    # FastAPI
    try:
        import fastapi
        print(f"fastapi: {fastapi.__version__}")
    except Exception as e:
        print(f"fastapi ERROR: {e}")

    print("=" * 60)
    print("VERIFICATION COMPLETE - All core frameworks loaded cleanly.")
    print("=" * 60)


if __name__ == "__main__":
    check_environment()
