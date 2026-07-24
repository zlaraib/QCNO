import os
import sys
import site
import qiskit
import qiskit_ibm_runtime

import numpy


from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ionq import IonQProvider

print("qiskit:", qiskit.__version__)
print("runtime:", qiskit_ibm_runtime.__version__)
print("numpy:", numpy.__version__)
print(qiskit_ibm_runtime.__file__)

# ============================================================
# Environment / package diagnostics
# ============================================================

print("=" * 70)
print("PYTHON / PACKAGE INFORMATION")
print("=" * 70)

print("Python executable:", sys.executable)
print("ENABLE_USER_SITE:", site.ENABLE_USER_SITE)
print("qiskit:", qiskit.__version__)
print("qiskit_ibm_runtime:", qiskit_ibm_runtime.__version__)
print("qiskit_ibm_runtime path:", qiskit_ibm_runtime.__file__)
print("HOME:", os.getenv("HOME"))
print("PWD:", os.getenv("PWD"))
print("CI:", os.getenv("CI"))
print("=" * 70)

# ============================================================
# Load credentials from environment
# ============================================================

IBM_AST244_TOKEN = os.getenv("IBM_QUANTUM_TOKEN")
IBM_AST244_CRN = os.getenv("IBM_AST244_CRN")
IONQ_AST244_API_KEY = os.getenv("IONQ_AST244_API_KEY")

HAS_IBM_CREDS = bool(IBM_AST244_TOKEN) and bool(IBM_AST244_CRN)
HAS_IONQ_CREDS = bool(IONQ_AST244_API_KEY)

print("\n" + "=" * 70)
print("CREDENTIAL CHECK")
print("=" * 70)

# Do NOT print actual tokens
print("IBM token loaded:", IBM_AST244_TOKEN is not None)
print("IBM CRN loaded:", IBM_AST244_CRN is not None)
print("IonQ key loaded:", IONQ_AST244_API_KEY is not None)

if os.getenv("CI"):
    print("Running in CI/Jenkins mode.")
else:
    print("Running in local/non-CI mode.")

print("=" * 70)

# ============================================================
# IBM AST244
# ============================================================

print("\n" + "=" * 70)
print("IBM AST244 BACKEND INFORMATION")
print("=" * 70)

if not HAS_IBM_CREDS:
    print("Skipping IBM AST244 connection.")
    print("Reason: IBM_QUANTUM_TOKEN and/or IBM_AST244_CRN is not available.")
    print("This is expected in Jenkins unless those credentials are added to Jenkins.")
else:
    try:
        ibm_service = QiskitRuntimeService(
            channel="ibm_cloud",
            token=IBM_AST244_TOKEN,
            instance=IBM_AST244_CRN,
        )

        ibm_backends = ibm_service.backends()

        print("✓ Connected to IBM AST244")
        print(f"✓ Available IBM backends: {len(ibm_backends)}\n")

        for backend in ibm_backends:
            try:
                status = backend.status()
                backend_name = backend.name if isinstance(backend.name, str) else backend.name()
                print(f"{backend_name:25s} operational={status.operational}")
            except Exception as e:
                backend_name = getattr(backend, "name", "unknown")
                if callable(backend_name):
                    backend_name = backend_name()
                print(f"{backend_name:25s} status unavailable: {type(e).__name__}")

    except Exception as e:
        print("✗ IBM connection error:")
        print(type(e).__name__, ":", e)

print("=" * 70)

# ============================================================
# IonQ AST244
# ============================================================

print("\n" + "=" * 70)
print("IONQ AST244 BACKEND INFORMATION")
print("=" * 70)

if not HAS_IONQ_CREDS:
    print("Skipping IonQ AST244 connection.")
    print("Reason: IONQ_AST244_API_KEY is not available.")
    print("This is expected in Jenkins unless that credential is added to Jenkins.")
else:
    try:
        provider = IonQProvider(IONQ_AST244_API_KEY)
        backends = provider.backends()

        print("✓ Connected to IonQ AST244")
        print(f"✓ Available IonQ backends: {len(backends)}\n")

        for backend in backends:
            ionq_backend_name = backend.name() if callable(backend.name) else backend.name

            print(f"Backend: {ionq_backend_name}")
            print(f"  Class: {type(backend).__name__}")

            try:
                status = backend.status()
                print(f"  Status: {status.status_msg}")
                print(f"  Operational: {status.operational}")
            except Exception as e:
                print(f"  Status unavailable: {type(e).__name__}: {e}")

            print()

    except Exception as e:
        print("✗ IonQ connection error:")
        print(type(e).__name__, ":", e)

print("=" * 70)