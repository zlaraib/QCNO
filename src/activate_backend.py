import os
import sys
import site

import numpy
import qiskit
import qiskit_ibm_runtime

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ionq import IonQProvider


def activate_backends():
    """
    Connect to the IBM and IonQ services when credentials are available.

    Returns
    -------
    ibm_service
        QiskitRuntimeService instance, or None if unavailable.
    ionq_provider
        IonQProvider instance, or None if unavailable.
    """
    print("=" * 70)
    print("PYTHON / PACKAGE INFORMATION")
    print("=" * 70)
    print("Python executable:", sys.executable)
    print("ENABLE_USER_SITE:", site.ENABLE_USER_SITE)
    print("qiskit:", qiskit.__version__)
    print("qiskit_ibm_runtime:", qiskit_ibm_runtime.__version__)
    print("numpy:", numpy.__version__)
    print("qiskit_ibm_runtime path:", qiskit_ibm_runtime.__file__)
    print("CI:", os.getenv("CI"))
    print("=" * 70)

    ibm_token = os.getenv("IBM_QUANTUM_TOKEN")
    ibm_crn = os.getenv("IBM_AST244_CRN")
    ionq_api_key = os.getenv("IONQ_AST244_API_KEY")

    print("IBM token loaded:", ibm_token is not None)
    print("IBM CRN loaded:", ibm_crn is not None)
    print("IonQ key loaded:", ionq_api_key is not None)

    ibm_service = None
    ionq_provider = None

    # IBM connection
    if ibm_token and ibm_crn:
        try:
            ibm_service = QiskitRuntimeService(
                channel="ibm_cloud",
                token=ibm_token,
                instance=ibm_crn,
            )

            ibm_backends = ibm_service.backends()

            print("✓ Connected to IBM AST244")
            print(f"✓ Available IBM backends: {len(ibm_backends)}")

            for backend in ibm_backends:
                backend_name = (
                    backend.name()
                    if callable(backend.name)
                    else backend.name
                )

                try:
                    status = backend.status()
                    print(
                        f"{backend_name:25s} "
                        f"operational={status.operational}"
                    )
                except Exception as exc:
                    print(
                        f"{backend_name:25s} status unavailable: "
                        f"{type(exc).__name__}"
                    )

        except Exception as exc:
            print(
                f"✗ IBM connection error: "
                f"{type(exc).__name__}: {exc}"
            )
    else:
        print("Skipping IBM: credentials are unavailable.")

    # IonQ connection
    if ionq_api_key:
        try:
            ionq_provider = IonQProvider(ionq_api_key)
            ionq_backends = ionq_provider.backends()

            print("✓ Connected to IonQ AST244")
            print(f"✓ Available IonQ backends: {len(ionq_backends)}")

            for backend in ionq_backends:
                backend_name = (
                    backend.name()
                    if callable(backend.name)
                    else backend.name
                )
                print(f"IonQ backend: {backend_name}")

        except Exception as exc:
            print(
                f"✗ IonQ connection error: "
                f"{type(exc).__name__}: {exc}"
            )
    else:
        print("Skipping IonQ: API key is unavailable.")

    return ibm_service, ionq_provider