import os
import sys
import site

import numpy
import qiskit
import qiskit_ibm_runtime

from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.fake_provider import FakeManilaV2, FakeGuadalupeV2
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


def backend_supports_direct_state(backend, backend_name=None):
    """
    Report whether a backend allows direct statevector access (Aer only).

    Inputs:
    - backend: the backend object returned by build_backend.
    - backend_name: optional name string; when provided it is used instead of
      inspecting the backend object.

    Output:
    - True for true Aer simulator backends, False otherwise.
    """
    if backend_name is not None:
        name = str(backend_name).lower()
        # only allow true Aer simulator backends for direct state access
        return "aer" in name

    name_attr = getattr(backend, "name", None)
    if callable(name_attr):
        name_attr = name_attr()
    if name_attr is None:
        name_attr = str(backend)

    name_attr = str(name_attr).lower()
    return "aer" in name_attr


def build_backend(backend_name, ibm_service, ionq_provider, backend_options=None,
                  ibm_backend="ibm_kingston"):
    """
    Select the execution backend for a given backend_name and apply any
    backend-specific options.

    Inputs:
    - backend_name: one of "manila", "guadalupe", "aer", "ibm", "ionq_qpu",
      "ionq_noisy_sim", "ionq_simulator".
    - ibm_service, ionq_provider: service / provider handles from
      activate_backends() (only the one matching backend_name is used).
    - backend_options: optional dict applied verbatim via
      backend.set_options(**backend_options) -- e.g. the Aer MPS options, or
      {"noise_model": noise_model} for a noisy Aer sim. It is applied with no
      compatibility check, so an option the selected backend does not accept
      will raise.
    - ibm_backend: device selection for backend_name == "ibm". A device name
      string (default "ibm_kingston") selects ibm_service.backend(ibm_backend);
      the sentinel "least_busy" selects
      ibm_service.least_busy(operational=True, simulator=False). Ignored for
      non-IBM backends.

    Output:
    - backend.
    """
    if backend_name == 'manila':
        backend = FakeManilaV2()

    elif backend_name == 'guadalupe':
        backend = FakeGuadalupeV2()  # fake ibm backend but runs for uptil 16 qubits

    elif backend_name == 'aer':
        # method (e.g. "matrix_product_state") and any noise_model are supplied
        # by the caller through backend_options and applied via set_options below.
        backend = AerSimulator()

    elif backend_name == 'ibm':
        if ibm_backend == "least_busy":
            backend = ibm_service.least_busy(operational=True, simulator=False)
        else:
            backend = ibm_service.backend(ibm_backend)

    elif backend_name == 'ionq_qpu':
        backend = ionq_provider.get_backend("ionq_qpu")

    elif backend_name == 'ionq_noisy_sim':
        # IonQ simulator with Aria-style noise, using QIS gates
        # TODO: identical to ionq_simulator below -- the two branches used to
        # differ only in the (now removed) gateset variables, which nothing read.
        # Noise has to be requested explicitly of the IonQ simulator, e.g.
        # backend_options={"noise_model": "aria-1"}, so this currently runs ideal.
        backend = ionq_provider.get_backend("ionq_simulator")

    elif backend_name == 'ionq_simulator':
        # Ideal IonQ simulator using QIS gates
        backend = ionq_provider.get_backend("ionq_simulator")

    else:
        raise ValueError(f"Unsupported backend_name: {backend_name}")

    if backend_options:
        backend.set_options(**backend_options)

    return backend