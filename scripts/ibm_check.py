"""Read-only IBM Quantum account check: lists backends, queue length and plan usage.

Reads IBM_QUANTUM_TOKEN / IBM_QUANTUM_INSTANCE from .env. Submits NO jobs, uses no QPU time.
"""

import os

from dotenv import load_dotenv
from qiskit_ibm_runtime import QiskitRuntimeService

load_dotenv()
token, instance = os.getenv("IBM_QUANTUM_TOKEN"), os.getenv("IBM_QUANTUM_INSTANCE")
if not token:
    raise SystemExit("IBM_QUANTUM_TOKEN missing: copy .env.example to .env and fill it in.")

service = QiskitRuntimeService(channel="ibm_cloud", token=token, instance=instance)
print("Connected. Instance:", (instance or "")[-40:])
for b in service.backends(operational=True, simulator=False):
    st = b.status()
    print(f"  {b.name:<20} qubits={b.num_qubits:<4} queue={st.pending_jobs}")
try:
    print("Usage:", service.usage())
except Exception as exc:  # usage API may be unavailable on some plans
    print("Usage unavailable:", type(exc).__name__, str(exc)[:200])
