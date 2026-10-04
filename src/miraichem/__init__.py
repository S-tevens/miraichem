"""MiraiChem: VQE configuration benchmarking for noisy IBM backends."""

import warnings

from scipy.sparse import SparseEfficiencyWarning

# Qiskit's UCCSD operator evolution triggers a harmless scipy sparse-format notice on every call.
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

__version__ = "0.1.0"
