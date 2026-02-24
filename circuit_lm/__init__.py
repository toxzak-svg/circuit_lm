"""circuit_lm – finite-state circuit language model.

Constraints enforced throughout this package:
- Zero floating-point arithmetic (no float literals, no logarithm calls, no numpy).
- No tensor / matmul dependencies.
- Solver: OR-Tools CP-SAT (integer constraint programming only).
"""

__version__ = "dev"
