"""Root conftest.py – add the repo root to sys.path.

This allows ``import circuit_lm`` to work in pytest without requiring the
package to be installed (useful for CI environments that run pytest directly
from a checkout).  When the package IS installed (``pip install -e .``),
this is a no-op.
"""

import sys
import pathlib

# Insert repo root at the front of sys.path if it is not already present
_repo_root = str(pathlib.Path(__file__).parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
