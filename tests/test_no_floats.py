"""CI gate: static scan for floating-point patterns in circuit_lm source.

Scanned directories: circuit_lm/, scripts/
Excluded:           tests/  (this file would otherwise self-report)

Patterns detected
-----------------
  float_literal   ``\\b\\d+\\.\\d+\\b``   e.g.  1.0  3.14  0.001
  float_call      ``\\bfloat\\s*\\(``     e.g.  float(x)
  math_log        ``\\bmath\\.log\\b``
  math_exp        ``\\bmath\\.exp\\b``
  math_sqrt       ``\\bmath\\.sqrt\\b``
  import_numpy    ``\\bimport\\s+numpy\\b``
  from_numpy      ``\\bfrom\\s+numpy\\b``
  import_torch    ``\\bimport\\s+torch\\b``
  from_torch      ``\\bfrom\\s+torch\\b``
  import_jax      ``\\bimport\\s+jax\\b``
  from_jax        ``\\bfrom\\s+jax\\b``

Pure comment lines (stripped line starts with '#') are skipped so that
developer notes like "# no numpy here" do not trigger false positives.
"""

from __future__ import annotations

import pathlib
import re

# ---------------------------------------------------------------------------
# Pattern registry
# ---------------------------------------------------------------------------

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("float_literal", re.compile(r"\b\d+\.\d+\b")),
    ("float_call",    re.compile(r"\bfloat\s*\(")),
    ("math_log",      re.compile(r"\bmath\.log\b")),
    ("math_exp",      re.compile(r"\bmath\.exp\b")),
    ("math_sqrt",     re.compile(r"\bmath\.sqrt\b")),
    ("import_numpy",  re.compile(r"\bimport\s+numpy\b")),
    ("from_numpy",    re.compile(r"\bfrom\s+numpy\b")),
    ("import_torch",  re.compile(r"\bimport\s+torch\b")),
    ("from_torch",    re.compile(r"\bfrom\s+torch\b")),
    ("import_jax",    re.compile(r"\bimport\s+jax\b")),
    ("from_jax",      re.compile(r"\bfrom\s+jax\b")),
]

# Directories relative to the repository root that will be scanned
SCAN_DIRS: list[str] = ["circuit_lm", "scripts"]


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_no_float_patterns() -> None:
    """Fail if any forbidden float / import pattern appears in source files."""
    repo_root = pathlib.Path(__file__).parent.parent
    violations: list[str] = []

    for scan_dir in SCAN_DIRS:
        target = repo_root / scan_dir
        if not target.exists():
            continue

        for py_file in sorted(target.rglob("*.py")):
            text = py_file.read_text(encoding="utf-8")
            for lineno, line in enumerate(text.splitlines(), start=1):
                stripped = line.strip()
                # Skip pure comment lines
                if stripped.startswith("#"):
                    continue
                for label, pat in PATTERNS:
                    if pat.search(line):
                        rel = py_file.relative_to(repo_root)
                        violations.append(
                            f"  {rel}:{lineno} [{label}]: {line.rstrip()}"
                        )

    if violations:
        msg = (
            f"Floating-point or forbidden-import patterns detected "
            f"({len(violations)} violation(s)):\n" + "\n".join(violations)
        )
        raise AssertionError(msg)
