"""CI gate: ensure circuit_lm's own source never imports forbidden libraries.

Design note – why AST and not sys.modules
------------------------------------------
OR-Tools CP-SAT (our only allowed solver dependency) itself pulls numpy,
pandas, and pyarrow into sys.modules as internal implementation details.
We cannot control that, so a naive sys.modules scan would always fail for
numpy.  Instead this test uses Python's ``ast`` module to parse every
circuit_lm source file and check for import statements referencing forbidden
modules.  This directly tests the constraint "circuit_lm's code does not
import X", which is the meaningful CI signal.

A separate check verifies that torch and jax – which are NOT OR-Tools
transitive dependencies – are genuinely absent from sys.modules after a full
package import.
"""

from __future__ import annotations

import ast
import importlib
import pathlib
import pkgutil
import sys

import circuit_lm

# Modules that must never appear in any circuit_lm import statement
FORBIDDEN_IMPORTS: frozenset[str] = frozenset(
    {"numpy", "torch", "jax", "tensorflow", "scipy"}
)

# Subset that should also be absent from sys.modules after package import.
# (OR-Tools does NOT depend on these, so their presence would be our fault.)
FORBIDDEN_IN_SYS_MODULES: frozenset[str] = frozenset({"torch", "jax", "tensorflow"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _circuit_lm_source_files() -> list[pathlib.Path]:
    """Return all .py source files inside the circuit_lm package directory."""
    pkg_root = pathlib.Path(circuit_lm.__file__).parent
    return sorted(pkg_root.rglob("*.py"))


def _submodule_names() -> list[str]:
    return [
        info.name
        for info in pkgutil.walk_packages(
            circuit_lm.__path__, prefix=circuit_lm.__name__ + "."
        )
    ]


def _forbidden_imports_in_file(
    src_path: pathlib.Path,
) -> list[tuple[int, str]]:
    """Parse *src_path* with the AST and return (lineno, module) for any
    forbidden import statements found."""
    source = src_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(src_path))
    except SyntaxError:
        return []

    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in FORBIDDEN_IMPORTS:
                    hits.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root in FORBIDDEN_IMPORTS:
                    hits.append((node.lineno, node.module))
    return hits


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_forbidden_import_statements_in_source() -> None:
    """Every .py file in circuit_lm/ must be free of forbidden import nodes.

    Uses AST parsing so that string literals that happen to mention a library
    name (e.g. in comments or docstrings embedded in source) are not flagged.
    """
    violations: list[str] = []
    for src in _circuit_lm_source_files():
        for lineno, mod in _forbidden_imports_in_file(src):
            violations.append(f"  {src.name}:{lineno} imports {mod!r}")

    assert not violations, (
        f"circuit_lm source files contain forbidden import statements "
        f"({len(violations)} violation(s)):\n" + "\n".join(violations)
    )


def test_torch_and_jax_not_in_sys_modules() -> None:
    """After importing all circuit_lm submodules, torch / jax must be absent.

    Note: numpy *will* appear in sys.modules because OR-Tools depends on it
    internally.  That is expected and does not violate the constraint that
    circuit_lm's *own* code is float-free and dependency-free.
    """
    for name in _submodule_names():
        try:
            importlib.import_module(name)
        except ImportError:
            pass

    loaded_roots = {m.split(".")[0] for m in sys.modules}
    violations = FORBIDDEN_IN_SYS_MODULES & loaded_roots
    assert not violations, (
        f"Forbidden modules found in sys.modules after importing circuit_lm: "
        f"{sorted(violations)}\n"
        "(numpy may legitimately appear as an OR-Tools internal dependency)"
    )


def test_forbidden_names_not_on_package_namespace() -> None:
    """circuit_lm's top-level namespace must not expose forbidden names."""
    for name in FORBIDDEN_IMPORTS:
        assert not hasattr(circuit_lm, name), (
            f"circuit_lm package exposes forbidden attribute: {name!r}"
        )
