"""Integer-only metrics.

No floating-point arithmetic is used.  Accuracy is represented as an
exact integer fraction or as integer basis-points (1/100 of a percent).

Basis-point encoding example
-----------------------------
  accuracy 25 % → 2500 bp
  accuracy  1 % →  100 bp
  accuracy  0 % →    0 bp
  accuracy 100% → 10000 bp

TODO: Perplexity approximation using integer log2 tables.
TODO: F1 / precision / recall for individual token classes.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _gcd(a: int, b: int) -> int:
    """Euclidean greatest common divisor (integers only)."""
    while b:
        a, b = b, a % b
    return a


# ---------------------------------------------------------------------------
# Accuracy representations
# ---------------------------------------------------------------------------


def accuracy_fraction(correct: int, total: int) -> tuple[int, int]:
    """Return accuracy as a reduced integer fraction ``(numerator, denominator)``.

    Returns ``(0, 1)`` when *total* is zero.

    >>> accuracy_fraction(1, 4)
    (1, 4)
    >>> accuracy_fraction(2, 4)
    (1, 2)
    """
    if total == 0:
        return (0, 1)
    g = _gcd(correct, total)
    return (correct // g, total // g)


def accuracy_pct_times100(correct: int, total: int) -> int:
    """Return accuracy in integer basis-points (hundredths of a percent).

    The result is ``(correct * 10000) // total``, avoiding all floats.

    Examples:
      - 1 correct out of 4 total  →  2500  (= 25 %)
      - 3 correct out of 3 total  → 10000  (= 100 %)
      - 0 total                   →      0

    >>> accuracy_pct_times100(1, 4)
    2500
    """
    if total == 0:
        return 0
    return (correct * 10000) // total


def format_accuracy(correct: int, total: int) -> str:
    """Format accuracy as ``'XX.YY%'`` using only integer arithmetic.

    1 correct out of 4 total  → "25.xx%" (25 percent, two decimal digits).
    All arithmetic uses only integer basis-points; no float values produced.
    """
    if total == 0:
        return "N/A (0 samples)"
    bps = accuracy_pct_times100(correct, total)
    whole = bps // 100
    frac = bps % 100
    # f-string: no float values – whole and frac are ints
    return f"{whole}.{frac:02d}%"
