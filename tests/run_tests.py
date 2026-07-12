#!/usr/bin/env python3
"""Zero-dependency test runner.

Runs every ``test_*`` function in every ``tests/test_*.py`` module using only the
standard library. If real ``pytest`` is installed you can equally run
``pytest tests/`` — the test files are pytest-compatible. This runner exists so
the suite is green in a locked-down environment with nothing installed.

It installs a tiny ``pytest`` shim (``approx``, ``raises``) into ``sys.modules``
only when the real package is unavailable, so ``import pytest`` in the test files
resolves either way.
"""
from __future__ import annotations

import glob
import importlib.util
import os
import sys
import traceback
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # for `import _bootstrap`
sys.path.insert(0, os.path.join(HERE, "..", "src"))


def _install_pytest_shim() -> None:
    try:
        import pytest  # noqa: F401
        return  # real pytest present; use it
    except ImportError:
        pass

    shim = types.ModuleType("pytest")

    class _Approx:
        def __init__(self, expected, rel=1e-6, abs=1e-9):
            self.expected = expected
            self.rel = rel
            self.abs = abs

        def __eq__(self, other):
            return abs(other - self.expected) <= max(self.abs, self.rel * abs(self.expected))

        def __repr__(self):
            return f"approx({self.expected})"

    def approx(expected, rel=1e-6, abs=1e-9):
        return _Approx(expected, rel, abs)

    class _Raises:
        def __init__(self, exc):
            self.exc = exc

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, tb):
            if exc_type is None:
                raise AssertionError(f"DID NOT RAISE {self.exc!r}")
            return issubclass(exc_type, self.exc)

    def raises(exc):
        return _Raises(exc)

    shim.approx = approx
    shim.raises = raises
    sys.modules["pytest"] = shim


def _load_module(path: str):
    name = "t_" + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    _install_pytest_shim()
    files = sorted(glob.glob(os.path.join(HERE, "test_*.py")))
    passed = failed = 0
    failures = []

    for path in files:
        mod = _load_module(path)
        for attr in sorted(dir(mod)):
            if not attr.startswith("test_"):
                continue
            fn = getattr(mod, attr)
            if not callable(fn):
                continue
            label = f"{os.path.basename(path)}::{attr}"
            try:
                fn()
                passed += 1
                print(f"  PASS {label}")
            except Exception:  # noqa: BLE001
                failed += 1
                failures.append((label, traceback.format_exc()))
                print(f"  FAIL {label}")

    print("\n" + "=" * 60)
    print(f"  {passed} passed, {failed} failed")
    print("=" * 60)
    for label, tb in failures:
        print(f"\n--- {label} ---\n{tb}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
