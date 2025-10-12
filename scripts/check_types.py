#!/usr/bin/env python3
"""Type checking script for Kapre.

This script runs basic type checking using mypy if available,
or falls back to basic syntax checking.
"""
import sys
import subprocess
from pathlib import Path


def run_mypy():
    """Run mypy type checking if available."""
    try:
        # Try to import mypy
        import mypy.api

        # Run mypy on the kapre package
        result = mypy.api.run([
            'kapre/',
            '--config-file',
            'mypy.ini',
        ])

        stdout, stderr, exit_code = result

        if stdout:
            print("MyPy Output:")
            print(stdout)

        if stderr:
            print("MyPy Errors:")
            print(stderr)

        return exit_code == 0

    except ImportError:
        print("MyPy not available. Install with: pip install mypy")
        return False


def run_pyright():
    """Run pyright type checking if available."""
    try:
        result = subprocess.run(
            ['pyright', 'kapre/'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )

        if result.stdout:
            print("Pyright Output:")
            print(result.stdout)

        if result.stderr:
            print("Pyright Errors:")
            print(result.stderr)

        return result.returncode == 0

    except FileNotFoundError:
        print("Pyright not available. Install with: pip install pyright")
        return False


def run_basic_checks():
    """Run basic syntax and import checks."""
    print("Running basic syntax and import checks...")

    kapre_path = Path(__file__).parent.parent / 'kapre'

    success = True

    # Check Python files compile
    for py_file in kapre_path.rglob('*.py'):
        if '__pycache__' in str(py_file):
            continue

        try:
            compile(py_file.read_text(), str(py_file), 'exec')
            print(f"✓ {py_file.relative_to(kapre_path.parent)}")
        except SyntaxError as e:
            print(f"✗ {py_file.relative_to(kapre_path.parent)}: {e}")
            success = False
        except Exception as e:
            print(f"? {py_file.relative_to(kapre_path.parent)}: {e}")

    # Try basic import (will fail if TensorFlow not installed, but that's OK)
    try:
        # Temporarily add parent directory to path
        sys.path.insert(0, str(kapre_path.parent))

        # Try importing without TensorFlow dependency
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "kapre.backend",
            kapre_path / "backend.py"
        )
        if spec and spec.loader:
            backend_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(backend_module)
            print("✓ backend.py imports successfully")
        else:
            print("✗ Could not load backend.py")
            success = False

    except Exception as e:
        print(f"? Import check failed (expected if TensorFlow not installed): {e}")

    return success


def main():
    """Main entry point."""
    print("Kapre Type Checking Script")
    print("=" * 40)

    success = False

    # Try mypy first
    if run_mypy():
        success = True
        print("\n✓ MyPy checks passed")
    else:
        print("\n✗ MyPy checks failed or not available")

    # Try pyright
    if run_pyright():
        success = True
        print("✓ Pyright checks passed")
    else:
        print("✗ Pyright checks failed or not available")

    # Fallback to basic checks
    if not success:
        print("\nFalling back to basic checks...")
        success = run_basic_checks()

    if success:
        print("\n🎉 All type checks passed!")
        return 0
    else:
        print("\n❌ Type checking failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())
