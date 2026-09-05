#!/usr/bin/env python3

"""Tests for dscore DER result verification."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parent.parent / "verify_der_result.py"


def load_verifier_module():
    """Load the result verifier without depending on the working directory."""

    spec = importlib.util.spec_from_file_location("verify_der_result", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the DER result verifier")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VERIFIER = load_verifier_module()


class DerResultTest(unittest.TestCase):
    """Check parsing and expectation boundaries."""

    def test_parses_overall_der(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = Path(temporary_directory) / "result.txt"
            result.write_text("File DER JER\none 14.00 20.00\n*** OVERALL *** 15.60 21.00\n")

            self.assertEqual(VERIFIER.parse_overall_der(result), 15.6)

    def test_requires_one_overall_row(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = Path(temporary_directory) / "result.txt"
            result.write_text("File DER JER\n")

            with self.assertRaisesRegex(ValueError, "expected one OVERALL row"):
                VERIFIER.parse_overall_der(result)

    def test_accepts_tolerance_boundary(self):
        expectation = VERIFIER.DerExpectation(expected=15.6, tolerance=1.0)

        self.assertTrue(expectation.accepts(14.6))
        self.assertTrue(expectation.accepts(16.6))
        self.assertFalse(expectation.accepts(16.61))


if __name__ == "__main__":
    unittest.main()
