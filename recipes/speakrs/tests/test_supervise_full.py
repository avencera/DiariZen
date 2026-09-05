#!/usr/bin/env python3

"""Regression tests for the Speakrs supervisor state machine."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


SUPERVISOR_PATH = Path(__file__).resolve().parents[1] / "supervise_full.sh"


class SupervisorStateMachineTest(unittest.TestCase):
    """Check completion detection after the restart budget is exhausted."""

    def run_supervisor(self, success_on, stale_completion=False):
        with tempfile.TemporaryDirectory() as temporary_directory:
            recipe_dir = Path(temporary_directory)
            (recipe_dir / "conf").mkdir()
            (recipe_dir / "conf" / "edge.toml").write_text("[training]\n")
            (recipe_dir / "success_on").write_text(f"{success_on}\n")
            if stale_completion:
                (recipe_dir / "edge.pipeline.log").write_text("previous run: Training loop finished at epoch 1\n")
            (recipe_dir / "run_full_pipeline.sh").write_text(
                """#!/usr/bin/env bash
set -euo pipefail
recipe_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
attempts_file="$recipe_dir/attempts"
success_on_file="$recipe_dir/success_on"
attempts=0
if [[ -f "$attempts_file" ]]; then
    attempts="$(cat "$attempts_file")"
fi
attempts=$((attempts + 1))
printf '%s\\n' "$attempts" > "$attempts_file"
success_on="$(cat "$success_on_file")"
if (( attempts == success_on )); then
    printf '%s\\n' 'Training loop finished at epoch 1'
fi
"""
            )
            (recipe_dir / "evaluate_full.sh").write_text(
                """#!/usr/bin/env bash
set -euo pipefail
recipe_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
printf '%s\\n' evaluated >> "$recipe_dir/evaluations"
"""
            )
            (recipe_dir / "supervise_full.sh").write_text(SUPERVISOR_PATH.read_text())
            for path in ("run_full_pipeline.sh", "evaluate_full.sh", "supervise_full.sh"):
                (recipe_dir / path).chmod(0o755)

            environment = os.environ.copy()
            environment.update(
                {
                    "DIARIZEN_TRAINING_CONFIG": str(recipe_dir / "conf" / "edge.toml"),
                    "DIARIZEN_EXPERIMENT_ID": "edge",
                    "DIARIZEN_SUPERVISOR_POLL_SECONDS": "0",
                }
            )
            completed = subprocess.run(
                [str(recipe_dir / "supervise_full.sh")],
                cwd=recipe_dir,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
            )

            return {
                "completed": completed,
                "attempts": int((recipe_dir / "attempts").read_text()),
                "evaluations": (
                    (recipe_dir / "evaluations").read_text().splitlines()
                    if (recipe_dir / "evaluations").exists()
                    else []
                ),
                "status": (recipe_dir / "edge.status").read_text().strip(),
                "supervisor_log": (recipe_dir / "edge.supervisor.log").read_text(),
            }

    def test_success_is_evaluated_after_each_attempt(self):
        for success_on in range(1, 6):
            with self.subTest(success_on=success_on):
                run = self.run_supervisor(success_on)

                self.assertEqual(run["completed"].returncode, 0, run["completed"].stderr)
                self.assertEqual(run["attempts"], success_on)
                self.assertEqual(run["evaluations"], ["evaluated"])
                self.assertEqual(run["status"], "ready_to_stop")
                self.assertIn("evaluation complete", run["supervisor_log"])

    def test_all_five_attempts_fail_without_evaluation(self):
        run = self.run_supervisor(success_on=0)

        self.assertEqual(run["completed"].returncode, 1, run["completed"].stderr)
        self.assertEqual(run["attempts"], 5)
        self.assertEqual(run["evaluations"], [])
        self.assertEqual(run["status"], "failed")

    def test_stale_completion_marker_does_not_count_as_completion(self):
        run = self.run_supervisor(success_on=0, stale_completion=True)

        self.assertEqual(run["completed"].returncode, 1, run["completed"].stderr)
        self.assertEqual(run["attempts"], 5)
        self.assertEqual(run["evaluations"], [])
        self.assertEqual(run["status"], "failed")

    def test_stale_completion_marker_does_not_skip_last_restart(self):
        run = self.run_supervisor(success_on=5, stale_completion=True)

        self.assertEqual(run["completed"].returncode, 0, run["completed"].stderr)
        self.assertEqual(run["attempts"], 5)
        self.assertEqual(run["evaluations"], ["evaluated"])
        self.assertEqual(run["status"], "ready_to_stop")


if __name__ == "__main__":
    unittest.main()
