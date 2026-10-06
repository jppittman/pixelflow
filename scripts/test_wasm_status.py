"""Check what check-wasm-status.sh reports, without a Rust toolchain.

The script's own verdict is the thing under test here, so `cargo` is faked on
PATH and every pass/fail combination is forced. That is the whole point: the
WASM gap means CI cannot exercise the failing half for real, and "we cannot
test it" is how the bug this replaces survived -- a step whose last command
was an `echo`, reporting success through three failed builds.

The script locates its baseline relative to its own path, so each case copies
it into a temporary tree with a baseline of that case's choosing. No seam in
the script exists for the test, and none is needed.
"""

import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


REAL_SCRIPT = Path(__file__).with_name("check-wasm-status.sh")
REAL_BASELINE = Path(__file__).with_name("wasm_status_baseline.txt")
BASH = shutil.which("bash")

# As the script labels them, which is also how the baseline names them.
SEARCH = "pixelflow-search"
CORE = "pixelflow-core"
WEB = "pixelflow-runtime web driver"
LABELS = (SEARCH, CORE, WEB)

FAKE_CARGO = """#!{python}
import json, os, sys
label = json.loads(os.environ["LABEL_FOR"])[sys.argv[sys.argv.index("-p") + 1]]
with open(os.environ["CALL_LOG"], "a") as log:
    log.write(json.dumps([sys.argv[1:], os.getenv("DISPLAY_DRIVER")]) + "\\n")
sys.exit(json.loads(os.environ["OUTCOMES"])[label])
"""

# `-p <crate>` is what the fake sees; the web driver's crate is pixelflow-runtime.
CRATE_TO_LABEL = {
    "pixelflow-search": SEARCH,
    "pixelflow-core": CORE,
    "pixelflow-runtime": WEB,
}

EXPECTED_CALLS = [
    [["check", "--target", "wasm32-unknown-unknown", "-p", "pixelflow-search"], "x11"],
    [["check", "--target", "wasm32-unknown-unknown", "-p", "pixelflow-core"], "x11"],
    [
        [
            "check", "--target", "wasm32-unknown-unknown", "-p", "pixelflow-runtime",
            "--no-default-features", "--features", "display_web",
        ],
        "web",
    ],
]


class WasmStatusTests(unittest.TestCase):
    def run_script(self, tmp, outcomes, baseline):
        """Run the script in a throwaway tree. `outcomes`: label -> exit code."""
        root = Path(tmp)
        scripts = root / "scripts"
        scripts.mkdir()
        shutil.copy(REAL_SCRIPT, scripts / REAL_SCRIPT.name)
        (scripts / REAL_BASELINE.name).write_text(
            "# a comment, and a blank line, both of which must be skipped\n\n"
            + "".join(f"{label}\n" for label in baseline)
        )

        cargo = root / "cargo"
        cargo.write_text(FAKE_CARGO.format(python=sys.executable))
        cargo.chmod(0o755)
        log = root / "calls.jsonl"

        env = dict(
            os.environ,
            PATH=f"{root}{os.pathsep}{os.environ['PATH']}",
            CALL_LOG=str(log),
            OUTCOMES=json.dumps(outcomes),
            LABEL_FOR=json.dumps(CRATE_TO_LABEL),
            # A pre-set driver the script must override for the web check only.
            DISPLAY_DRIVER="x11",
        )
        result = subprocess.run(
            [BASH, str(scripts / REAL_SCRIPT.name)],
            env=env, text=True, capture_output=True, check=False,
        )
        calls = [json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []
        return result, calls

    def test_it_should_exit_zero_exactly_when_failures_match_the_baseline(self):
        for failing in itertools.product((False, True), repeat=len(LABELS)):
            expected_failures = {l for l, f in zip(LABELS, failing) if f}
            for baseline in itertools.chain.from_iterable(
                itertools.combinations(LABELS, n) for n in range(len(LABELS) + 1)
            ):
                with self.subTest(failing=failing, baseline=baseline), \
                        tempfile.TemporaryDirectory() as tmp:
                    outcomes = {l: (101 if f else 0) for l, f in zip(LABELS, failing)}
                    result, _ = self.run_script(tmp, outcomes, baseline)
                    matches = expected_failures == set(baseline)
                    self.assertEqual(
                        result.returncode, 0 if matches else 1,
                        f"failing={failing} baseline={baseline}\n{result.stdout}\n{result.stderr}",
                    )

    def test_it_should_run_and_report_every_crate_even_after_one_fails(self):
        # The first crate failing must not stop the other two: the log is
        # supposed to say what the state of wasm32 support IS, not where it
        # first tripped.
        with tempfile.TemporaryDirectory() as tmp:
            outcomes = {SEARCH: 101, CORE: 101, WEB: 101}
            result, calls = self.run_script(tmp, outcomes, LABELS)
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertEqual(calls, EXPECTED_CALLS)
            self.assertEqual(result.stdout.count("::endgroup::"), len(LABELS))
            for label in LABELS:
                self.assertIn(f"FAIL: {label}", result.stdout)

    def test_it_should_report_each_crate_that_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            outcomes = {label: 0 for label in LABELS}
            result, calls = self.run_script(tmp, outcomes, ())
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertEqual(calls, EXPECTED_CALLS)
            for label in LABELS:
                self.assertIn(f"PASS: {label}", result.stdout)

    def test_it_should_name_a_new_failure_and_a_closed_gap_distinctly(self):
        # The two directions are different news and must not share a message:
        # one says wasm32 got worse, the other says to prune the baseline.
        with tempfile.TemporaryDirectory() as tmp:
            result, _ = self.run_script(tmp, {SEARCH: 101, CORE: 0, WEB: 0}, (CORE,))
            self.assertEqual(result.returncode, 1)
            self.assertIn(f"::error::{SEARCH} stopped building", result.stdout)
            self.assertIn(f"::error::{CORE} now builds", result.stdout)

    def test_the_shipped_baseline_should_name_only_crates_this_script_checks(self):
        entries = [
            line for line in REAL_BASELINE.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertTrue(entries, "an empty baseline should be deleted, not shipped")
        self.assertEqual(sorted(set(entries)), sorted(entries), "duplicate baseline entry")
        for entry in entries:
            self.assertIn(entry, LABELS, f"{entry!r} is not a label this script reports")


if __name__ == "__main__":
    unittest.main()
