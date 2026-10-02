"""Check reporting without a Rust toolchain or a working WASM backend."""

import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("check-wasm-status.sh")
BASH = shutil.which("bash")
CRATES = ("pixelflow-search", "pixelflow-core", "pixelflow-runtime")


class WasmStatusTests(unittest.TestCase):
    def test_report_every_crate_and_fail_if_any_build_fails(self):
        for outcomes in itertools.product((0, 101), repeat=len(CRATES)):
            with self.subTest(outcomes=outcomes), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                cargo = root / "cargo"
                cargo.write_text(
                    f"#!{sys.executable}\n"
                    "import json, os, sys\n"
                    "crate = sys.argv[sys.argv.index('-p') + 1]\n"
                    "with open(os.environ['CALL_LOG'], 'a') as log:\n"
                    "    log.write(json.dumps([sys.argv[1:], os.getenv('DISPLAY_DRIVER')]) + '\\n')\n"
                    "sys.exit(json.loads(os.environ['OUTCOMES'])[crate])\n"
                )
                cargo.chmod(0o755)
                log = root / "calls.jsonl"
                env = dict(os.environ, PATH=f"{root}{os.pathsep}{os.environ['PATH']}",
                           CALL_LOG=str(log), OUTCOMES=json.dumps(dict(zip(CRATES, outcomes))),
                           DISPLAY_DRIVER="x11")
                result = subprocess.run([BASH, str(SCRIPT)], cwd=root, env=env,
                                        text=True, capture_output=True, check=False)
                self.assertEqual(result.returncode, int(any(outcomes)), result.stderr)
                calls = [json.loads(line) for line in log.read_text().splitlines()]
                expected = [
                    [["check", "--target", "wasm32-unknown-unknown", "-p", crate], "x11"]
                    for crate in CRATES[:2]
                ]
                expected.append([
                    ["check", "--target", "wasm32-unknown-unknown", "-p", CRATES[2],
                     "--no-default-features", "--features", "display_web"], "web"
                ])
                self.assertEqual(calls, expected)
                self.assertEqual(result.stdout.count("::endgroup::"), len(CRATES))
                for crate, status in zip(CRATES, outcomes):
                    label = "FAIL" if status else "PASS"
                    self.assertIn(f"{label}: {crate}", result.stdout)


if __name__ == "__main__":
    unittest.main()
