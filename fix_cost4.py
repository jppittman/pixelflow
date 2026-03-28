# Let's check `git log -S with_fma pixelflow-search/src/egraph/cost.rs`
import subprocess

try:
    out = subprocess.check_output(["git", "log", "-p", "-S", "with_fma", "pixelflow-search/src/egraph/cost.rs"]).decode("utf-8")
    print(out)
except Exception as e:
    print(e)
