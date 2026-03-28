# Ah, maybe I should check the git history to see when `with_fma` was removed?
import subprocess

out = subprocess.check_output(["git", "log", "-S", "fully_optimized", "pixelflow-search/src/egraph/cost.rs"]).decode("utf-8")
print(out)
