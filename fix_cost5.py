import subprocess

try:
    out = subprocess.check_output(["git", "log", "-p", "-S", "with_fma", "pixelflow-compiler/src/"]).decode("utf-8")
    print(out)
except Exception as e:
    print(e)
