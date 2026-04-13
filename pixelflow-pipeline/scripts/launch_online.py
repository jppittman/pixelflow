#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["optuna", "requests"]
# ///
"""
Read best hyperparams from the offline Optuna study and launch an online training run.

Usage:
    # After offline sweep completes:
    uv run pixelflow-pipeline/scripts/launch_online.py

    # Override rounds / trajectories:
    uv run pixelflow-pipeline/scripts/launch_online.py --rounds 100 --traj 30

    # Dry-run (print command, don't execute):
    uv run pixelflow-pipeline/scripts/launch_online.py --dry-run

    # Skip the persistent critic server (legacy subprocess mode):
    uv run pixelflow-pipeline/scripts/launch_online.py --no-server
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import optuna
import requests


CRITIC_SERVER_PORT = 8765
CRITIC_SERVER_URL = f"http://localhost:{CRITIC_SERVER_PORT}"
# Maximum seconds to wait for the critic server to become healthy at startup.
CRITIC_SERVER_STARTUP_TIMEOUT_S = 30


def find_workspace_root() -> Path:
    current = Path.cwd()
    while current != current.parent:
        if (current / "Cargo.toml").exists():
            if "[workspace]" in (current / "Cargo.toml").read_text():
                return current
        current = current.parent
    return Path.cwd()


def load_best_params(study_db: Path, study_name: str) -> dict:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = f"sqlite:///{study_db}"
    study = optuna.load_study(study_name=study_name, storage=storage)
    complete = [t for t in study.trials if t.state.name == "COMPLETE"]
    if not complete:
        raise RuntimeError(f"No completed trials in study '{study_name}'")
    print(f"Study '{study_name}': {len(complete)} complete, {len(study.trials)-len(complete)} other")
    print(f"Best trial #{study.best_trial.number}: score={study.best_value:.4f}")
    return dict(study.best_trial.params)


def _spawn_critic_server(
    workspace: Path,
    critic_pt: Path,
    params: dict,
    port: int,
) -> subprocess.Popen:
    """Spawn critic_server.py as a background process and return the handle.

    The server logs to stderr only so its output doesn't pollute stdout.
    """
    server_script = workspace / "pixelflow-pipeline" / "scripts" / "critic_server.py"
    if not server_script.exists():
        raise FileNotFoundError(
            f"critic_server.py not found at {server_script}. "
            "Cannot start persistent critic server."
        )

    cmd = [
        "uv", "run", str(server_script),
        "--port", str(port),
        "--checkpoint", str(critic_pt),
        "--lr",           str(params.get("critic_lr", 1e-4)),
        "--dropout",      str(params.get("critic_dropout", 0.1)),
    ]
    print(f"Spawning critic server: {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.Popen(
        cmd,
        cwd=str(workspace),
        stdout=subprocess.DEVNULL,  # server uses stderr for all logging
        stderr=None,                # inherit — lets server logs reach the terminal
    )
    return proc


def _wait_for_critic_server(url: str, timeout_s: int) -> bool:
    """Poll GET /health until OK or timeout. Returns True on success."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            resp = requests.get(f"{url}/health", timeout=2)
            if resp.status_code == 200 and resp.json().get("status") == "ok":
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(0.5)
    return False


def main():
    parser = argparse.ArgumentParser(description="Launch online training with best offline params")
    parser.add_argument("--study-db", default="/tmp/optuna_unified/study.db",
                        help="Path to Optuna SQLite study DB")
    parser.add_argument("--study-name", default="unified_v3_offline",
                        help="Optuna study name")
    parser.add_argument("--rounds", type=int, default=100,
                        help="Online training rounds")
    parser.add_argument("--traj", type=int, default=30,
                        help="Trajectories per round")
    parser.add_argument("--max-steps", type=int, default=50,
                        help="Max hill-climbing steps per trajectory")
    parser.add_argument("--output-dir", default="pixelflow-pipeline/data/online_run",
                        help="Output directory for checkpoints and trajectories")
    parser.add_argument("--init-model", default=None,
                        help="Optional initial model checkpoint to load at startup")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the command but don't execute")
    parser.add_argument("--params-json", default=None,
                        help="Use params from this JSON file instead of Optuna DB")
    parser.add_argument("--critic-epochs", type=int, default=None,
                        help="Override critic_epochs from params (default: use Optuna value)")
    parser.add_argument("--traj-per-round", type=int, default=None,
                        help="Override trajectories-per-round from params")
    parser.add_argument(
        "--no-server",
        action="store_true",
        help=(
            "Skip pre-spawning the persistent critic server here. "
            "The Rust trainer may still auto-start critic_server.py if "
            "--critic-url is unreachable. Use this for debugging launcher-side "
            "server startup only."
        ),
    )
    args = parser.parse_args()

    workspace = find_workspace_root()

    if args.params_json:
        params = json.loads(Path(args.params_json).read_text())
        if "best_params" in params:
            params = params["best_params"]
        print(f"Loaded params from {args.params_json}")
    else:
        db = Path(args.study_db)
        if not db.exists():
            print(f"ERROR: Study DB not found at {db}", file=sys.stderr)
            print("Run the offline Optuna sweep first, or pass --params-json", file=sys.stderr)
            sys.exit(1)
        params = load_best_params(db, args.study_name)

    print("\nBest params:")
    for k, v in sorted(params.items()):
        print(f"  {k}: {v}")

    output_dir = workspace / args.output_dir
    final_model = output_dir / "final_model.bin"
    critic_pt = output_dir / "critic.pt"

    # ------------------------------------------------------------------
    # Optionally start the persistent critic server
    # ------------------------------------------------------------------
    server_proc: subprocess.Popen | None = None
    critic_url_flag: list[str] = []

    if not args.no_server and not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            server_proc = _spawn_critic_server(
                workspace, critic_pt, params, CRITIC_SERVER_PORT
            )
            print(
                f"Waiting up to {CRITIC_SERVER_STARTUP_TIMEOUT_S}s for critic server...",
                file=sys.stderr,
            )
            if _wait_for_critic_server(CRITIC_SERVER_URL, CRITIC_SERVER_STARTUP_TIMEOUT_S):
                print(
                    f"Critic server ready at {CRITIC_SERVER_URL}",
                    file=sys.stderr,
                )
                critic_url_flag = ["--critic-url", CRITIC_SERVER_URL]
            else:
                print(
                    "WARNING: Critic server did not become healthy within "
                    f"{CRITIC_SERVER_STARTUP_TIMEOUT_S}s. "
                    "Falling back to subprocess mode.",
                    file=sys.stderr,
                )
                server_proc.terminate()
                server_proc = None
        except FileNotFoundError as exc:
            print(
                f"WARNING: Could not start critic server ({exc}). "
                "Falling back to subprocess mode.",
                file=sys.stderr,
            )
            server_proc = None

    cmd = [
        "cargo", "run", "--release",
        "-p", "pixelflow-pipeline",
        "--bin", "train_unified",
        "--features", "training",
        "--",
        "--rounds",                 str(args.rounds),
        "--trajectories-per-round", str(args.traj_per_round if args.traj_per_round is not None else args.traj),
        "--max-steps",              str(args.max_steps),
        "--final-model",            str(final_model),
        "--output-dir",             str(output_dir),
        "--critic-checkpoint",      str(critic_pt),
        "--corpus-fraction",        str(params.get("corpus_fraction", 0.3)),
        "--threshold",              str(params.get("threshold", 0.3)),
        "--lr",                     str(params["lr"]),
        "--momentum",               str(params["momentum"]),
        "--weight-decay",           str(params["weight_decay"]),
        "--grad-clip",              str(params["grad_clip"]),
        "--entropy-coeff",          str(params["entropy_coeff"]),
        "--value-coeff",            str(params["value_coeff"]),
        "--mini-batch-size",        str(params["mini_batch_size"]),
        "--updates-per-round",      str(params["updates_per_round"]),
        "--critic-epochs",          str(args.critic_epochs if args.critic_epochs is not None else params["critic_epochs"]),
        "--critic-lr",              str(params["critic_lr"]),
        "--critic-dropout",         str(params["critic_dropout"]),
        "--critic-mini-batch-size", str(params.get("critic_mini_batch_size", 32)),
        "--miss-penalty",           str(params["miss_penalty"]),
        "--replay-capacity",        "200000",
        "--relabel-interval",       "5",
        "--seed",                   "42",
        *critic_url_flag,
    ]

    if args.init_model is not None:
        init_model = workspace / args.init_model
        cmd.extend(["--model", str(init_model)])

    print(f"\nOutput dir: {output_dir}")
    print(f"Init model: {init_model}")
    print(f"Final model: {final_model}")
    print(f"\nCommand:\n  " + " \\\n  ".join(cmd))

    if args.dry_run:
        print("\n(dry-run: not executing)")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print("\nLaunching...", flush=True)
    try:
        result = subprocess.run(cmd, cwd=workspace)
    finally:
        # Always shut down the server when the training run finishes or crashes.
        if server_proc is not None:
            print("Shutting down critic server...", file=sys.stderr)
            server_proc.send_signal(signal.SIGTERM)
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
