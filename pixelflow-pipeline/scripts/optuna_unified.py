#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["optuna"]
# ///
"""
Optuna hyperparameter tuning via unix socket server.

The Rust training server loads the corpus ONCE, then accepts trial configs
as JSON over a unix socket. Each trial completes in ~1-2s instead of ~1.5h.

Usage:
    # Start the server (in another terminal, or let this script do it):
    cargo run --release -p pixelflow-pipeline --features training \
        --bin train_unified -- --server /tmp/train_unified.sock

    # Run the sweep:
    uv run pixelflow-pipeline/scripts/optuna_unified.py \
        --n-trials 100 --max-rounds 15

    # Reuse an already-running server and keep logs visible:
    uv run pixelflow-pipeline/scripts/optuna_unified.py \
        --socket-path /tmp/train_unified.sock --no-server

    # With final long training run:
    uv run pixelflow-pipeline/scripts/optuna_unified.py \
        --n-trials 100 --max-rounds 15 --final-rounds 1000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import optuna


SOCKET_PATH = "/tmp/train_unified.sock"


def find_workspace_root() -> Path:
    """Find the workspace root by looking for Cargo.toml with [workspace]."""
    current = Path.cwd()
    while current != current.parent:
        cargo_toml = current / "Cargo.toml"
        if cargo_toml.exists():
            content = cargo_toml.read_text()
            if "[workspace]" in content:
                return current
        current = current.parent
    return Path.cwd()


def send_trial(config: dict, socket_path: str = SOCKET_PATH, timeout: float = 300) -> dict:
    """Send a trial config to the Rust server, return response dict."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect(socket_path)
    try:
        payload = json.dumps(config) + "\n"
        sock.sendall(payload.encode())
        # Read response until connection closes
        data = b""
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            data += chunk
        return json.loads(data)
    finally:
        sock.close()


def _score_metrics(metrics: list[dict], mae_weight: float) -> float:
    """Composite score from trial metrics. Lower is better.

    Primary signal: negative median speedup (we want to MAXIMIZE speedup).
    Secondary: extraction MAE penalty if available (most runs report 0.0).
    Skipped rounds (no trajectories) are ignored.
    """
    best = float("inf")
    for m in metrics:
        if m.get("skipped"):
            continue
        speedup = m.get("speedup_median", 0)
        mae = m.get("extraction_mae", m.get("judge_mae", 0))
        extraction_loss = m.get("avg_extraction_loss", m.get("avg_value_loss"))
        try:
            speedup, mae = float(speedup), float(mae)
        except (TypeError, ValueError):
            speedup = float("nan")
            try:
                extraction_loss = float(extraction_loss)
            except (TypeError, ValueError):
                continue
            if math.isfinite(extraction_loss):
                best = min(best, extraction_loss)
            continue
        if not math.isfinite(speedup) or speedup <= 0:
            try:
                extraction_loss = float(extraction_loss)
            except (TypeError, ValueError):
                continue
            if math.isfinite(extraction_loss):
                best = min(best, extraction_loss)
            continue
        if not math.isfinite(mae):
            mae = 0.0
        score = -speedup + mae_weight * mae
        if score < best:
            best = score
    return best


def _score_trial_response(resp: dict, mae_weight: float) -> float:
    """Score a full trial response, preferring val_speedup when available.

    val_speedup is the median speedup on held-out real shader expressions
    (psychedelic red, channel, normalize, etc.). It can't be gamed by picking
    easy synthetic expressions during training.

    Falls back to in-sample speedup via _score_metrics if val_speedup is absent
    (e.g., from an old server version).
    """
    val_speedup = resp.get("val_speedup")
    if val_speedup is not None:
        try:
            v = float(val_speedup)
            if math.isfinite(v) and v > 0:
                # Negate: Optuna minimizes, we want to maximize speedup.
                return -v
        except (TypeError, ValueError):
            pass
    # Fall back to in-sample score
    return _score_metrics(resp.get("metrics", []), mae_weight)


def _safe_float(value: object, default: float = 0.0) -> float:
    """Best-effort float conversion for log output."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def ensure_server(workspace_root: Path, socket_path: str) -> subprocess.Popen | None:
    """Start the Rust training server if not already running."""
    # Check if socket exists and server responds
    if Path(socket_path).exists():
        try:
            test = send_trial({"rounds": 0, "trajectories_per_round": 0, "seed": 0}, timeout=5)
            print(f"Server already running at {socket_path}", flush=True)
            return None
        except (ConnectionRefusedError, FileNotFoundError, OSError):
            # Stale socket
            Path(socket_path).unlink(missing_ok=True)

    print("Building + starting training server...", flush=True)
    # Build first
    result = subprocess.run(
        ["cargo", "build", "-p", "pixelflow-pipeline", "--bin", "train_unified",
         "--release", "--features", "training"],
        cwd=workspace_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        print(f"ERROR: Build failed:\n{result.stderr[-500:]}", file=sys.stderr)
        sys.exit(1)

    # Start server
    proc = subprocess.Popen(
        ["cargo", "run", "-p", "pixelflow-pipeline", "--bin", "train_unified",
         "--release", "--features", "training", "--", "--server", socket_path],
        cwd=workspace_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    # Wait for socket to appear
    for _ in range(60):
        if Path(socket_path).exists():
            time.sleep(0.5)  # give it a moment after socket creation
            print(f"Server started (PID {proc.pid})", flush=True)
            return proc
        time.sleep(1)
        if proc.poll() is not None:
            stderr = proc.stderr.read().decode() if proc.stderr else ""
            print(f"ERROR: Server exited early:\n{stderr[-500:]}", file=sys.stderr)
            sys.exit(1)

    print("ERROR: Server did not create socket within 60s", file=sys.stderr)
    proc.kill()
    sys.exit(1)


def objective(
    trial: optuna.Trial,
    args: argparse.Namespace,
) -> float:
    """Send trial config to server, score the response."""
    # ── Policy optimizer ──
    lr_min, lr_max = (1e-4, 2e-3) if args.offline else (6e-4, 2.5e-3)
    momentum_min, momentum_max = (0.1, 0.8) if args.offline else (0.35, 0.65)
    wd_min, wd_max = (1e-7, 5e-4) if args.offline else (5e-5, 5e-4)
    gc_min, gc_max = (
        max(args.grad_clip_min, 0.05),
        min(args.grad_clip_max, 1.0),
    ) if args.offline else (
        max(args.grad_clip_min, 0.08),
        min(args.grad_clip_max, 0.3),
    )
    ent_min, ent_max = (0.001, 0.03) if args.offline else (7e-4, 3e-3)
    value_min, value_max = (0.1, 2.0) if args.offline else (0.12, 0.8)
    miss_min, miss_max = (0.0, 0.9) if args.offline else (0.5, 0.85)

    lr = trial.suggest_float("lr", lr_min, lr_max, log=True)
    momentum = trial.suggest_float("momentum", momentum_min, momentum_max)
    weight_decay = trial.suggest_float("weight_decay", wd_min, wd_max, log=True)
    grad_clip = trial.suggest_float("grad_clip", gc_min, gc_max, log=True)
    entropy_coeff = trial.suggest_float("entropy_coeff", ent_min, ent_max, log=True)
    entropy_floor_ratio = trial.suggest_float(
        "entropy_floor_ratio",
        0.05 if args.offline else 0.15,
        1.0 if args.offline else 0.5,
    )
    value_coeff = trial.suggest_float("value_coeff", value_min, value_max, log=True)
    miss_penalty = trial.suggest_float("miss_penalty", miss_min, miss_max)
    entropy_floor = max(1e-4, entropy_coeff * entropy_floor_ratio)

    # ── Replay buffer ──
    mini_batch_choices = [512, 1024, 2048] if args.offline else [1024, 2048]
    update_max = 12 if args.offline else 10
    replay_choices = [50_000, 100_000, 200_000] if args.offline else [50_000, 100_000]
    mini_batch_size = trial.suggest_categorical("mini_batch_size", mini_batch_choices)
    updates_per_round = trial.suggest_int("updates_per_round", 1, update_max)
    replay_capacity = trial.suggest_categorical("replay_capacity", replay_choices)

    # ── Critic / relabeling ──
    critic_epochs = args.critic_epochs
    critic_lr = args.critic_lr
    critic_dropout = args.critic_dropout
    critic_mini_batch_size = args.critic_mini_batch_size
    if args.offline:
        critic_epochs = trial.suggest_categorical("critic_epochs", [5, 10, 20])
        critic_lr = trial.suggest_float("critic_lr", 5e-5, 3e-4, log=True)
        critic_dropout = trial.suggest_float("critic_dropout", 0.0, 0.15)
        critic_mini_batch_size = trial.suggest_categorical(
            "critic_mini_batch_size", [32, 64, 128]
        )

    # ── Corpus mix ──
    corpus_fraction = 0.3

    # ── Mask ──
    threshold = 0.3
    max_steps = 50
    trajectories_per_round = args.trajectories_per_round

    seed = args.seed + trial.number * 1000

    config = {
        "rounds": args.max_rounds,
        "trajectories_per_round": trajectories_per_round,
        "max_steps": max_steps,
        "lr": lr,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        "entropy_coeff": entropy_coeff,
        "entropy_floor": entropy_floor,
        "value_coeff": value_coeff,
        "miss_penalty": miss_penalty,
        "threshold": threshold,
        "mini_batch_size": mini_batch_size,
        "updates_per_round": updates_per_round,
        "corpus_fraction": corpus_fraction,
        "seed": seed,
        "replay_capacity": replay_capacity,
        "offline": args.offline,
        "trajectory_dir": str(Path(args.trajectory_dir).resolve()) if args.trajectory_dir else None,
        "max_trajectory_files": args.max_trajectory_files,
        "critic_epochs": critic_epochs,
        "critic_lr": critic_lr,
        "critic_dropout": critic_dropout,
        "critic_mini_batch_size": critic_mini_batch_size,
    }

    print(
        f"\n[Trial {trial.number}] mode={'offline' if args.offline else 'online'} "
        f"lr={lr:.6f} mom={momentum:.2f} wd={weight_decay:.2e} "
        f"gc={grad_clip:.2f} ent={entropy_coeff:.4f}/{entropy_floor:.4f} val={value_coeff:.2f} "
        f"bs={mini_batch_size} upd={updates_per_round} replay={replay_capacity} "
        f"miss={miss_penalty:.2f} thresh={threshold:.2f} corpus={corpus_fraction:.2f} "
        f"critic=({critic_epochs},{critic_lr:.2e},{critic_dropout:.2f},{critic_mini_batch_size}) "
        f"traj={trajectories_per_round} steps={max_steps}",
        flush=True,
    )

    t0 = time.monotonic()
    try:
        # ~120s per round (self-play + JIT + critic + SGD) + startup buffer
        per_round_budget = args.offline_round_time_budget if args.offline else 120
        timeout = max(300, args.max_rounds * per_round_budget + 300)
        resp = send_trial(config, args.socket_path, timeout=timeout)
    except Exception as e:
        print(f"    ERROR: {e}", flush=True)
        return float("inf")

    elapsed = time.monotonic() - t0

    if "error" in resp:
        print(f"    SERVER ERROR: {resp['error']}", flush=True)
        return float("inf")

    metrics = resp.get("metrics", [])
    if not metrics:
        print(f"    NO METRICS returned ({elapsed:.1f}s)", flush=True)
        return float("inf")

    # Offline trials are scored on held-out validation speedup only, so pruning
    # on per-round replay losses is the wrong objective. Keep pruning for online
    # self-play, where round metrics do reflect the same task.
    if not args.offline:
        running_best = float("inf")
        for m in metrics:
            r = m.get("round", 0)
            score = _score_metrics([m], args.mae_weight)
            if score < running_best:
                running_best = score
            trial.report(running_best, step=r)
            if trial.should_prune():
                # Can't actually stop the server mid-trial, but we can prune early
                # for future scheduling decisions
                print(f"    PRUNED at round {r} ({elapsed:.1f}s)", flush=True)
                raise optuna.TrialPruned()

    # Final score: prefer val_speedup (held-out real shaders) over in-sample.
    best_score = _score_trial_response(resp, args.mae_weight)
    val_speedup = resp.get("val_speedup")

    last = metrics[-1]
    s = last.get('speedup_median', 0)
    mae = last.get('extraction_mae', last.get('judge_mae', 0))
    g = last.get('grad_norm_raw', last.get('grad_norm', 0))
    gc = last.get('grad_norm_clipped', 0)
    s = _safe_float(s)
    mae = _safe_float(mae)
    g = _safe_float(g)
    gc = _safe_float(gc)
    val_f = _safe_float(val_speedup, default=float("nan"))
    if args.offline:
        val_str = f"{val_f:.3f}x" if math.isfinite(val_f) else "n/a"
        print(
            f"    => val={val_str} grad_raw={g:.2f} grad_clip={gc:.2f} "
            f"score={best_score:.3f} ({elapsed:.1f}s)",
            flush=True,
        )
    else:
        val_str = f" val={val_f:.3f}x" if math.isfinite(val_f) else ""
        print(
            f"    => speedup={s:.3f}x{val_str} mae={mae:.3f} grad_raw={g:.2f} grad_clip={gc:.2f} "
            f"score={best_score:.3f} ({elapsed:.1f}s)",
            flush=True,
        )
    return best_score


def main():
    parser = argparse.ArgumentParser(
        description="Optuna tuning via unix socket training server"
    )
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--max-rounds", type=int, default=15,
                        help="Training rounds per trial")
    parser.add_argument("--final-rounds", type=int, default=0,
                        help="Rounds for final training with best params (0 to skip)")
    parser.add_argument("--mae-weight", type=float, default=0.5,
                        help="Weight on extraction_mae in composite score")
    parser.add_argument("--grad-clip-min", type=float, default=0.03,
                        help="Lower bound for log-uniform grad_clip sweep")
    parser.add_argument("--grad-clip-max", type=float, default=3.0,
                        help="Upper bound for log-uniform grad_clip sweep")
    parser.add_argument("--offline", action="store_true",
                        help="Use existing .pftraj files instead of fresh self-play")
    parser.add_argument("--trajectory-dir", type=str, default=None,
                        help="Directory containing trajectories_r*.pftraj for offline trials")
    parser.add_argument("--max-trajectory-files", type=int, default=0,
                        help="Limit offline trials to the most recent N trajectory files (0 = all)")
    parser.add_argument("--trajectories-per-round", type=int, default=200,
                        help="Online self-play trajectories per round")
    parser.add_argument("--offline-round-time-budget", type=int, default=45,
                        help="Estimated seconds per offline round when setting per-trial socket timeouts")
    parser.add_argument("--critic-epochs", type=int, default=20,
                        help="Default critic epochs when not sweeping offline critic training")
    parser.add_argument("--critic-lr", type=float, default=1.66e-4,
                        help="Default critic learning rate when not sweeping offline critic training")
    parser.add_argument("--critic-dropout", type=float, default=0.124,
                        help="Default critic dropout when not sweeping offline critic training")
    parser.add_argument("--critic-mini-batch-size", type=int, default=32,
                        help="Default critic mini-batch size when not sweeping offline critic training")
    parser.add_argument("--study-name", type=str, default="unified_v3")
    parser.add_argument("--output-dir", type=str, default="/tmp/optuna_unified")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=None,
                        help="Total optimization timeout in seconds")
    parser.add_argument("--socket-path", type=str, default=SOCKET_PATH,
                        help="Unix socket path for training server")
    parser.add_argument("--no-server", action="store_true",
                        help="Assume the Rust training server is already running at --socket-path")
    parser.add_argument("--resume", action="store_true",
                        help="Resume existing study")
    args = parser.parse_args()
    if args.grad_clip_min <= 0 or args.grad_clip_max <= args.grad_clip_min:
        parser.error("--grad-clip-min must be > 0 and --grad-clip-max must be greater")

    workspace_root = find_workspace_root()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / "study.db"

    print(f"Workspace: {workspace_root}", flush=True)
    print(f"Study DB: {db_path}", flush=True)
    print(f"Socket: {args.socket_path}", flush=True)
    print(f"Grad clip sweep: [{args.grad_clip_min}, {args.grad_clip_max}] log-uniform", flush=True)
    print(f"Mode: {'offline' if args.offline else 'online'}", flush=True)
    if args.offline:
        print(f"Trajectory dir: {args.trajectory_dir or '(server output dir)'}", flush=True)
        print(f"Max trajectory files: {args.max_trajectory_files}", flush=True)

    # Ensure server is running unless the caller is explicitly managing it.
    server_proc = None
    if not args.no_server:
        server_proc = ensure_server(workspace_root, args.socket_path)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    storage = f"sqlite:///{db_path}"

    # Delete stale study if needed
    if db_path.exists() and not args.resume:
        try:
            old = optuna.load_study(study_name=args.study_name, storage=storage)
            if old.trials and all(
                t.value is None or not math.isfinite(t.value)
                for t in old.trials if t.state == optuna.trial.TrialState.COMPLETE
            ):
                print("Deleting stale study with all-infinity results...", flush=True)
                optuna.delete_study(study_name=args.study_name, storage=storage)
        except KeyError:
            pass

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="minimize",
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=3,
            max_resource=args.max_rounds,
            reduction_factor=3,
        ),
        load_if_exists=True,
    )

    if len(study.trials) == 0:
        # Seed with known-good defaults
        if args.offline:
            seed_trial = {
                "lr": 6.65e-4,
                "momentum": 0.49,
                "weight_decay": 1.8e-4,
                "grad_clip": 0.08,
                "entropy_coeff": 0.0011,
                "entropy_floor_ratio": 0.2727272727,
                "value_coeff": 0.19,
                "mini_batch_size": 2048,
                "updates_per_round": 7,
                "replay_capacity": 50_000,
                "miss_penalty": 0.74,
                "threshold": 0.3,
                "corpus_fraction": 0.3,
                "critic_epochs": 10,
                "critic_lr": 9.33e-5,
                "critic_dropout": 0.06,
                "critic_mini_batch_size": 64,
            }
        else:
            seed_trial = {
                "lr": 1.2e-3,
                "momentum": 0.49,
                "weight_decay": 1.8e-4,
                "grad_clip": 0.15,
                "entropy_coeff": 0.0011,
                "entropy_floor_ratio": 0.2727272727,
                "value_coeff": 0.19,
                "mini_batch_size": 2048,
                "updates_per_round": 7,
                "replay_capacity": 50000,
                "miss_penalty": 0.74,
                "threshold": 0.3,
                "corpus_fraction": 0.3,
                "max_steps": 50,
                "trajectories_per_round": args.trajectories_per_round,
            }
        study.enqueue_trial(seed_trial)

    print(
        f"\nOptuna '{args.study_name}' "
        f"({args.n_trials} trials, {args.max_rounds} rounds/trial)\n",
        flush=True,
    )

    try:
        study.optimize(
            lambda trial: objective(trial, args),
            n_trials=args.n_trials,
            timeout=args.timeout,
            show_progress_bar=True,
        )
    finally:
        # Kill server if we started it
        if server_proc is not None:
            server_proc.terminate()
            server_proc.wait(timeout=5)
            Path(args.socket_path).unlink(missing_ok=True)

    # ── Results ──
    print("\n" + "=" * 60, flush=True)
    print("OPTIMIZATION COMPLETE", flush=True)
    print("=" * 60, flush=True)

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best score: {study.best_trial.value:.6f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")

    best_params_path = workspace_root / "pixelflow-pipeline" / "data" / "best_unified_params.json"
    best_params_path.write_text(json.dumps(
        {"best_trial": study.best_trial.number,
         "best_value": study.best_value,
         "best_params": study.best_params},
        indent=2,
    ))
    print(f"\nSaved best params to: {best_params_path}")

    # ── Final training with best params ──
    if args.final_rounds > 0:
        print("\n" + "=" * 60, flush=True)
        print(f"TRAINING FINAL MODEL ({args.final_rounds} rounds)", flush=True)
        print("=" * 60, flush=True)

        best = study.best_trial.params
        final_config = {
            "rounds": args.final_rounds,
            "trajectories_per_round": args.trajectories_per_round,
            "max_steps": best.get("max_steps", 50),
            "lr": best["lr"],
            "momentum": best["momentum"],
            "weight_decay": best["weight_decay"],
            "grad_clip": best["grad_clip"],
            "entropy_coeff": best["entropy_coeff"],
            "entropy_floor": max(1e-4, best["entropy_coeff"] * best.get("entropy_floor_ratio", 0.2040816327)),
            "value_coeff": best["value_coeff"],
            "miss_penalty": best["miss_penalty"],
            "threshold": best.get("threshold", 0.3),
            "mini_batch_size": best["mini_batch_size"],
            "updates_per_round": best["updates_per_round"],
            "corpus_fraction": best.get("corpus_fraction", 0.3),
            "seed": args.seed,
            "replay_capacity": best.get("replay_capacity", 200000),
            "offline": args.offline,
            "trajectory_dir": str(Path(args.trajectory_dir).resolve()) if args.trajectory_dir else None,
            "max_trajectory_files": args.max_trajectory_files,
            "critic_epochs": best.get("critic_epochs", args.critic_epochs),
            "critic_lr": best.get("critic_lr", args.critic_lr),
            "critic_dropout": best.get("critic_dropout", args.critic_dropout),
            "critic_mini_batch_size": best.get(
                "critic_mini_batch_size", args.critic_mini_batch_size
            ),
        }
        if not args.offline:
            final_config["trajectories_per_round"] = best.get(
                "trajectories_per_round", args.trajectories_per_round
            )

        print(f"Config: lr={best['lr']:.6f} mom={best['momentum']:.2f} "
              f"gc={best['grad_clip']:.2f} ent={best['entropy_coeff']:.4f}", flush=True)

        # Re-ensure server (it might have been killed in the finally block)
        server_proc2 = ensure_server(workspace_root, args.socket_path)

        try:
            # Long timeout for 1000 rounds
            timeout = max(300, args.final_rounds * 5 + 120)
            resp = send_trial(final_config, args.socket_path, timeout=timeout)

            if "error" in resp:
                print(f"ERROR: {resp['error']}", file=sys.stderr)
            else:
                metrics = resp.get("metrics", [])
                if metrics:
                    last = metrics[-1]
                    print(f"\nFinal speedup: {last.get('speedup_median', '?')}")
                    print(
                        f"Final extraction_mae: "
                        f"{last.get('extraction_mae', last.get('judge_mae', '?'))}"
                    )
                    print(f"Final grad_norm: {last.get('grad_norm', '?')}")
                    print(f"Rounds completed: {len(metrics)}")
        finally:
            if server_proc2 is not None:
                server_proc2.terminate()
                server_proc2.wait(timeout=5)
            Path(args.socket_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
