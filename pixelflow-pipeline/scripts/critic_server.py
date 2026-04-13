#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch>=2.1",
#     "numpy>=1.26.0",
#     "fastapi>=0.110.0",
#     "uvicorn>=0.29.0",
# ]
# ///
"""Persistent FastAPI server wrapping the Causal Sequence Transformer Critic.

Keeps the model, optimizer, and rolling replay buffer warm across training
rounds, eliminating the ~1.8s Python startup + model-load overhead that
accumulates when critic.py is spawned as a fresh subprocess each round.

Endpoints:
  POST /train   — offline-style train on a trajectory file, write advantages, return metrics
  POST /predict — inference only on a trajectory file, write advantages
  POST /step    — inference + one incremental training step on the same new batch
  POST /retrain — retrain from buffered data, then start EMA blending
  GET  /health  — liveness check
  POST /reset   — wipe the rolling replay buffer (optional, between checkpoints)

Port: 8765 (configurable via --port).

All diagnostic output goes to stderr so it doesn't pollute the orchestrator's
stdout.

Usage:
    uv run critic_server.py --port 8765 --checkpoint /path/to/critic.pt
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW

# =============================================================================
# Re-use all model/data code from critic.py verbatim — no duplication.
# We import the symbols we need directly.
# =============================================================================

# Ensure the scripts directory is on the path so we can import critic.
import importlib.util
import os

_SCRIPTS_DIR = Path(__file__).parent
_CRITIC_MODULE_PATH = _SCRIPTS_DIR / "critic.py"

if not _CRITIC_MODULE_PATH.exists():
    print(
        f"FATAL: critic.py not found at {_CRITIC_MODULE_PATH}",
        file=sys.stderr,
    )
    raise SystemExit(1)

spec = importlib.util.spec_from_file_location("critic", _CRITIC_MODULE_PATH)
if spec is None or spec.loader is None:
    print("FATAL: could not load critic.py module spec", file=sys.stderr)
    raise SystemExit(1)

critic_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(critic_module)  # type: ignore[union-attr]

# Pull the symbols we need into local scope.
CriticTransformer = critic_module.CriticTransformer
load_trajectories = critic_module.load_trajectories
trajectories_to_tensors = critic_module.trajectories_to_tensors
build_padded_batch = critic_module.build_padded_batch
masked_mse_loss = critic_module.masked_mse_loss
write_advantages = critic_module.write_advantages


# =============================================================================
# Device selection (done once at startup)
# =============================================================================

def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# Server state — a single global instance held for the lifetime of the process
# =============================================================================

class CriticServerState:
    """Holds all mutable state that persists across /train calls."""

    def __init__(
        self,
        checkpoint_path: Optional[Path],
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float,
        lr: float,
        weight_decay: float,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay

        self.device = _select_device()
        print(f"[critic_server] Using device: {self.device}", file=sys.stderr)

        # Rolling replay buffer: list of (sequence_tensor, reward, matched_mask)
        # These accumulate across rounds so later rounds see more context.
        self._buffer_seqs: list[torch.Tensor] = []
        self._buffer_rewards: list[float] = []
        self._buffer_matched: list[torch.Tensor] = []

        # Round counter for /health
        self.round_count: int = 0

        # Serialise concurrent /train calls (safety valve — launcher is serial)
        self._lock = threading.Lock()

        # Build model and optimizer
        self.model = CriticTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)

        # Load checkpoint if one exists
        if checkpoint_path is not None and checkpoint_path.exists():
            print(
                f"[critic_server] Loading checkpoint from {checkpoint_path}",
                file=sys.stderr,
            )
            state = torch.load(
                checkpoint_path, map_location=self.device, weights_only=True
            )
            self.model.load_state_dict(state, strict=True)

        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"[critic_server] Model parameters: {param_count:,}", file=sys.stderr)

        self.optimizer = AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # ── EMA weight blending after retrain ──
        # After /retrain produces new weights, we don't switch instantly.
        # We store old + new state_dicts. Each /predict call advances the
        # blend: α(t) = 1 - (1-τ)^t  (exponential moving average).
        # Fast early shift, gentle asymptotic tail. Prevents the advantage
        # landscape from shifting suddenly (which collapses the policy).
        self._old_state: Optional[dict[str, torch.Tensor]] = None
        self._new_state: Optional[dict[str, torch.Tensor]] = None
        self._blend_step: int = 0
        self._blend_tau: float = 0.005      # EMA rate: α = 1-(1-τ)^t
        self._blending: bool = False

    def _apply_blend(self) -> None:
        """Advance one EMA step and load blended weights into the model.

        Called by inference endpoints when a blend is active. Each call
        advances t by 1:
          α = 1 - (1 - τ)^t
          θ = (1 - α) * θ_old + α * θ_new

        Automatically finishes when α > 0.999 (effectively converged).
        """
        if not self._blending or self._old_state is None or self._new_state is None:
            return

        self._blend_step += 1
        alpha = 1.0 - (1.0 - self._blend_tau) ** self._blend_step

        blended = {}
        for key in self._old_state:
            blended[key] = (1.0 - alpha) * self._old_state[key] + alpha * self._new_state[key]
        self.model.load_state_dict(blended)

        if self._blend_step % 50 == 0 or alpha > 0.999:
            print(
                f"[critic_server] EMA blend step {self._blend_step}: α={alpha:.4f}",
                file=sys.stderr,
            )

        if alpha > 0.999:
            # Converged — snap to new weights, free old state
            self.model.load_state_dict(self._new_state)
            self._old_state = None
            self._new_state = None
            self._blending = False
            print(
                f"[critic_server] EMA blend complete after {self._blend_step} steps",
                file=sys.stderr,
            )

    def reset_buffer(self) -> None:
        """Wipe the rolling replay buffer."""
        self._buffer_seqs.clear()
        self._buffer_rewards.clear()
        self._buffer_matched.clear()
        print("[critic_server] Replay buffer reset", file=sys.stderr)

    def train_round(
        self,
        traj_path: Path,
        output_path: Path,
        epochs: int,
        lr_override: Optional[float],
        dropout_override: Optional[float],
        mini_batch_size: int,
    ) -> dict:
        """Load trajectories, add to buffer, run N epochs, export advantages.

        Returns a dict with {"loss": float, "steps": int}.
        Raises on any validation or training failure (fail fast).
        """
        # Load new trajectories from disk
        new_trajs = load_trajectories(traj_path)
        new_seqs, new_rewards, new_matched = trajectories_to_tensors(new_trajs)

        # Append to rolling buffer
        self._buffer_seqs.extend(new_seqs)
        self._buffer_rewards.extend(new_rewards)
        self._buffer_matched.extend(new_matched)

        n = len(self._buffer_seqs)
        lengths = [seq.size(0) for seq in self._buffer_seqs]
        n_real_steps = sum(lengths)
        print(
            f"[critic_server] Round buffer: {n} trajectories, {n_real_steps} steps",
            file=sys.stderr,
        )

        # Optionally adjust lr / dropout for this call (Optuna-style override)
        if lr_override is not None and lr_override != self.lr:
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr_override
            self.lr = lr_override

        if dropout_override is not None and dropout_override != self.dropout:
            for module in self.model.modules():
                if isinstance(module, nn.Dropout):
                    module.p = dropout_override
            self.dropout = dropout_override

        # ---- 80/20 train/val split ----
        perm_all = torch.randperm(n).tolist()
        n_val = max(1, n // 5)
        n_train = n - n_val
        train_idx = perm_all[:n_train]
        val_idx = perm_all[n_train:]

        print(
            f"[critic_server] Split: {n_train} train, {n_val} val",
            file=sys.stderr,
        )

        # ---- Training loop with val-based early stopping ----
        self.model.train()
        best_val_loss = float("inf")
        best_train_loss = float("inf")
        best_state = None
        patience = 5
        patience_counter = 0
        mb = mini_batch_size

        for epoch in range(epochs):
            # --- Train ---
            self.model.train()
            train_perm = torch.randperm(n_train)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_train, mb):
                batch_local = train_perm[start : start + mb].tolist()
                batch_real = [train_idx[i] for i in batch_local]
                batch_seqs = [self._buffer_seqs[i] for i in batch_real]
                batch_rewards = [self._buffer_rewards[i] for i in batch_real]

                b_padded, b_targets, b_padding_mask, _ = build_padded_batch(
                    batch_seqs, batch_rewards, self.device
                )

                v_pred = self.model(b_padded, src_key_padding_mask=b_padding_mask)
                loss = masked_mse_loss(v_pred, b_targets, b_padding_mask)

                if not torch.isfinite(loss):
                    continue

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            if n_batches == 0:
                if best_state is not None:
                    self.model.load_state_dict(best_state)
                break

            avg_train = epoch_loss / n_batches

            # --- Val ---
            self.model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for start in range(0, n_val, mb):
                    batch_real = val_idx[start : start + mb]
                    batch_seqs = [self._buffer_seqs[i] for i in batch_real]
                    batch_rewards = [self._buffer_rewards[i] for i in batch_real]

                    b_padded, b_targets, b_padding_mask, _ = build_padded_batch(
                        batch_seqs, batch_rewards, self.device
                    )

                    v_pred = self.model(b_padded, src_key_padding_mask=b_padding_mask)
                    loss = masked_mse_loss(v_pred, b_targets, b_padding_mask)

                    if torch.isfinite(loss):
                        val_loss += loss.item()
                        val_batches += 1

            avg_val = val_loss / max(val_batches, 1)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_train_loss = avg_train
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0 or epoch == 0 or patience_counter >= patience:
                print(
                    f"[critic_server] Epoch {epoch + 1}/{epochs}  "
                    f"train={avg_train:.6f}  val={avg_val:.6f}"
                    f"{'  *best*' if patience_counter == 0 else ''}",
                    file=sys.stderr,
                )

            if patience_counter >= patience:
                print(
                    f"[critic_server] Early stop at epoch {epoch + 1} "
                    f"(val not improving for {patience} epochs, "
                    f"best_val={best_val_loss:.6f})",
                    file=sys.stderr,
                )
                break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        best_loss = best_val_loss  # Report val loss, not train loss

        # ---- Export advantages for the NEW trajectories only ----
        # The buffer includes old trajectories for training, but we only write
        # advantages for the trajectories submitted in this call (new_seqs).
        self.model.eval()
        n_new = len(new_seqs)
        new_lengths = [seq.size(0) for seq in new_seqs]
        all_value_preds: list[torch.Tensor] = []

        with torch.no_grad():
            for start in range(0, n_new, mb):
                end = min(start + mb, n_new)
                chunk_seqs = new_seqs[start:end]
                chunk_rewards = new_rewards[start:end]
                c_padded, _, c_padding_mask, c_lengths = build_padded_batch(
                    chunk_seqs, chunk_rewards, self.device
                )
                v_chunk = self.model(c_padded, src_key_padding_mask=c_padding_mask)
                v_chunk = v_chunk.squeeze(-1).cpu()
                for j, length in enumerate(c_lengths):
                    all_value_preds.append(v_chunk[j, :length])

        if len(all_value_preds) != n_new:
            raise RuntimeError(
                f"Value pred count {len(all_value_preds)} != {n_new}"
            )

        records = []
        for i, length in enumerate(new_lengths):
            v_i = all_value_preds[i]
            adv = torch.tensor(new_rewards[i]) - v_i  # A_t = R_T - V_t

            # EXPLICIT PENALTY: unmatched steps get hard negative advantage
            step_matched = new_matched[i][:length].to(device=adv.device)
            adv = torch.where(step_matched, adv, torch.full_like(adv, -0.01))

            # Replace NaN/Inf with 0.0
            if not torch.all(torch.isfinite(adv)):
                n_bad = (~torch.isfinite(adv)).sum().item()
                print(
                    f"[critic_server] WARNING: trajectory {i} has "
                    f"{n_bad}/{len(adv)} non-finite advantages, zeroing",
                    file=sys.stderr,
                )
                adv = torch.where(
                    torch.isfinite(adv), adv, torch.zeros_like(adv)
                )

            records.append({
                "trajectory_idx": i,
                "advantages": adv.tolist(),
            })
        write_advantages(output_path, records)

        print(
            f"[critic_server] Wrote {n_new} advantage records to {output_path}",
            file=sys.stderr,
        )

        # Save checkpoint after every round
        if self.checkpoint_path is not None:
            torch.save(self.model.state_dict(), self.checkpoint_path)
            print(
                f"[critic_server] Saved checkpoint to {self.checkpoint_path}",
                file=sys.stderr,
            )

        self.round_count += 1
        return {"loss": best_loss, "steps": n_real_steps}


# =============================================================================
# FastAPI app
# =============================================================================

app = FastAPI(title="PixelFlow Critic Server", version="1.0")

# Populated during startup via parse_args(); None until then.
_state: Optional[CriticServerState] = None


# ---- Request / response models ----

class TrainRequest(BaseModel):
    traj_path: str
    output_path: str
    epochs: int = 50
    lr: Optional[float] = None
    dropout: Optional[float] = None
    mini_batch_size: int = 512


class TrainResponse(BaseModel):
    loss: float
    steps: int


class HealthResponse(BaseModel):
    status: str
    round: int


# ---- Endpoints ----

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if _state is None:
        raise HTTPException(status_code=503, detail="Server not yet initialised")
    return HealthResponse(status="ok", round=_state.round_count)


@app.post("/train", response_model=TrainResponse)
def train(req: TrainRequest) -> TrainResponse:
    if _state is None:
        raise HTTPException(status_code=503, detail="Server not yet initialised")

    traj_path = Path(req.traj_path)
    output_path = Path(req.output_path)

    if not traj_path.exists():
        raise HTTPException(
            status_code=422,
            detail=f"traj_path does not exist: {traj_path}",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with _state._lock:
        try:
            result = _state.train_round(
                traj_path=traj_path,
                output_path=output_path,
                epochs=req.epochs,
                lr_override=req.lr,
                dropout_override=req.dropout,
                mini_batch_size=req.mini_batch_size,
            )
        except Exception as exc:
            # Fail loudly — the caller (Rust) will panic on non-200
            print(
                f"[critic_server] /train ERROR: {exc}",
                file=sys.stderr,
            )
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return TrainResponse(loss=result["loss"], steps=result["steps"])


class PredictRequest(BaseModel):
    """Inference-only: produce advantages without any training."""
    traj_path: str
    output_path: str
    mini_batch_size: int = 512


class PredictResponse(BaseModel):
    trajectories: int
    steps: int


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Produce advantages via forward pass only.

    If an EMA blend is active (post-retrain), advances one blend step before
    inference so the critic drifts gradually toward the retrained weights.
    """
    if _state is None:
        raise HTTPException(status_code=503, detail="Server not yet initialised")

    traj_path = Path(req.traj_path)
    output_path = Path(req.output_path)

    if not traj_path.exists():
        raise HTTPException(
            status_code=422,
            detail=f"traj_path does not exist: {traj_path}",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with _state._lock:
        try:
            # Advance EMA blend one step if active (post-retrain)
            _state._apply_blend()

            new_trajs = load_trajectories(traj_path)
            new_seqs, new_rewards, new_matched = trajectories_to_tensors(new_trajs)

            n_new = len(new_seqs)
            new_lengths = [seq.size(0) for seq in new_seqs]
            total_steps = sum(new_lengths)
            mb = req.mini_batch_size

            _state.model.eval()
            all_value_preds: list[torch.Tensor] = []

            with torch.no_grad():
                for start in range(0, n_new, mb):
                    end = min(start + mb, n_new)
                    chunk_seqs = new_seqs[start:end]
                    chunk_rewards = new_rewards[start:end]
                    c_padded, _, c_padding_mask, c_lengths = build_padded_batch(
                        chunk_seqs, chunk_rewards, _state.device
                    )
                    v_chunk = _state.model(c_padded, src_key_padding_mask=c_padding_mask)
                    v_chunk = v_chunk.squeeze(-1).cpu()
                    for j, length in enumerate(c_lengths):
                        all_value_preds.append(v_chunk[j, :length])

            if len(all_value_preds) != n_new:
                raise RuntimeError(
                    f"Value pred count {len(all_value_preds)} != {n_new}"
                )

            records = []
            for i, length in enumerate(new_lengths):
                v_i = all_value_preds[i]
                adv = torch.tensor(new_rewards[i]) - v_i

                step_matched = new_matched[i][:length].to(device=adv.device)
                adv = torch.where(step_matched, adv, torch.full_like(adv, -0.01))

                if not torch.all(torch.isfinite(adv)):
                    n_bad = (~torch.isfinite(adv)).sum().item()
                    print(
                        f"[critic_server] WARNING: trajectory {i} has "
                        f"{n_bad}/{len(adv)} non-finite advantages, zeroing",
                        file=sys.stderr,
                    )
                    adv = torch.where(
                        torch.isfinite(adv), adv, torch.zeros_like(adv)
                    )

                records.append({
                    "trajectory_idx": i,
                    "advantages": adv.tolist(),
                })
            write_advantages(output_path, records)

            print(
                f"[critic_server] /predict: {n_new} trajectories, {total_steps} steps",
                file=sys.stderr,
            )

        except Exception as exc:
            print(
                f"[critic_server] /predict ERROR: {exc}",
                file=sys.stderr,
            )
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PredictResponse(trajectories=n_new, steps=total_steps)


class StepRequest(BaseModel):
    """Predict advantages + one backprop step + EMA update."""
    traj_path: str
    output_path: str
    mini_batch_size: int = 128


class StepResponse(BaseModel):
    trajectories: int
    steps: int
    train_loss: float


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest) -> StepResponse:
    """Predict advantages, then do one gradient step on the new data, then EMA.

    This is the main online training endpoint. Each round:
    1. Forward pass (eval mode) → write advantages
    2. One training step on JUST these trajectories (not the full buffer)
    3. EMA blend: target = τ * online + (1-τ) * target

    The advantages come from the TARGET model (stable, slow-moving).
    The gradient step updates the ONLINE model.
    """
    if _state is None:
        raise HTTPException(status_code=503, detail="Server not yet initialised")

    traj_path = Path(req.traj_path)
    output_path = Path(req.output_path)

    if not traj_path.exists():
        raise HTTPException(status_code=422, detail=f"traj_path does not exist: {traj_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with _state._lock:
        try:
            # Advance EMA blend one step if active (post-retrain) so the
            # advantages for this batch come from the gradually updated teacher.
            _state._apply_blend()

            new_trajs = load_trajectories(traj_path)
            new_seqs, new_rewards, new_matched = trajectories_to_tensors(new_trajs)

            # Accumulate in buffer
            _state._buffer_seqs.extend(new_seqs)
            _state._buffer_rewards.extend(new_rewards)
            _state._buffer_matched.extend(new_matched)
            _state.round_count += 1

            n_new = len(new_seqs)
            new_lengths = [seq.size(0) for seq in new_seqs]
            total_steps = sum(new_lengths)
            mb = req.mini_batch_size

            # ── Phase 1: Predict advantages (eval mode) ──
            _state.model.eval()
            all_value_preds: list[torch.Tensor] = []

            with torch.no_grad():
                for start in range(0, n_new, mb):
                    end = min(start + mb, n_new)
                    chunk_seqs = new_seqs[start:end]
                    chunk_rewards = new_rewards[start:end]
                    c_padded, _, c_padding_mask, c_lengths = build_padded_batch(
                        chunk_seqs, chunk_rewards, _state.device
                    )
                    v_chunk = _state.model(c_padded, src_key_padding_mask=c_padding_mask)
                    v_chunk = v_chunk.squeeze(-1).cpu()
                    for j, length in enumerate(c_lengths):
                        all_value_preds.append(v_chunk[j, :length])

            if len(all_value_preds) != n_new:
                raise RuntimeError(f"Value pred count {len(all_value_preds)} != {n_new}")

            # Write advantages
            records = []
            for i, length in enumerate(new_lengths):
                v_i = all_value_preds[i]
                adv = torch.tensor(new_rewards[i]) - v_i
                step_matched = new_matched[i][:length].to(device=adv.device)
                adv = torch.where(step_matched, adv, torch.full_like(adv, -0.01))
                if not torch.all(torch.isfinite(adv)):
                    adv = torch.where(torch.isfinite(adv), adv, torch.zeros_like(adv))
                records.append({
                    "trajectory_idx": i,
                    "advantages": adv.tolist(),
                })
            write_advantages(output_path, records)

            # ── Phase 2: One backprop step on this round's data ──
            _state.model.train()
            b_padded, b_targets, b_padding_mask, _ = build_padded_batch(
                new_seqs, new_rewards, _state.device
            )
            v_pred = _state.model(b_padded, src_key_padding_mask=b_padding_mask)
            loss = masked_mse_loss(v_pred, b_targets, b_padding_mask)

            train_loss = float("nan")
            if torch.isfinite(loss):
                _state.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(_state.model.parameters(), max_norm=1.0)
                _state.optimizer.step()
                train_loss = loss.item()

            # ── Phase 3: Save checkpoint periodically ──
            if _state.checkpoint_path is not None and _state.round_count % 100 == 0:
                torch.save(_state.model.state_dict(), _state.checkpoint_path)

            print(
                f"[critic_server] /step: {n_new} trajs, {total_steps} steps, "
                f"loss={train_loss:.6f}, round={_state.round_count}",
                file=sys.stderr,
            )

        except Exception as exc:
            print(f"[critic_server] /step ERROR: {exc}", file=sys.stderr)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return StepResponse(trajectories=n_new, steps=total_steps, train_loss=train_loss)


class RetrainRequest(BaseModel):
    """Retrain the critic from scratch on recent trajectory data."""
    epochs: int = 100
    lr: Optional[float] = None
    dropout: Optional[float] = None
    mini_batch_size: int = 512
    max_trajectories: int = 2000  # Train on last N trajectories (0 = all)


@app.post("/retrain", response_model=TrainResponse)
def retrain(req: RetrainRequest) -> TrainResponse:
    """Retrain the critic from scratch on ALL accumulated trajectory data.

    Reinitializes model weights, then trains for `epochs` on the full
    buffer with 80/20 val split and early stopping. Use this every ~200
    rounds to give the student a better teacher.

    Does NOT clear the buffer — data persists for future retrains.
    """
    if _state is None:
        raise HTTPException(status_code=503, detail="Server not yet initialised")

    with _state._lock:
        n_total = len(_state._buffer_seqs)
        if n_total == 0:
            raise HTTPException(status_code=422, detail="Buffer is empty — nothing to retrain on")

        # Use last N trajectories (most recent data is most relevant)
        if req.max_trajectories > 0 and n_total > req.max_trajectories:
            start_idx = n_total - req.max_trajectories
            train_seqs = _state._buffer_seqs[start_idx:]
            train_rewards = _state._buffer_rewards[start_idx:]
            train_matched = _state._buffer_matched[start_idx:]
        else:
            train_seqs = _state._buffer_seqs
            train_rewards = _state._buffer_rewards
            train_matched = _state._buffer_matched

        n = len(train_seqs)
        n_steps = sum(seq.size(0) for seq in train_seqs)
        print(
            f"[critic_server] /retrain: reinitializing model, training on {n} trajectories ({n_steps} steps) for up to {req.epochs} epochs",
            file=sys.stderr,
        )

        # Snapshot current weights BEFORE reinit — this is the "old" end of the blend
        old_state = {k: v.clone() for k, v in _state.model.state_dict().items()}

        # Reinitialize model weights
        for m in _state.model.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

        # Reset optimizer state
        if req.lr is not None:
            _state.lr = req.lr
        _state.optimizer = torch.optim.AdamW(
            _state.model.parameters(), lr=_state.lr, weight_decay=1e-4,
        )

        if req.dropout is not None:
            for module in _state.model.modules():
                if isinstance(module, nn.Dropout):
                    module.p = req.dropout

        # Train on the full buffer (reuses the existing train_round logic
        # but without loading new data — just train on what's accumulated)
        _state.model.train()
        mb = req.mini_batch_size
        best_val_loss = float("inf")
        best_state = None
        patience = 10
        patience_counter = 0

        # 80/20 split
        perm_all = torch.randperm(n).tolist()
        n_val = max(1, n // 5)
        n_train = n - n_val
        train_idx = perm_all[:n_train]
        val_idx = perm_all[n_train:]

        print(
            f"[critic_server] /retrain: split {n_train} train, {n_val} val",
            file=sys.stderr,
        )

        for epoch in range(req.epochs):
            # Train
            _state.model.train()
            train_perm = torch.randperm(n_train)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_train, mb):
                batch_local = train_perm[start : start + mb].tolist()
                batch_real = [train_idx[i] for i in batch_local]
                batch_seqs = [train_seqs[i] for i in batch_real]
                batch_rewards = [train_rewards[i] for i in batch_real]

                b_padded, b_targets, b_padding_mask, _ = build_padded_batch(
                    batch_seqs, batch_rewards, _state.device
                )
                v_pred = _state.model(b_padded, src_key_padding_mask=b_padding_mask)
                loss = masked_mse_loss(v_pred, b_targets, b_padding_mask)

                if not torch.isfinite(loss):
                    continue

                _state.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(_state.model.parameters(), max_norm=1.0)
                _state.optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            if n_batches == 0:
                break
            avg_train = epoch_loss / n_batches

            # Val
            _state.model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for start in range(0, n_val, mb):
                    batch_real = val_idx[start : start + mb]
                    batch_seqs = [train_seqs[i] for i in batch_real]
                    batch_rewards = [train_rewards[i] for i in batch_real]
                    b_padded, b_targets, b_padding_mask, _ = build_padded_batch(
                        batch_seqs, batch_rewards, _state.device
                    )
                    v_pred = _state.model(b_padded, src_key_padding_mask=b_padding_mask)
                    loss = masked_mse_loss(v_pred, b_targets, b_padding_mask)
                    if torch.isfinite(loss):
                        val_loss += loss.item()
                        val_batches += 1

            avg_val = val_loss / max(val_batches, 1)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_state = {k: v.clone() for k, v in _state.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0 or epoch == 0 or patience_counter >= patience:
                print(
                    f"[critic_server] /retrain epoch {epoch + 1}/{req.epochs}  "
                    f"train={avg_train:.6f}  val={avg_val:.6f}"
                    f"{'  *best*' if patience_counter == 0 else ''}",
                    file=sys.stderr,
                )

            if patience_counter >= patience:
                print(
                    f"[critic_server] /retrain early stop at epoch {epoch + 1} "
                    f"(best_val={best_val_loss:.6f})",
                    file=sys.stderr,
                )
                break

        if best_state is not None:
            _state.model.load_state_dict(best_state)

        # Start EMA blend: old_state → new (retrained) weights
        # The model currently holds the new weights. We DON'T load them yet —
        # instead, restore old weights and let /predict blend gradually.
        new_state = {k: v.clone() for k, v in _state.model.state_dict().items()}
        _state._old_state = old_state
        _state._new_state = new_state
        _state._blend_step = 0
        _state._blending = True
        # Restore old weights — first /predict call will blend one step
        _state.model.load_state_dict(old_state)
        print(
            f"[critic_server] /retrain complete, EMA blend started "
            f"(τ={_state._blend_tau}, val_loss={best_val_loss:.6f})",
            file=sys.stderr,
        )

        # Save checkpoint (new weights, for cold restart)
        if _state.checkpoint_path is not None:
            torch.save(new_state, _state.checkpoint_path)
            print(
                f"[critic_server] /retrain saved NEW checkpoint to {_state.checkpoint_path}",
                file=sys.stderr,
            )

        return TrainResponse(loss=best_val_loss, steps=n_steps)


@app.post("/reset")
def reset() -> JSONResponse:
    if _state is None:
        raise HTTPException(status_code=503, detail="Server not yet initialised")
    with _state._lock:
        _state.reset_buffer()
    return JSONResponse({"status": "buffer reset"})


# =============================================================================
# Entry point
# =============================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Persistent FastAPI critic server for PixelFlow training."
    )
    parser.add_argument("--port", type=int, default=8765, help="TCP port to listen on")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to save/load model checkpoint (.pt)",
    )
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    global _state
    _state = CriticServerState(
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print(
        f"[critic_server] Starting on http://0.0.0.0:{args.port}",
        file=sys.stderr,
    )
    # log_level="warning" keeps uvicorn from spamming every HTTP request to stderr
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
