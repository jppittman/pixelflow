#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch>=2.1",
#     "numpy>=1.26.0",
# ]
# ///
"""Causal Sequence Transformer Critic for temporal credit assignment.

Reads self-play trajectories (`.pftraj`), trains a value model V_t for each
step, exports per-step advantages A_t = R_T - V_t.

Variable-length trajectories are batched with pad_sequence + padding masks
for efficient GPU matrix multiplication. Device auto-selected: CUDA > MPS > CPU.

Usage:
    uv run critic.py train --input trajectories.pftraj --output advantages.pfadv
    uv run critic.py train --input trajectories.pftraj --output advantages.pfadv --checkpoint critic.pt
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW

# =============================================================================
# Constants — must match pixelflow-pipeline/src/training/unified.rs
# =============================================================================

GRAPH_ACC_DIM = 132  # GraphAccumulator (saturation/search state — changes per step)
MASK_STATS_DIM = 6   # frac_approved, frac_matched, log1p_unions, mean_prob, budget_norm, epochs_norm
STEP_DIM = GRAPH_ACC_DIM + MASK_STATS_DIM  # 138 floats per step
# TODO: add RULE_DIM=32 (mean approved rule_embed) when rule embedding table is added;
# that will bring STEP_DIM from 138 to 170.
# NOTE: EdgeAccumulator deliberately excluded — it encodes the INITIAL
# expression (frozen for the whole trajectory), not the search state.
# Giving it to the critic lets it shortcut credit assignment by just
# predicting cost directly (val_loss=0.008 = zero-signal advantages).


# =============================================================================
# SinusoidalPE — standard sinusoidal positional encoding
# =============================================================================

class SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al. 2017)."""

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        self.d_model = d_model
        pe = self._build_pe(d_model, max_len)
        # (1, max_len, d_model) — registered as buffer, not parameter
        self.register_buffer("pe", pe.unsqueeze(0))

    @staticmethod
    def _build_pe(d_model: int, max_len: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding. x: (batch, seq_len, d_model)."""
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            # Grow PE buffer on demand — no silent truncation
            new_pe = self._build_pe(self.d_model, seq_len).unsqueeze(0).to(self.pe.device)
            self.pe = new_pe
        return x + self.pe[:, :seq_len]


# =============================================================================
# CriticTransformer — causal transformer value network
# =============================================================================

class CriticTransformer(nn.Module):
    """Causal Transformer that predicts per-step value V_t.

    Each step sees only itself and prior steps (causal mask).
    Padding positions are masked via src_key_padding_mask so batches of
    variable-length trajectories can be processed in a single forward pass.
    The value head outputs a scalar V_t per step.
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(STEP_DIM, d_model)
        self.pe = SinusoidalPE(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            enable_nested_tensor=False,  # MPS doesn't support nested tensor ops
        )
        self.value_head = nn.Linear(d_model, 1)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, STEP_DIM) step features (may include padding).
            src_key_padding_mask: (batch, seq_len) bool tensor where True
                marks padding positions that should be ignored by attention.

        Returns:
            (batch, seq_len, 1) predicted values V_t.
        """
        h = self.input_proj(x)
        h = self.pe(h)
        T = h.size(1)
        # Causal mask: position t can attend to positions <= t.
        # TransformerEncoder expects True = "block this position" for bool masks.
        causal = torch.triu(
            torch.ones(T, T, device=h.device, dtype=torch.bool), diagonal=1
        )
        # causal mask (T, T) blocks future; src_key_padding_mask (B, T)
        # blocks padding keys. PyTorch combines them correctly — a position
        # is blocked if EITHER mask says so.
        h = self.transformer(
            h, mask=causal, src_key_padding_mask=src_key_padding_mask
        )
        return self.value_head(h)  # (batch, seq_len, 1)


# =============================================================================
# Data loading — fail fast, fail loudly
# =============================================================================

TRAJ_MAGIC = b"PFTJ0002"
ADV_MAGIC = b"PFAD0001"


class BinaryReader:
    def __init__(self, data: bytes, path: Path):
        self.data = data
        self.path = path
        self.off = 0

    def take(self, n: int) -> bytes:
        end = self.off + n
        if end > len(self.data):
            raise ValueError(
                f"Unexpected EOF in {self.path} at byte {self.off}, wanted {n} bytes"
            )
        chunk = self.data[self.off:end]
        self.off = end
        return chunk

    def u8(self) -> int:
        return self.take(1)[0]

    def u32(self) -> int:
        return struct.unpack_from("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack_from("<Q", self.take(8))[0]

    def i32(self) -> int:
        return struct.unpack_from("<i", self.take(4))[0]

    def f32(self) -> float:
        return struct.unpack_from("<f", self.take(4))[0]

    def f64(self) -> float:
        return struct.unpack_from("<d", self.take(8))[0]

    def string(self) -> str:
        n = self.u32()
        return self.take(n).decode("utf-8")

    def f32_vec(self) -> list[float]:
        n = self.u32()
        if n == 0:
            return []
        values = struct.unpack_from(f"<{n}f", self.take(4 * n))
        return list(values)

    def edges(self) -> list[tuple[int, int, int]]:
        n = self.u32()
        out = []
        for _ in range(n):
            parent = self.u8()
            child = self.u8()
            depth = struct.unpack_from("<H", self.take(2))[0]
            out.append((parent, child, depth))
        return out


def _load_trajectories_binary(path: Path) -> list[dict]:
    data = path.read_bytes()
    r = BinaryReader(data, path)
    magic = r.take(8)
    if magic != TRAJ_MAGIC:
        raise ValueError(
            f"Invalid trajectory binary magic in {path}: {magic!r} "
            f"(expected {TRAJ_MAGIC!r}; PFTJ0001 files are no longer supported)"
        )

    trajectories: list[dict] = []
    for _ in range(r.u32()):
        trajectory_id = r.string()
        step_count = r.u32()
        initial_cost_ns = r.f64()
        final_cost_ns = r.f64()
        initial_cost = r.f32() if r.u8() else None
        final_cost = r.f32() if r.u8() else None
        initial_nodes = r.u64()
        node_budget = r.u64()
        initial_accumulator_state = r.f32_vec()
        initial_edges = r.edges()
        final_accumulator_state = r.f32_vec()
        final_edges = r.edges()

        # PFTJ0002: intermediate extraction-head pairs
        n_intermediate_pairs = r.u32()
        intermediate_pairs: list[tuple[list[float], list[tuple], float]] = []
        for _ in range(n_intermediate_pairs):
            acc_state = r.f32_vec()
            pair_edges = r.edges()
            cost_ns = r.f64()
            intermediate_pairs.append((acc_state, pair_edges, cost_ns))

        # PFTJ0002: epoch-granular steps (one step = one mask decision epoch)
        steps = []
        for _step in range(step_count):
            graph_accumulator_state = r.f32_vec()

            mask_len = r.u32()
            mask: list[tuple[int, float, bool]] = []
            for _ in range(mask_len):
                rule_idx = r.u32()
                action_prob = r.f32()
                approved = bool(r.u8())
                mask.append((rule_idx, action_prob, approved))

            rule_outcomes_len = r.u32()
            rule_outcomes: list[tuple[int, int]] = []
            for _ in range(rule_outcomes_len):
                outcome_rule_idx = r.u32()
                unions_produced = r.u32()
                rule_outcomes.append((outcome_rule_idx, unions_produced))

            budget_remaining = r.i32()
            epochs_remaining = r.i32()
            jit_cost_ns = r.f64()

            steps.append({
                "graph_accumulator_state": graph_accumulator_state,
                "mask": mask,
                "rule_outcomes": rule_outcomes,
                "budget_remaining": budget_remaining,
                "epochs_remaining": epochs_remaining,
                "jit_cost_ns": jit_cost_ns,
            })

        trajectories.append({
            "trajectory_id": trajectory_id,
            "steps": steps,
            "initial_cost_ns": initial_cost_ns,
            "final_cost_ns": final_cost_ns,
            "initial_cost": initial_cost,
            "final_cost": final_cost,
            "initial_nodes": initial_nodes,
            "node_budget": node_budget,
            "initial_accumulator_state": initial_accumulator_state,
            "initial_edges": initial_edges,
            "final_accumulator_state": final_accumulator_state,
            "final_edges": final_edges,
            "intermediate_pairs": intermediate_pairs,
        })

    if r.off != len(data):
        raise ValueError(f"Trailing bytes in {path}: {len(data) - r.off}")
    if not trajectories:
        raise ValueError(f"No trajectories found in {path} (file is empty)")
    print(
        f"Loaded {len(trajectories)} binary trajectories from {path}",
        file=sys.stderr,
    )
    return trajectories


def write_advantages(path: Path, records: list[dict]) -> None:
    """Write advantages as binary `.pfadv`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(ADV_MAGIC)
        f.write(struct.pack("<I", len(records)))
        for record in records:
            adv = record["advantages"]
            f.write(struct.pack("<Q", int(record["trajectory_idx"])))
            f.write(struct.pack("<I", len(adv)))
            if adv:
                f.write(struct.pack(f"<{len(adv)}f", *[float(x) for x in adv]))


def load_trajectories(path: Path) -> list[dict]:
    """Load binary trajectories."""
    if not path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {path}")
    return _load_trajectories_binary(path)


def trajectories_to_tensors(
    trajectories: list[dict],
) -> tuple[list[torch.Tensor], list[float], list[torch.Tensor]]:
    """Convert trajectories to per-trajectory tensors.

    Returns:
        sequences: list of (seq_len_i, STEP_DIM) tensors (variable length).
        rewards: list of terminal rewards (log-ns domain).
        matched: list of (seq_len_i) boolean tensors indicating if rule applied.
    """
    sequences: list[torch.Tensor] = []
    rewards: list[float] = []
    matched_masks: list[torch.Tensor] = []

    for traj in trajectories:
        steps = traj["steps"]
        step_features = []
        step_matches = []
        for step in steps:
            # GraphAccumulator: the search state that changes per epoch-step.
            # This is what the policy head saw when deciding which rules to approve.
            # We deliberately EXCLUDE the EdgeAccumulator — it encodes the initial
            # expression (frozen for the whole trajectory) and lets the critic
            # shortcut credit assignment by predicting cost directly.
            gacc = list(step["graph_accumulator_state"])
            if len(gacc) != GRAPH_ACC_DIM:
                raise ValueError(
                    f"graph_accumulator_state has {len(gacc)} floats, expected "
                    f"{GRAPH_ACC_DIM} (trajectory_id={traj.get('trajectory_id', '<unknown>')})"
                )

            # Normalize VSA sections [0..128] by 1/sqrt(node_count)
            edge_count = gacc[128]
            node_count = gacc[129]
            scale = 1.0 / math.sqrt(max(1.0, node_count))
            for i in range(128):
                gacc[i] *= scale
            node_budget = gacc[130]
            epoch_budget = gacc[131]
            gacc[128] = math.log2(1.0 + edge_count)
            gacc[129] = math.log2(1.0 + node_count)
            gacc[130] = math.log2(1.0 + node_budget)
            gacc[131] = math.log2(1.0 + epoch_budget)

            # Mask summary stats (6 floats)
            mask = step["mask"]
            rule_outcomes = step["rule_outcomes"]
            n_rules = len(mask)
            n_approved = sum(1 for _, _, approved in mask if approved)
            n_matched = sum(1 for _, unions in rule_outcomes if unions > 0)
            total_unions = sum(unions for _, unions in rule_outcomes)

            frac_approved = n_approved / n_rules if n_rules > 0 else 0.0
            n_ran = len(rule_outcomes)
            frac_matched = n_matched / n_ran if n_ran > 0 else 0.0
            log1p_unions = math.log1p(total_unions)
            mean_prob = sum(p for _, p, _ in mask) / n_rules if n_rules > 0 else 0.0
            budget_norm = float(step["budget_remaining"]) / 1000.0   # rough normalization
            epochs_norm = float(step["epochs_remaining"]) / 20.0    # rough normalization

            mask_stats = [frac_approved, frac_matched, log1p_unions, mean_prob, budget_norm, epochs_norm]

            features = gacc[:GRAPH_ACC_DIM] + mask_stats
            if len(features) != STEP_DIM:
                raise ValueError(
                    f"Step feature length {len(features)} != {STEP_DIM} "
                    f"(trajectory_id={traj['trajectory_id']})"
                )
            step_features.append(features)

            # A step is "matched" if any rule produced at least one union
            step_matched = any(unions > 0 for _, unions in rule_outcomes)
            step_matches.append(step_matched)

        if not step_features:
            raise ValueError(
                f"Trajectory {traj.get('trajectory_id', '<unknown>')!r} has zero steps — "
                "binary data may be corrupt"
            )

        seq = torch.tensor(step_features, dtype=torch.float32)
        sequences.append(seq)
        
        matches = torch.tensor(step_matches, dtype=torch.bool)
        matched_masks.append(matches)

        # Terminal reward in log-ns domain: relative improvements matter
        # equally at 1ns and 100ns.  Floor at 0.5ns prevents -inf.
        # Matches the convention in train_judge.rs: ln(max(ns, 0.5)).
        reward = -math.log(max(traj["final_cost_ns"], 0.5))
        rewards.append(reward)

    return sequences, rewards, matched_masks


def build_padded_batch(
    sequences: list[torch.Tensor],
    rewards: list[float],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """Pad variable-length sequences into a single batch.

    Returns:
        padded: (B, T_max, STEP_DIM) padded input features.
        targets: (B, T_max, 1) per-step targets (reward broadcast, 0 at padding).
        padding_mask: (B, T_max) bool — True at padding positions.
        lengths: list of original sequence lengths per trajectory.
    """
    lengths = [seq.size(0) for seq in sequences]
    if not lengths:
        raise ValueError("Cannot build batch from empty sequence list")

    # pad_sequence: pads shorter sequences with 0.0 to match longest
    # Input: list of (T_i, STEP_DIM), Output: (B, T_max, STEP_DIM)
    padded = pad_sequence(sequences, batch_first=True, padding_value=0.0).to(device)
    B, T_max, _ = padded.shape

    # Padding mask: True where position >= original length (padding)
    # Shape: (B, T_max)
    arange = torch.arange(T_max, device=device).unsqueeze(0)  # (1, T_max)
    len_tensor = torch.tensor(lengths, device=device).unsqueeze(1)  # (B, 1)
    padding_mask = arange >= len_tensor  # (B, T_max)

    # Targets: each real position gets the trajectory's terminal reward.
    # Padding positions get 0.0 (masked out of loss anyway).
    targets = torch.zeros(B, T_max, 1, device=device, dtype=torch.float32)
    for i, (length, reward) in enumerate(zip(lengths, rewards)):
        targets[i, :length, 0] = reward

    return padded, targets, padding_mask, lengths


def masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """MSE loss that ignores padding positions.

    Args:
        pred: (B, T, 1) predicted values.
        target: (B, T, 1) target values.
        padding_mask: (B, T) bool — True at padding positions.

    Returns:
        Scalar mean MSE over non-padding positions. Raises if all positions
        are padding (should never happen with validated data).
    """
    # real_mask: (B, T, 1) — True at real (non-padding) positions
    real_mask = ~padding_mask.unsqueeze(-1)  # (B, T, 1)
    n_real = real_mask.sum()
    if n_real == 0:
        raise RuntimeError(
            "masked_mse_loss: all positions are padding — "
            "this should never happen with validated trajectories"
        )
    sq_err = (pred - target) ** 2
    # Zero out padding contributions, sum, divide by count of real positions
    return (sq_err * real_mask).sum() / n_real


# =============================================================================
# Training + advantage export
# =============================================================================

def train_and_export(args: argparse.Namespace) -> None:
    """Train the Critic and export per-step advantages."""
    # Select device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}", file=sys.stderr)

    # Load data
    trajectories = load_trajectories(Path(args.input))
    sequences, rewards, matched_masks = trajectories_to_tensors(trajectories)

    # Pre-compute sequence lengths for use in the export phase
    lengths = [seq.size(0) for seq in sequences]
    n = len(sequences)
    n_real_steps = sum(lengths)
    print(
        f"Dataset: {n} trajectories, {n_real_steps} real steps",
        file=sys.stderr,
    )

    # Build model
    model = CriticTransformer(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    # Optionally load checkpoint — strict=True so dimension mismatches crash immediately
    if args.checkpoint and Path(args.checkpoint).exists():
        print(
            f"Loading checkpoint from {args.checkpoint}", file=sys.stderr
        )
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=True)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}", file=sys.stderr)

    optimizer = AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # ---- Mini-batch training loop ----
    model.train()
    best_loss = float("inf")
    best_state = None
    mb = args.mini_batch_size

    for epoch in range(args.epochs):
        # Shuffle trajectory indices each epoch
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n, mb):
            batch_idx = perm[start : start + mb].tolist()
            batch_seqs = [sequences[i] for i in batch_idx]
            batch_rewards = [rewards[i] for i in batch_idx]

            b_padded, b_targets, b_padding_mask, _ = build_padded_batch(
                batch_seqs, batch_rewards, device
            )

            v_pred = model(b_padded, src_key_padding_mask=b_padding_mask)
            loss = masked_mse_loss(v_pred, b_targets, b_padding_mask)

            if not torch.isfinite(loss):
                print(
                    f"Epoch {epoch + 1}/{args.epochs} mini-batch {n_batches}: "
                    f"loss=NaN — skipping batch",
                    file=sys.stderr,
                )
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if n_batches == 0:
            print(
                f"Epoch {epoch + 1}/{args.epochs}: all batches NaN — reverting to best",
                file=sys.stderr,
            )
            if best_state is not None:
                model.load_state_dict(best_state)
            break

        avg_loss = epoch_loss / n_batches
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1}/{args.epochs}  loss={avg_loss:.6f}",
                file=sys.stderr,
            )

    # ---- Export advantages ----
    output_path = Path(args.output)
    model.eval()
    all_value_preds: list[torch.Tensor] = []  # list of (T_i,) tensors, one per trajectory

    with torch.no_grad():
        for start in range(0, n, mb):
            end = min(start + mb, n)
            chunk_seqs = sequences[start:end]
            chunk_rewards = rewards[start:end]
            c_padded, _, c_padding_mask, c_lengths = build_padded_batch(
                chunk_seqs, chunk_rewards, device
            )
            v_chunk = model(c_padded, src_key_padding_mask=c_padding_mask)  # (B, T, 1)
            v_chunk = v_chunk.squeeze(-1).cpu()  # (B, T)
            for j, length in enumerate(c_lengths):
                all_value_preds.append(v_chunk[j, :length])  # (T_i,)

    if len(all_value_preds) != n:
        raise RuntimeError(
            f"Value pred count {len(all_value_preds)} != {n}"
        )

    records = []
    for i, length in enumerate(lengths):
        v_i = all_value_preds[i]          # (T_i,)
        adv = torch.tensor(rewards[i]) - v_i   # A_t = R_T - V_t

        # EXPLICIT PENALTY: unmatched steps get hard negative advantage
        step_matched = matched_masks[i][:length].to(device=adv.device)
        adv = torch.where(step_matched, adv, torch.full_like(adv, -0.01))

        # Replace NaN/Inf with 0.0
        if not torch.all(torch.isfinite(adv)):
            n_bad = (~torch.isfinite(adv)).sum().item()
            print(
                f"WARNING: trajectory {i} has {n_bad}/{len(adv)} non-finite advantages, zeroing them",
                file=sys.stderr,
            )
            adv = torch.where(torch.isfinite(adv), adv, torch.zeros_like(adv))

        records.append({
            "trajectory_idx": i,
            "advantages": adv.tolist(),
        })
    write_advantages(output_path, records)

    print(
        f"Wrote {len(sequences)} advantage records to {output_path}",
        file=sys.stderr,
    )

    # ---- Save checkpoint ----
    if args.checkpoint:
        torch.save(model.state_dict(), args.checkpoint)
        print(f"Saved checkpoint to {args.checkpoint}", file=sys.stderr)


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Causal Sequence Transformer Critic for temporal credit assignment.",
    )
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser(
        "train", help="Train critic and export advantages"
    )
    train_parser.add_argument(
        "--input", required=True, help="Path to trajectory batch (.pftraj)"
    )
    train_parser.add_argument(
        "--output", required=True, help="Path to write advantages batch (.pfadv)"
    )
    train_parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to save/load model checkpoint (.pt)",
    )
    train_parser.add_argument(
        "--epochs", type=int, default=50, help="Training epochs (default: 50)"
    )
    train_parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)"
    )
    train_parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay (default: 1e-5)",
    )
    train_parser.add_argument(
        "--d-model",
        type=int,
        default=128,
        help="Transformer model dimension (default: 128)",
    )
    train_parser.add_argument(
        "--nhead",
        type=int,
        default=4,
        help="Number of attention heads (default: 4)",
    )
    train_parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="Number of transformer layers (default: 3)",
    )
    train_parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate (default: 0.1)",
    )
    train_parser.add_argument(
        "--mini-batch-size",
        type=int,
        default=64,
        help="Trajectories per mini-batch during training (default: 64)",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help(sys.stderr)
        raise SystemExit(1)

    if args.command == "train":
        train_and_export(args)
    else:
        raise ValueError(f"Unknown command: {args.command!r}")


if __name__ == "__main__":
    main()
