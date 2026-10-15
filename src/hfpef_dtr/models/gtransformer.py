"""
G-Transformer: Transformer-based counterfactual outcome prediction for DTR.

Architecture
------------
Input per time step t:
  - State s_t: (D_temporal,) temporal features + (D_static,) static covariates
  - Action a_t: discrete integer → learned embedding

Encoder: causal (autoregressive) Transformer
  (s_0, a_0, s_1, a_1, ..., s_{T-1}, a_{T-1}) → hidden representations h_t

Heads:
  - Next-state prediction: h_t → ŝ_{t+1} (MSE loss, teacher-forced)
  - Outcome prediction: h_{T_last} → P(Y=1) (BCE loss on mortality_28d)

Counterfactual evaluation:
  - Replace observed a_t with π_rule(s_t) actions
  - Roll forward using learned transition model
  - Read off predicted outcome → Direct Method (DM) policy value

Protocol reference: Step 80c (v3.1 § Step C primary deep pair)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal bin indices."""

    def __init__(self, d_model: int, max_len: int = 24):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) → (B, T, d_model)"""
        return x + self.pe[:, : x.size(1)]


class GTransformer(nn.Module):
    """
    G-Transformer for sequential counterfactual prediction.

    Forward pass (teacher-forced training):
      Given (s_0..T-1, a_0..T-1), predict (s_1..T, Y).

    Counterfactual rollout:
      Replace actions with π_target, use predicted states for future steps.
    """

    def __init__(
        self,
        d_temporal: int,
        d_static: int,
        n_actions: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        action_embed_dim: int = 16,
        static_embed_dim: int = 32,
        n_bins: int = 12,
    ):
        super().__init__()
        self.d_temporal = d_temporal
        self.d_static = d_static
        self.n_actions = n_actions
        self.d_model = d_model
        self.n_bins = n_bins

        # ── Input projections ──
        self.state_proj = nn.Linear(d_temporal, d_model)
        self.action_embed = nn.Embedding(n_actions, action_embed_dim)

        # Static embedding (expanded to each time step)
        self.has_static = d_static > 0
        if self.has_static:
            self.static_embed = nn.Sequential(
                nn.Linear(d_static, static_embed_dim),
                nn.ReLU(),
                nn.Linear(static_embed_dim, static_embed_dim),
            )
        else:
            static_embed_dim = 0

        # Fusion: state + action + static → d_model
        fusion_dim = d_model + action_embed_dim + static_embed_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, max_len=n_bins)

        # ── Causal Transformer encoder ──
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # Causal mask (upper-triangular)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(n_bins, n_bins), diagonal=1).bool(),
        )

        # ── Output heads ──
        # Next-state prediction: h_t → ŝ_{t+1}
        self.next_state_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_temporal),
        )

        # Outcome prediction: pooled representation → P(mortality)
        self.outcome_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def _encode(
        self,
        temporal: torch.Tensor,
        static: torch.Tensor | None,
        action: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode (state, action) sequence through transformer.

        Returns: (B, T, d_model) hidden representations.
        """
        B, T, _ = temporal.shape

        # State projection
        s = self.state_proj(temporal)  # (B, T, d_model)

        # Action embedding
        a = self.action_embed(action)  # (B, T, action_embed_dim)

        # Static embedding
        if self.has_static and static is not None:
            st = self.static_embed(static)  # (B, static_embed_dim)
            st = st.unsqueeze(1).expand(-1, T, -1)  # (B, T, static_embed_dim)
            x = torch.cat([s, a, st], dim=-1)  # (B, T, fusion_dim)
        else:
            x = torch.cat([s, a], dim=-1)

        # Fusion + positional encoding
        x = self.fusion(x)  # (B, T, d_model)
        x = self.pos_enc(x)

        # Transformer with causal mask
        # Key padding mask: True where invalid
        key_padding_mask = valid_mask == 0  # (B, T)
        causal = self.causal_mask[:T, :T]

        h = self.transformer(x, mask=causal, src_key_padding_mask=key_padding_mask)
        return h

    def forward(
        self,
        temporal: torch.Tensor,
        static: torch.Tensor | None,
        action: torch.Tensor,
        valid_mask: torch.Tensor,
        absorbing_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Teacher-forced forward pass.

        Returns dict:
          h: (B, T, d_model)
          next_state_pred: (B, T-1, D_temporal) — predicted s_{t+1} from h_t
          outcome_logit: (B, 1) — mortality prediction
        """
        h = self._encode(temporal, static, action, valid_mask)

        # Next-state: use h_{0..T-2} to predict s_{1..T-1}
        next_state_pred = self.next_state_head(h[:, :-1])  # (B, T-1, D_temporal)

        # Outcome: masked mean pool over valid non-absorbing bins
        active_mask = (valid_mask * absorbing_mask).unsqueeze(-1)  # (B, T, 1)
        h_pooled = (h * active_mask).sum(dim=1) / active_mask.sum(dim=1).clamp(min=1)
        outcome_logit = self.outcome_head(h_pooled)  # (B, 1)

        return {
            "h": h,
            "next_state_pred": next_state_pred,
            "outcome_logit": outcome_logit,
        }

    @torch.no_grad()
    def predict_counterfactual(
        self,
        temporal: torch.Tensor,
        static: torch.Tensor | None,
        target_action: torch.Tensor,
        valid_mask: torch.Tensor,
        absorbing_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Counterfactual prediction under target policy actions.

        For the Direct Method (DM): substitute observed actions with
        target_action and predict outcome.

        In this simplified version we use teacher-forced encoding with
        target actions (no autoregressive rollout), which is valid when
        treatment does not affect future state observations in the data
        (i.e., we condition on observed states but intervene on actions).
        This is the standard G-computation/G-Transformer approach.

        For a full autoregressive rollout, see predict_counterfactual_rollout().
        """
        h = self._encode(temporal, static, target_action, valid_mask)

        active_mask = (valid_mask * absorbing_mask).unsqueeze(-1)
        h_pooled = (h * active_mask).sum(dim=1) / active_mask.sum(dim=1).clamp(min=1)
        outcome_logit = self.outcome_head(h_pooled)
        outcome_prob = torch.sigmoid(outcome_logit)

        return {
            "outcome_logit": outcome_logit,
            "outcome_prob": outcome_prob,
        }

    @torch.no_grad()
    def predict_counterfactual_rollout(
        self,
        temporal_t0: torch.Tensor,
        static: torch.Tensor | None,
        target_actions: torch.Tensor,
        valid_mask: torch.Tensor,
        absorbing_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Full autoregressive counterfactual rollout.

        Starting from observed s_0, at each step:
          1. Encode (s_0..t, a_0..t) → h_t
          2. Predict s_{t+1} = next_state_head(h_t)
          3. Use predicted s_{t+1} for next step

        This is more faithful to the G-Transformer formulation but
        accumulates prediction error over time. Use with caution for T>6.
        """
        B, T, D = temporal_t0.shape
        device = temporal_t0.device

        # Start with observed first time step
        states = temporal_t0.clone()
        predicted_states = torch.zeros_like(temporal_t0)
        predicted_states[:, 0] = temporal_t0[:, 0]

        for t in range(T - 1):
            # Encode up to step t
            h = self._encode(
                predicted_states[:, : t + 1],
                static,
                target_actions[:, : t + 1],
                valid_mask[:, : t + 1],
            )
            # Predict next state
            s_next = self.next_state_head(h[:, t: t + 1])  # (B, 1, D)
            predicted_states[:, t + 1] = s_next.squeeze(1)

        # Final encoding with all predicted states
        h = self._encode(predicted_states, static, target_actions, valid_mask)
        active_mask = (valid_mask * absorbing_mask).unsqueeze(-1)
        h_pooled = (h * active_mask).sum(dim=1) / active_mask.sum(dim=1).clamp(min=1)
        outcome_logit = self.outcome_head(h_pooled)

        return {
            "outcome_logit": outcome_logit,
            "outcome_prob": torch.sigmoid(outcome_logit),
            "predicted_states": predicted_states,
        }


class GTransformerLoss(nn.Module):
    """
    Combined loss for G-Transformer training.

    L = w_ns * L_next_state + w_out * L_outcome
    """

    def __init__(
        self,
        weight_next_state: float = 1.0,
        weight_outcome: float = 1.0,
    ):
        super().__init__()
        self.w_ns = weight_next_state
        self.w_out = weight_outcome

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Compute losses.

        Returns dict with: loss (total), loss_next_state, loss_outcome.
        """
        # Next-state MSE (only on valid, non-absorbing bins)
        ns_pred = outputs["next_state_pred"]  # (B, T-1, D)
        ns_target = batch["temporal"][:, 1:]  # (B, T-1, D) — observed s_{1..T-1}
        ns_mask = (batch["valid_mask"][:, 1:] * batch["absorbing_mask"][:, 1:]).unsqueeze(-1)  # (B, T-1, 1)

        ns_diff = (ns_pred - ns_target) ** 2 * ns_mask
        loss_ns = ns_diff.sum() / ns_mask.sum().clamp(min=1)

        # Outcome BCE
        outcome_logit = outputs["outcome_logit"].squeeze(-1)  # (B,)
        outcome_target = batch["mortality_28d"]  # (B,)
        loss_out = F.binary_cross_entropy_with_logits(outcome_logit, outcome_target)

        loss = self.w_ns * loss_ns + self.w_out * loss_out

        return {
            "loss": loss,
            "loss_next_state": loss_ns.detach(),
            "loss_outcome": loss_out.detach(),
        }


def build_gtransformer_from_config(
    d_temporal: int,
    d_static: int,
    n_actions: int,
    model_cfg: dict,
    grid_cfg: dict,
) -> GTransformer:
    """Construct G-Transformer from parsed YAML config."""
    return GTransformer(
        d_temporal=d_temporal,
        d_static=d_static,
        n_actions=n_actions,
        d_model=model_cfg.get("d_model", 128),
        n_heads=model_cfg.get("n_heads", 4),
        n_layers=model_cfg.get("n_layers", 4),
        d_ff=model_cfg.get("d_ff", 256),
        dropout=model_cfg.get("dropout", 0.1),
        action_embed_dim=model_cfg.get("action_embed_dim", 16),
        n_bins=grid_cfg.get("n_bins", 12),
    )
