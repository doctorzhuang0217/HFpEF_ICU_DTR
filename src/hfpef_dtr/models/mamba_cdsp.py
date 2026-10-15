"""
Mamba-CDSP: Selective State-Space Model for causal DTR outcome prediction.

Architecture
------------
Uses a simplified Mamba-style selective state-space layer instead of
Transformer attention or GRU recurrence. Key properties:
  - Linear-time sequence processing (vs quadratic for Transformers)
  - Input-dependent (selective) state transitions via learned gates
  - Causal by construction (unidirectional SSM scan)

"CDSP" = Causal Dynamic State Propagation — the SSM parameters
(Δ, B, C) are conditioned on the input at each time step, allowing
the model to selectively remember or forget information.

Role: Optional model (Step 80e). Provides architectural diversity
beyond attention (80c/80d) and recurrence (80a/80b).

OPE: FQE only.

Protocol reference: Step 80e (v3.1 § Step C optional)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSMLayer(nn.Module):
    """
    Simplified selective state-space layer (Mamba-style).

    At each time step, the model computes input-dependent parameters:
      - Δ_t: discretisation step (controls how much new input enters state)
      - B_t: input-to-state projection
      - C_t: state-to-output projection

    State update (discretised):
      h_t = (1 - Δ_t) * h_{t-1} + Δ_t * B_t * x_t
      y_t = C_t * h_t

    This is a simplified version that captures the key selective-scan
    idea without the full Mamba hardware-aware implementation.
    """

    def __init__(self, d_model: int, d_state: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Input-dependent parameter projections
        self.delta_proj = nn.Sequential(
            nn.Linear(d_model, d_state),
            nn.Softplus(),  # Δ must be positive
        )
        self.B_proj = nn.Linear(d_model, d_state)
        self.C_proj = nn.Linear(d_model, d_state)

        # Output projection: d_state → d_model
        self.out_proj = nn.Linear(d_state, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x: (B, T, d_model) → (B, T, d_model)

        Performs selective scan (causal by construction).
        """
        B, T, D = x.shape
        residual = x
        x_normed = self.norm(x)

        # Compute input-dependent SSM parameters
        delta = self.delta_proj(x_normed)  # (B, T, d_state) — discretisation
        B_t = self.B_proj(x_normed)        # (B, T, d_state) — input gate
        C_t = self.C_proj(x_normed)        # (B, T, d_state) — output gate

        # Selective scan (sequential — causal by construction)
        h = torch.zeros(B, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(T):
            # Discretised state update
            decay = torch.exp(-delta[:, t])  # (B, d_state) — close to 1 means remember
            h_candidate = decay * h + (1 - decay) * B_t[:, t]  # (B, d_state)

            if valid_mask is not None:
                m_t = valid_mask[:, t].unsqueeze(-1).to(x.dtype)
                # Keep previous state for padded timesteps to avoid bias-driven drift.
                h = m_t * h_candidate + (1 - m_t) * h
            else:
                h = h_candidate

            y_t = (C_t[:, t] * h).unsqueeze(1)  # (B, 1, d_state)

            if valid_mask is not None:
                y_t = y_t * valid_mask[:, t].view(B, 1, 1).to(x.dtype)

            outputs.append(y_t)

        y = torch.cat(outputs, dim=1)  # (B, T, d_state)
        y = self.out_proj(y)           # (B, T, d_model)
        y = self.dropout(y)

        return residual + y


class MambaCDSP(nn.Module):
    """
    Mamba-CDSP: Selective state-space model for DTR outcome prediction.

    Stacks multiple SelectiveSSMLayer blocks with outcome prediction head.
    No next-state prediction — outcome-only (like CT-Legacy).
    """

    def __init__(
        self,
        d_temporal: int,
        d_static: int,
        n_actions: int,
        d_model: int = 128,
        d_state: int = 64,
        n_layers: int = 4,
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
        self.action_embed = nn.Embedding(n_actions, action_embed_dim)

        self.has_static = d_static > 0
        if self.has_static:
            self.static_embed = nn.Sequential(
                nn.Linear(d_static, static_embed_dim),
                nn.ReLU(),
                nn.Linear(static_embed_dim, static_embed_dim),
            )
        else:
            static_embed_dim = 0

        input_dim = d_temporal + action_embed_dim + static_embed_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Stacked SSM layers ──
        self.ssm_layers = nn.ModuleList([
            SelectiveSSMLayer(d_model, d_state, dropout=dropout)
            for _ in range(n_layers)
        ])

        # ── Outcome head ──
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
        """Encode (state, action) sequence → (B, T, d_model)."""
        B, T, _ = temporal.shape

        a = self.action_embed(action)

        if self.has_static and static is not None:
            st = self.static_embed(static).unsqueeze(1).expand(-1, T, -1)
            x = torch.cat([temporal, a, st], dim=-1)
        else:
            x = torch.cat([temporal, a], dim=-1)

        x = self.input_proj(x)

        # Zero out invalid positions before SSM scan
        x = x * valid_mask.unsqueeze(-1)

        for layer in self.ssm_layers:
            x = layer(x, valid_mask=valid_mask)

        # Keep explicit zero at padded positions for stable pooling / downstream OPE.
        x = x * valid_mask.unsqueeze(-1)

        return x

    def forward(
        self,
        temporal: torch.Tensor,
        static: torch.Tensor | None,
        action: torch.Tensor,
        valid_mask: torch.Tensor,
        absorbing_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass — outcome prediction only."""
        h = self._encode(temporal, static, action, valid_mask)

        active_mask = (valid_mask * absorbing_mask).unsqueeze(-1)
        h_pooled = (h * active_mask).sum(dim=1) / active_mask.sum(dim=1).clamp(min=1)
        outcome_logit = self.outcome_head(h_pooled)

        return {
            "h": h,
            "outcome_logit": outcome_logit,
        }

    @torch.no_grad()
    def predict_outcome(
        self,
        temporal: torch.Tensor,
        static: torch.Tensor | None,
        action: torch.Tensor,
        valid_mask: torch.Tensor,
        absorbing_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Predict outcome probability."""
        h = self._encode(temporal, static, action, valid_mask)

        active_mask = (valid_mask * absorbing_mask).unsqueeze(-1)
        h_pooled = (h * active_mask).sum(dim=1) / active_mask.sum(dim=1).clamp(min=1)
        outcome_logit = self.outcome_head(h_pooled)

        return {
            "outcome_logit": outcome_logit,
            "outcome_prob": torch.sigmoid(outcome_logit),
        }

    # OPE interface compatibility
    predict_counterfactual = predict_outcome


class MambaCDSPLoss(nn.Module):
    """Outcome-only loss for Mamba-CDSP."""

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        outcome_logit = outputs["outcome_logit"].squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(
            outcome_logit, batch["mortality_28d"]
        )
        return {
            "loss": loss,
            "loss_outcome": loss.detach(),
        }


def build_mamba_cdsp_from_config(
    d_temporal: int,
    d_static: int,
    n_actions: int,
    model_cfg: dict,
    grid_cfg: dict,
) -> MambaCDSP:
    """Construct Mamba-CDSP from parsed YAML config."""
    return MambaCDSP(
        d_temporal=d_temporal,
        d_static=d_static,
        n_actions=n_actions,
        d_model=model_cfg.get("d_model", 128),
        d_state=model_cfg.get("d_state", 64),
        n_layers=model_cfg.get("n_layers", 4),
        dropout=model_cfg.get("dropout", 0.1),
        n_bins=grid_cfg.get("n_bins", 12),
    )
