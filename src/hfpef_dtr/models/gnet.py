"""
G-Net: RNN-based G-computation model for DTR outcome prediction.

Architecture
------------
Input per time step t:
  - State s_t: (D_temporal,) temporal features + (D_static,) static covariates
  - Action a_t: discrete integer → learned embedding

Encoder: GRU with action-conditioned state transitions
  (s_0, a_0, s_1, a_1, ...) → hidden representations h_t

Heads:
  - Next-state prediction: h_t → ŝ_{t+1}
  - Outcome prediction: h_{T_last} → P(Y=1)

Role: Confirmatory model (Step 80a). Simpler architecture than G-Transformer,
serves as a consistency check for 80c results.

Protocol reference: Step 80a (v3.1 § Step C confirmatory)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GNet(nn.Module):
    """
    G-Net: GRU-based sequential model for counterfactual outcome prediction.

    Simpler alternative to G-Transformer. Uses GRU instead of self-attention,
    providing a different inductive bias (recurrence vs attention).
    """

    def __init__(
        self,
        d_temporal: int,
        d_static: int,
        n_actions: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
        action_embed_dim: int = 16,
        static_embed_dim: int = 32,
        n_bins: int = 12,
    ):
        super().__init__()
        self.d_temporal = d_temporal
        self.d_static = d_static
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.n_bins = n_bins
        self.n_layers = n_layers

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

        # Input fusion: temporal + action + static → hidden_dim
        input_dim = d_temporal + action_embed_dim + static_embed_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── GRU encoder ──
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # ── Output heads ──
        self.next_state_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_temporal),
        )

        self.outcome_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _encode(
        self,
        temporal: torch.Tensor,
        static: torch.Tensor | None,
        action: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode (state, action) sequence through GRU.

        Runs full-length GRU over all T bins. Invalid bins (valid_mask=0)
        contain zero-filled features, which the GRU processes as "no activity".
        Masking is applied downstream in pooling and loss computation.

        Note: pack_padded_sequence is NOT used because valid_mask may have
        internal gaps (e.g. [1,1,0,1,...]) — pack assumes a contiguous prefix.

        Returns: (B, T, hidden_dim) hidden representations.
        """
        B, T, _ = temporal.shape

        # Action embedding
        a = self.action_embed(action)  # (B, T, action_embed_dim)

        # Static embedding
        if self.has_static and static is not None:
            st = self.static_embed(static).unsqueeze(1).expand(-1, T, -1)
            x = torch.cat([temporal, a, st], dim=-1)
        else:
            x = torch.cat([temporal, a], dim=-1)

        x = self.input_proj(x)  # (B, T, hidden_dim)

        # Full-length GRU (no packing — valid_mask may have internal gaps)
        h, _ = self.gru(x)
        return h  # (B, T, hidden_dim)

    def forward(
        self,
        temporal: torch.Tensor,
        static: torch.Tensor | None,
        action: torch.Tensor,
        valid_mask: torch.Tensor,
        absorbing_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Teacher-forced forward pass."""
        h = self._encode(temporal, static, action, valid_mask)

        # Next-state prediction
        next_state_pred = self.next_state_head(h[:, :-1])  # (B, T-1, D_temporal)

        # Outcome: masked mean pool
        active_mask = (valid_mask * absorbing_mask).unsqueeze(-1)
        h_pooled = (h * active_mask).sum(dim=1) / active_mask.sum(dim=1).clamp(min=1)
        outcome_logit = self.outcome_head(h_pooled)

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
        """Counterfactual prediction under target policy (G-computation)."""
        h = self._encode(temporal, static, target_action, valid_mask)

        active_mask = (valid_mask * absorbing_mask).unsqueeze(-1)
        h_pooled = (h * active_mask).sum(dim=1) / active_mask.sum(dim=1).clamp(min=1)
        outcome_logit = self.outcome_head(h_pooled)

        return {
            "outcome_logit": outcome_logit,
            "outcome_prob": torch.sigmoid(outcome_logit),
        }


class GNetLoss(nn.Module):
    """Combined loss for G-Net training (same structure as GTransformerLoss)."""

    def __init__(self, weight_next_state: float = 1.0, weight_outcome: float = 1.0):
        super().__init__()
        self.w_ns = weight_next_state
        self.w_out = weight_outcome

    def forward(self, outputs: dict, batch: dict) -> dict[str, torch.Tensor]:
        ns_pred = outputs["next_state_pred"]
        ns_target = batch["temporal"][:, 1:]
        ns_mask = (batch["valid_mask"][:, 1:] * batch["absorbing_mask"][:, 1:]).unsqueeze(-1)

        ns_diff = (ns_pred - ns_target) ** 2 * ns_mask
        loss_ns = ns_diff.sum() / ns_mask.sum().clamp(min=1)

        outcome_logit = outputs["outcome_logit"].squeeze(-1)
        loss_out = F.binary_cross_entropy_with_logits(outcome_logit, batch["mortality_28d"])

        loss = self.w_ns * loss_ns + self.w_out * loss_out

        return {
            "loss": loss,
            "loss_next_state": loss_ns.detach(),
            "loss_outcome": loss_out.detach(),
        }


def build_gnet_from_config(
    d_temporal: int,
    d_static: int,
    n_actions: int,
    model_cfg: dict,
    grid_cfg: dict,
) -> GNet:
    """Construct G-Net from parsed YAML config."""
    return GNet(
        d_temporal=d_temporal,
        d_static=d_static,
        n_actions=n_actions,
        hidden_dim=model_cfg.get("hidden_dim", 128),
        n_layers=model_cfg.get("n_layers", 2),
        dropout=model_cfg.get("dropout", 0.1),
        n_bins=grid_cfg.get("n_bins", 12),
    )
