"""
CT-Legacy: Causal Transformer ablation model (outcome-only, no G-computation).

Architecture
------------
Identical to G-Transformer *except*:
  - No next-state prediction head (outcome-only Transformer)
  - Smaller capacity (d_model=64, n_layers=3, d_ff=128 by default)
  - No counterfactual rollout capability

Role: Legacy ablation (Step 80d). Demonstrates that the full G-Transformer
with next-state prediction and G-computation rollout provides additional
value over a naive outcome-only Transformer.

OPE: FQE only (no DM — model doesn't learn state transitions).

Protocol reference: Step 80d (v3.1 § Step C legacy ablation)
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
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class CTLegacy(nn.Module):
    """
    Causal Transformer — outcome-only ablation.

    Same encoder as G-Transformer but without next-state prediction.
    Serves as a baseline to quantify the benefit of explicit state
    transition modelling (G-computation).
    """

    def __init__(
        self,
        d_temporal: int,
        d_static: int,
        n_actions: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 128,
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

        self.has_static = d_static > 0
        if self.has_static:
            self.static_embed = nn.Sequential(
                nn.Linear(d_static, static_embed_dim),
                nn.ReLU(),
                nn.Linear(static_embed_dim, static_embed_dim),
            )
        else:
            static_embed_dim = 0

        fusion_dim = d_model + action_embed_dim + static_embed_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

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
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(n_bins, n_bins), diagonal=1).bool(),
        )

        # ── Outcome head only (no next-state) ──
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

        s = self.state_proj(temporal)
        a = self.action_embed(action)

        if self.has_static and static is not None:
            st = self.static_embed(static).unsqueeze(1).expand(-1, T, -1)
            x = torch.cat([s, a, st], dim=-1)
        else:
            x = torch.cat([s, a], dim=-1)

        x = self.fusion(x)
        x = self.pos_enc(x)

        key_padding_mask = valid_mask == 0
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


class CTLegacyLoss(nn.Module):
    """Outcome-only loss for CT-Legacy (no next-state component)."""

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


def build_ct_legacy_from_config(
    d_temporal: int,
    d_static: int,
    n_actions: int,
    model_cfg: dict,
    grid_cfg: dict,
) -> CTLegacy:
    """Construct CT-Legacy from parsed YAML config."""
    return CTLegacy(
        d_temporal=d_temporal,
        d_static=d_static,
        n_actions=n_actions,
        d_model=model_cfg.get("d_model", 64),
        n_heads=model_cfg.get("n_heads", 4),
        n_layers=model_cfg.get("n_layers", 3),
        d_ff=model_cfg.get("d_ff", 128),
        dropout=model_cfg.get("dropout", 0.1),
        n_bins=grid_cfg.get("n_bins", 12),
    )
