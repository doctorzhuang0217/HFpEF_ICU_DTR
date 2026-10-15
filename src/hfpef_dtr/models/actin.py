"""
ACTIN: Adversarial Counterfactual Temporal Inference Network.

Architecture
------------
Input per time step t:
  - State s_t: (D_temporal,) temporal features + (D_static,) static covariates
  - Action a_t: discrete integer → learned embedding

Encoder: GRU with action-conditioned state transitions
  (s_0, a_0, s_1, a_1, ...) → hidden representations h_t

Heads:
  - Outcome prediction: pooled h → P(Y=1)
  - Balancing discriminator (adversarial): h_t → P(a_t | h_t)
    Trained to predict treatment assignment from representations.
    The encoder is trained to FOOL the discriminator (gradient reversal),
    encouraging treatment-invariant representations.

Key difference from G-Net/G-Transformer:
  - No next-state prediction head
  - Adversarial balancing replaces G-computation for confounding adjustment
  - Cannot produce counterfactual rollouts → no DM evaluation
  - OPE relies on DR (importance-weighted) and FQE (Q-network)

Protocol reference: Step 80b (v3.1 § Step C confirmatory)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# ── Gradient Reversal Layer ──────────────────────────────────────────


class _GradientReversalFn(Function):
    """Reverses gradient during backward pass, scales by lambda."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def gradient_reversal(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    """Apply gradient reversal layer."""
    return _GradientReversalFn.apply(x, lambda_)


# ── Model ────────────────────────────────────────────────────────────


class ACTIN(nn.Module):
    """
    ACTIN: Adversarial balancing model for DTR outcome prediction.

    The encoder learns treatment-invariant representations via adversarial
    training against a treatment discriminator. This encourages the outcome
    head to rely on prognostic (not treatment-predictive) features.

    Training uses gradient reversal: the encoder minimizes outcome loss
    while maximizing the discriminator's treatment prediction loss.
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
        balancing_hidden_dim: int = 64,
        balancing_n_layers: int = 2,
    ):
        super().__init__()
        self.d_temporal = d_temporal
        self.d_static = d_static
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
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

        # ── Outcome head ──
        self.outcome_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # ── Balancing discriminator (adversarial) ──
        # Predicts treatment assignment from representation
        bal_layers = []
        in_dim = hidden_dim
        for _ in range(balancing_n_layers):
            bal_layers.extend([
                nn.Linear(in_dim, balancing_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = balancing_hidden_dim
        bal_layers.append(nn.Linear(in_dim, n_actions))
        self.balancing_head = nn.Sequential(*bal_layers)

    def _encode(
        self,
        temporal: torch.Tensor,
        static: torch.Tensor | None,
        action: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode (state, action) sequence through GRU → (B, T, hidden_dim)."""
        B, T, _ = temporal.shape

        a = self.action_embed(action)

        if self.has_static and static is not None:
            st = self.static_embed(static).unsqueeze(1).expand(-1, T, -1)
            x = torch.cat([temporal, a, st], dim=-1)
        else:
            x = torch.cat([temporal, a], dim=-1)

        x = self.input_proj(x)
        h, _ = self.gru(x)
        return h

    def forward(
        self,
        temporal: torch.Tensor,
        static: torch.Tensor | None,
        action: torch.Tensor,
        valid_mask: torch.Tensor,
        absorbing_mask: torch.Tensor,
        grl_lambda: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass with adversarial balancing.

        grl_lambda controls the gradient reversal strength.
        Typically annealed from 0 → 1 during training.
        """
        h = self._encode(temporal, static, action, valid_mask)

        # Outcome: masked mean pool
        active_mask = (valid_mask * absorbing_mask).unsqueeze(-1)
        h_pooled = (h * active_mask).sum(dim=1) / active_mask.sum(dim=1).clamp(min=1)
        outcome_logit = self.outcome_head(h_pooled)

        # Balancing: per-timestep treatment prediction with gradient reversal
        # GRL makes encoder learn representations that are NOT predictive of treatment
        h_reversed = gradient_reversal(h, grl_lambda)
        balancing_logits = self.balancing_head(h_reversed)  # (B, T, n_actions)

        return {
            "h": h,
            "outcome_logit": outcome_logit,
            "balancing_logits": balancing_logits,
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
        """Predict outcome probability (no balancing head)."""
        h = self._encode(temporal, static, action, valid_mask)

        active_mask = (valid_mask * absorbing_mask).unsqueeze(-1)
        h_pooled = (h * active_mask).sum(dim=1) / active_mask.sum(dim=1).clamp(min=1)
        outcome_logit = self.outcome_head(h_pooled)

        return {
            "outcome_logit": outcome_logit,
            "outcome_prob": torch.sigmoid(outcome_logit),
        }

    # OPE interface compatibility: evaluate_dr / evaluate_dm call predict_counterfactual
    predict_counterfactual = predict_outcome


class ACTINLoss(nn.Module):
    """
    Combined loss for ACTIN training.

    L = w_out * L_outcome + w_bal * L_balancing

    The balancing loss is cross-entropy on treatment prediction.
    Via gradient reversal, the encoder learns to minimize outcome loss
    while making the balancing head's job harder.
    """

    def __init__(self, weight_outcome: float = 1.0, weight_balancing: float = 1.0):
        super().__init__()
        self.w_out = weight_outcome
        self.w_bal = weight_balancing

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        # Outcome BCE
        outcome_logit = outputs["outcome_logit"].squeeze(-1)
        loss_out = F.binary_cross_entropy_with_logits(
            outcome_logit, batch["mortality_28d"]
        )

        # Balancing cross-entropy (per valid timestep)
        bal_logits = outputs["balancing_logits"]  # (B, T, n_actions)
        action = batch["action"]  # (B, T)
        vmask = batch["valid_mask"]  # (B, T)
        amask = batch["absorbing_mask"]  # (B, T)
        active = vmask * amask  # (B, T)

        B, T, C = bal_logits.shape
        # Flatten and select only valid + non-absorbing steps.
        # This avoids including padded steps in CE computation.
        bal_flat = bal_logits.reshape(B * T, C)
        act_flat = action.reshape(B * T)
        active_flat = active.reshape(B * T) > 0.5

        if active_flat.any():
            loss_bal = F.cross_entropy(bal_flat[active_flat], act_flat[active_flat])
        else:
            # Keep computational graph valid in degenerate all-masked batches.
            loss_bal = bal_flat.sum() * 0.0

        loss = self.w_out * loss_out + self.w_bal * loss_bal

        return {
            "loss": loss,
            "loss_outcome": loss_out.detach(),
            "loss_balancing": loss_bal.detach(),
        }


def build_actin_from_config(
    d_temporal: int,
    d_static: int,
    n_actions: int,
    model_cfg: dict,
    grid_cfg: dict,
) -> ACTIN:
    """Construct ACTIN from parsed YAML config."""
    return ACTIN(
        d_temporal=d_temporal,
        d_static=d_static,
        n_actions=n_actions,
        hidden_dim=model_cfg.get("hidden_dim", 128),
        n_layers=model_cfg.get("n_layers", 2),
        dropout=model_cfg.get("dropout", 0.1),
        n_bins=grid_cfg.get("n_bins", 12),
        balancing_hidden_dim=model_cfg.get("balancing_hidden_dim", 64),
        balancing_n_layers=model_cfg.get("balancing_n_layers", 2),
    )
