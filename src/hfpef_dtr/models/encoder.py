"""
Missingness-aware BiGRU encoder for ICU time-series representation learning.

Architecture
------------
Input: (batch, T=12, D_temporal) + (batch, D_static)
  ↓
  [Optional static embedding] → expand to T → concat with temporal
  ↓
  BiGRU(hidden_dim, n_layers) → h_t (batch, T, 2*hidden_dim)
  ↓
  Projection → z_t (batch, T, latent_dim)
  ↓
  [Decoder] Reconstruction head → x̂_t (batch, T, D_temporal)
  [Optional] Next-state prediction head → x̂_{t+1}
  [Optional] Mortality classification head → ŷ (batch, 1)

Protocol reference: Step 50a (v3.1 § 11 Step A representation learning)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class StaticEmbedding(nn.Module):
    """Project static covariates and expand to sequence length."""

    def __init__(self, d_static: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_static, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, static: torch.Tensor, seq_len: int) -> torch.Tensor:
        """(B, D_s) → (B, T, embed_dim)"""
        h = self.net(static)  # (B, embed_dim)
        return h.unsqueeze(1).expand(-1, seq_len, -1)  # (B, T, embed_dim)


class BiGRUEncoder(nn.Module):
    """Bidirectional GRU encoder producing per-timestep latent states."""

    def __init__(
        self,
        d_temporal: int,
        d_static: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
        static_embed_dim: int = 32,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.d_temporal = d_temporal
        self.d_static = d_static
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Static embedding
        self.has_static = d_static > 0
        if self.has_static:
            self.static_embed = StaticEmbedding(d_static, static_embed_dim)
            gru_input_dim = d_temporal + static_embed_dim
        else:
            gru_input_dim = d_temporal

        # Input projection + layer norm
        self.input_proj = nn.Linear(gru_input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()

        # BiGRU
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # Latent projection
        self.latent_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim) if use_layer_norm else nn.Identity(),
        )

    def forward(
        self,
        temporal: torch.Tensor,
        static: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        temporal : (B, T, D_temporal)
        static : (B, D_static) or None

        Returns
        -------
        z : (B, T, latent_dim)
        """
        B, T, _ = temporal.shape

        # Concat static embedding if available
        if self.has_static and static is not None:
            s_embed = self.static_embed(static, T)  # (B, T, static_embed_dim)
            x = torch.cat([temporal, s_embed], dim=-1)
        else:
            x = temporal

        # Input projection
        x = self.input_proj(x)
        x = self.input_norm(x)

        # BiGRU
        h, _ = self.gru(x)  # (B, T, 2*hidden_dim)

        # Latent projection
        z = self.latent_proj(h)  # (B, T, latent_dim)
        return z


class ReconstructionDecoder(nn.Module):
    """Decode latent z_t back to temporal features for masked reconstruction."""

    def __init__(
        self,
        latent_dim: int,
        d_output: int,
        hidden_dim: int = 128,
        n_layers: int = 1,
    ):
        super().__init__()
        layers = []
        in_dim = latent_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, d_output))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """(B, T, latent_dim) → (B, T, D_output)"""
        return self.net(z)


class NextStatePredHead(nn.Module):
    """Predict x_{t+1} from z_t."""

    def __init__(self, latent_dim: int, d_output: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_output),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """(B, T, latent_dim) → (B, T, D_output), used with z[:, :-1] to predict x[:, 1:]"""
        return self.net(z)


class MortalityHead(nn.Module):
    """Classify mortality from pooled latent representation."""

    def __init__(self, latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : (B, T, latent_dim)
        valid_mask : (B, T) — 1 where bin is valid

        Returns
        -------
        logit : (B, 1)
        """
        # Masked mean pooling
        mask = valid_mask.unsqueeze(-1)  # (B, T, 1)
        z_masked = z * mask
        z_pooled = z_masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B, latent_dim)
        return self.net(z_pooled)


class BaselineEncoderModel(nn.Module):
    """
    Full self-supervised encoder model with reconstruction + optional heads.

    Combines encoder + decoder + optional next-state + optional mortality heads.
    """

    def __init__(
        self,
        d_temporal: int,
        d_static: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
        static_embed_dim: int = 32,
        use_layer_norm: bool = True,
        decoder_hidden_dim: int = 128,
        decoder_n_layers: int = 1,
        enable_next_state: bool = True,
        enable_mortality: bool = False,
    ):
        super().__init__()
        self.d_temporal = d_temporal
        self.latent_dim = latent_dim

        self.encoder = BiGRUEncoder(
            d_temporal=d_temporal,
            d_static=d_static,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_layers=n_layers,
            dropout=dropout,
            static_embed_dim=static_embed_dim,
            use_layer_norm=use_layer_norm,
        )

        self.recon_decoder = ReconstructionDecoder(
            latent_dim=latent_dim,
            d_output=d_temporal,
            hidden_dim=decoder_hidden_dim,
            n_layers=decoder_n_layers,
        )

        self.enable_next_state = enable_next_state
        if enable_next_state:
            self.next_state_head = NextStatePredHead(
                latent_dim=latent_dim,
                d_output=d_temporal,
                hidden_dim=decoder_hidden_dim,
            )

        self.enable_mortality = enable_mortality
        if enable_mortality:
            self.mortality_head = MortalityHead(latent_dim=latent_dim)

    def forward(
        self,
        temporal: torch.Tensor,
        static: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Returns dict with keys: z, recon, [next_state], [mortality_logit]
        """
        z = self.encoder(temporal, static)  # (B, T, latent_dim)
        recon = self.recon_decoder(z)       # (B, T, D_temporal)

        out = {"z": z, "recon": recon}

        if self.enable_next_state:
            # Predict x_{t+1} from z_t (use bins 0..T-2 to predict 1..T-1)
            out["next_state"] = self.next_state_head(z[:, :-1])

        if self.enable_mortality and valid_mask is not None:
            out["mortality_logit"] = self.mortality_head(z, valid_mask)

        return out


class GradientReversalFunction(torch.autograd.Function):
    """Gradient reversal layer for domain-adversarial training (DANN)."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    """Wraps GradientReversalFunction for use in nn.Sequential."""

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

    def set_alpha(self, alpha: float):
        self.alpha = alpha


class DomainDiscriminator(nn.Module):
    """
    Domain classifier for adversarial representation learning.

    Takes latent z (pooled or per-bin) and predicts source domain.
    Attached via gradient reversal layer to make encoder domain-invariant.
    """

    def __init__(
        self,
        latent_dim: int,
        n_domains: int = 2,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.grl = GradientReversalLayer()

        layers = []
        in_dim = latent_dim
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, n_domains))
        self.classifier = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """(B, latent_dim) → (B, n_domains) logits."""
        return self.classifier(self.grl(z))

    def set_alpha(self, alpha: float):
        self.grl.set_alpha(alpha)


class DANNEncoderModel(nn.Module):
    """
    Domain-Adversarial encoder model.

    Same encoder backbone as BaselineEncoderModel, but with an additional
    domain discriminator attached via gradient reversal. During training:
    - Encoder minimizes reconstruction + next-state + domain confusion
    - Discriminator maximizes domain classification accuracy
    - Net effect: encoder learns domain-invariant representations.

    Protocol reference: Step 55a (v3.1 § 11)
    """

    def __init__(
        self,
        d_temporal: int,
        d_static: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
        static_embed_dim: int = 32,
        use_layer_norm: bool = True,
        decoder_hidden_dim: int = 128,
        decoder_n_layers: int = 1,
        enable_next_state: bool = True,
        enable_mortality: bool = False,
        n_domains: int = 2,
        domain_hidden_dim: int = 64,
        domain_n_layers: int = 2,
        domain_dropout: float = 0.2,
    ):
        super().__init__()
        self.d_temporal = d_temporal
        self.latent_dim = latent_dim

        self.encoder = BiGRUEncoder(
            d_temporal=d_temporal,
            d_static=d_static,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_layers=n_layers,
            dropout=dropout,
            static_embed_dim=static_embed_dim,
            use_layer_norm=use_layer_norm,
        )

        self.recon_decoder = ReconstructionDecoder(
            latent_dim=latent_dim,
            d_output=d_temporal,
            hidden_dim=decoder_hidden_dim,
            n_layers=decoder_n_layers,
        )

        self.enable_next_state = enable_next_state
        if enable_next_state:
            self.next_state_head = NextStatePredHead(
                latent_dim=latent_dim,
                d_output=d_temporal,
                hidden_dim=decoder_hidden_dim,
            )

        self.enable_mortality = enable_mortality
        if enable_mortality:
            self.mortality_head = MortalityHead(latent_dim=latent_dim)

        # Domain adversary
        self.domain_disc = DomainDiscriminator(
            latent_dim=latent_dim,
            n_domains=n_domains,
            hidden_dim=domain_hidden_dim,
            n_layers=domain_n_layers,
            dropout=domain_dropout,
        )

    def forward(
        self,
        temporal: torch.Tensor,
        static: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Returns dict with: z, recon, [next_state], [mortality_logit], domain_logits.

        domain_logits uses mean-pooled z through gradient reversal layer.
        """
        z = self.encoder(temporal, static)  # (B, T, latent_dim)
        recon = self.recon_decoder(z)

        out = {"z": z, "recon": recon}

        if self.enable_next_state:
            out["next_state"] = self.next_state_head(z[:, :-1])

        if self.enable_mortality and valid_mask is not None:
            out["mortality_logit"] = self.mortality_head(z, valid_mask)

        # Domain classification on mean-pooled z
        if valid_mask is not None:
            mask = valid_mask.unsqueeze(-1)
            z_pooled = (z * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            z_pooled = z.mean(dim=1)
        out["domain_logits"] = self.domain_disc(z_pooled)

        return out

    def set_dann_alpha(self, alpha: float):
        """Set gradient reversal strength (schedule during training)."""
        self.domain_disc.set_alpha(alpha)

    def load_baseline_weights(self, baseline_ckpt: dict):
        """Warm-start from 50a baseline encoder checkpoint."""
        own_state = self.state_dict()
        loaded = 0
        for name, param in baseline_ckpt["model_state_dict"].items():
            if name in own_state and own_state[name].shape == param.shape:
                own_state[name].copy_(param)
                loaded += 1
        return loaded


def build_dann_model_from_config(
    d_temporal: int,
    d_static: int,
    model_cfg: dict,
    pretext_cfg: dict,
    dann_cfg: dict,
) -> DANNEncoderModel:
    """Construct DANN model from parsed YAML config."""
    return DANNEncoderModel(
        d_temporal=d_temporal,
        d_static=d_static,
        hidden_dim=model_cfg.get("hidden_dim", 128),
        latent_dim=model_cfg.get("latent_dim", 64),
        n_layers=model_cfg.get("n_layers", 2),
        dropout=model_cfg.get("dropout", 0.1),
        static_embed_dim=model_cfg.get("static_embed_dim", 32),
        use_layer_norm=model_cfg.get("use_layer_norm", True),
        decoder_hidden_dim=model_cfg.get("decoder_hidden_dim", 128),
        decoder_n_layers=model_cfg.get("decoder_n_layers", 1),
        enable_next_state=pretext_cfg.get("next_state_prediction", {}).get("enabled", True),
        enable_mortality=pretext_cfg.get("mortality_head", {}).get("enabled", False),
        n_domains=dann_cfg.get("n_domains", 2),
        domain_hidden_dim=dann_cfg.get("domain_hidden_dim", 64),
        domain_n_layers=dann_cfg.get("domain_n_layers", 2),
        domain_dropout=dann_cfg.get("domain_dropout", 0.2),
    )


def build_model_from_config(
    d_temporal: int,
    d_static: int,
    model_cfg: dict,
    pretext_cfg: dict,
) -> BaselineEncoderModel:
    """Construct model from parsed YAML config."""
    return BaselineEncoderModel(
        d_temporal=d_temporal,
        d_static=d_static,
        hidden_dim=model_cfg.get("hidden_dim", 128),
        latent_dim=model_cfg.get("latent_dim", 64),
        n_layers=model_cfg.get("n_layers", 2),
        dropout=model_cfg.get("dropout", 0.1),
        static_embed_dim=model_cfg.get("static_embed_dim", 32),
        use_layer_norm=model_cfg.get("use_layer_norm", True),
        decoder_hidden_dim=model_cfg.get("decoder_hidden_dim", 128),
        decoder_n_layers=model_cfg.get("decoder_n_layers", 1),
        enable_next_state=pretext_cfg.get("next_state_prediction", {}).get("enabled", True),
        enable_mortality=pretext_cfg.get("mortality_head", {}).get("enabled", False),
    )
