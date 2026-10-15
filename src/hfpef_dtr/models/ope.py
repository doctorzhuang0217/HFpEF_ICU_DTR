"""
Off-Policy Evaluation (OPE) module for Step C deep causal models.

Implements:
  - Direct Method (DM): G-computation via learned outcome model
  - Doubly-Robust (DR): combines propensity weighting with outcome model
  - Fitted Q-Evaluation (FQE): external Q-network evaluation

All methods operate on test-set data and produce:
  - Point estimate of policy value V(π)
  - ATE = V(π_rule) - V(π_observed)
  - Bootstrap confidence intervals

Protocol reference: configs/70e_ope_config_v1.yaml
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


# ── Direct Method (DM) ────────────────────────────────────────────────


def evaluate_dm(
    model,
    dataset,
    device: torch.device,
    batch_size: int = 256,
    use_rollout: bool = False,
) -> dict:
    """
    Direct Method: estimate E[Y | do(π_rule)] using learned outcome model.

    For each stay, substitute observed actions with rule-policy actions
    and predict counterfactual outcome.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    obs_probs = []
    cf_probs = []

    for batch in tqdm(loader, desc="    DM eval", leave=False, mininterval=2):
        temporal = batch["temporal"].to(device)
        static = batch["static"].to(device)
        action = batch["action"].to(device)
        rule_action = batch["rule_action"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        absorbing_mask = batch["absorbing_mask"].to(device)

        # Observed policy outcome
        if use_rollout:
            out_obs = model.predict_counterfactual_rollout(
                temporal, static, action, valid_mask, absorbing_mask
            )
        else:
            out_obs = model.predict_counterfactual(
                temporal, static, action, valid_mask, absorbing_mask
            )
        obs_probs.append(out_obs["outcome_prob"].cpu().numpy())

        # Rule policy outcome
        if use_rollout:
            out_cf = model.predict_counterfactual_rollout(
                temporal, static, rule_action, valid_mask, absorbing_mask
            )
        else:
            out_cf = model.predict_counterfactual(
                temporal, static, rule_action, valid_mask, absorbing_mask
            )
        cf_probs.append(out_cf["outcome_prob"].cpu().numpy())

    obs_probs = np.concatenate(obs_probs).squeeze()
    cf_probs = np.concatenate(cf_probs).squeeze()

    v_observed = obs_probs.mean()
    v_rule = cf_probs.mean()
    ate = v_rule - v_observed

    return {
        "estimator": "DM",
        "v_observed": float(v_observed),
        "v_rule": float(v_rule),
        "ate": float(ate),
        "obs_probs": obs_probs,
        "cf_probs": cf_probs,
    }


# ── Doubly-Robust (DR) ────────────────────────────────────────────────


def evaluate_dr(
    model,
    dataset,
    device: torch.device,
    batch_size: int = 256,
    epsilon: float = 0.05,
    trim_quantiles: tuple[float, float] = (5.0, 95.0),
) -> dict:
    """
    Doubly-Robust estimator combining outcome model with importance weighting.

    DR = (1/n) Σ [ μ̂(π_rule) + w_i * (Y_i - μ̂(π_obs)) ]

    where w_i is the cumulative importance ratio for the rule policy.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_y = []
    all_mu_obs = []
    all_mu_rule = []
    all_log_ratios = []

    n_actions = dataset.n_actions

    for batch in tqdm(loader, desc="    DR eval", leave=False, mininterval=2):
        temporal = batch["temporal"].to(device)
        static = batch["static"].to(device)
        action = batch["action"].to(device)
        rule_action = batch["rule_action"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        absorbing_mask = batch["absorbing_mask"].to(device)
        propensity = batch["propensity"].numpy()  # (B, T)
        mortality = batch["mortality_28d"].numpy()

        # Outcome model predictions
        out_obs = model.predict_counterfactual(
            temporal, static, action, valid_mask, absorbing_mask
        )
        out_rule = model.predict_counterfactual(
            temporal, static, rule_action, valid_mask, absorbing_mask
        )

        all_y.append(mortality)
        all_mu_obs.append(out_obs["outcome_prob"].cpu().numpy().squeeze())
        all_mu_rule.append(out_rule["outcome_prob"].cpu().numpy().squeeze())

        # Per-step importance ratios: π_rule(a|s) / μ(a|s)
        action_np = batch["action"].numpy()
        rule_np = batch["rule_action"].numpy()
        vmask = batch["valid_mask"].numpy()
        amask = batch["absorbing_mask"].numpy()

        B, T = action_np.shape
        log_ratios = np.zeros(B)

        for b_idx in range(B):
            lr = 0.0
            for t in range(T):
                if vmask[b_idx, t] < 0.5 or amask[b_idx, t] < 0.5:
                    continue
                # Softened deterministic rule policy
                if action_np[b_idx, t] == rule_np[b_idx, t]:
                    pi_rule = 1.0 - epsilon + epsilon / n_actions
                else:
                    pi_rule = epsilon / n_actions

                mu = max(propensity[b_idx, t], 1e-6)
                ratio = pi_rule / mu
                ratio = np.clip(ratio, 0.01, 200.0)
                lr += np.log(ratio)
            log_ratios[b_idx] = lr

        all_log_ratios.append(log_ratios)

    y = np.concatenate(all_y)
    mu_obs = np.concatenate(all_mu_obs)
    mu_rule = np.concatenate(all_mu_rule)
    log_w = np.concatenate(all_log_ratios)

    # Exponentiate and trim
    w = np.exp(np.clip(log_w, -50, 50))
    lo, hi = np.percentile(w, trim_quantiles)
    w_trimmed = np.clip(w, lo, hi)

    # DR correction term uses unnormalized weights with sample mean scaling.
    v_rule_dr = mu_rule.mean() + np.mean(w_trimmed * (y - mu_obs))

    # Observed mortality
    v_observed = y.mean()

    ate_dr = v_rule_dr - v_observed
    ess = (w_trimmed.sum() ** 2) / (w_trimmed ** 2).sum()

    return {
        "estimator": "DR",
        "v_observed": float(v_observed),
        "v_rule": float(v_rule_dr),
        "ate": float(ate_dr),
        "ess": float(ess),
        "weights_mean": float(w_trimmed.mean()),
        "weights_median": float(np.median(w_trimmed)),
        "weights_max": float(w_trimmed.max()),
        "weights_p99": float(np.percentile(w_trimmed, 99)),
        "mu_obs_mean": float(mu_obs.mean()),
        "mu_rule_mean": float(mu_rule.mean()),
    }


# ── Fitted Q-Evaluation (FQE) ─────────────────────────────────────────


class FQENetwork(nn.Module):
    """Q-network for Fitted Q-Evaluation."""

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dims: list[int] = (256, 128),
        activation: str = "relu",
    ):
        super().__init__()
        act_fn = nn.ReLU if activation == "relu" else nn.GELU

        layers = []
        in_dim = state_dim + n_actions  # state + one-hot action
        for hd in hidden_dims:
            layers.extend([nn.Linear(in_dim, hd), act_fn(), nn.Dropout(0.1)])
            in_dim = hd
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self.n_actions = n_actions

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        state: (B, state_dim)
        action: (B,) integer
        Returns: Q(s, a) → (B, 1)
        """
        action_oh = F.one_hot(action.long(), self.n_actions).float()
        x = torch.cat([state, action_oh], dim=-1)
        return self.net(x)

    def q_all_actions(self, state: torch.Tensor) -> torch.Tensor:
        """Compute Q(s, a) for all actions. Returns (B, n_actions)."""
        B = state.size(0)
        qs = []
        for a in range(self.n_actions):
            action = torch.full((B,), a, dtype=torch.long, device=state.device)
            qs.append(self.forward(state, action))
        return torch.cat(qs, dim=-1)  # (B, n_actions)


def train_fqe(
    dataset,
    model,
    device: torch.device,
    fqe_cfg: dict,
    gamma: float = 1.0,
    epsilon: float = 0.05,
) -> FQENetwork:
    """
    Train FQE Q-network to evaluate rule policy.

    Uses fitted Q-iteration:
      Q^{k+1}(s_t, a_t) ← r_t + γ * Σ_a π(a|s_{t+1}) * Q^k(s_{t+1}, a)
    """
    n_actions = dataset.n_actions
    d_temporal = dataset.d_temporal

    # State representation: temporal features at each time step
    state_dim = d_temporal

    q_net = FQENetwork(
        state_dim=state_dim,
        n_actions=n_actions,
        hidden_dims=fqe_cfg.get("hidden_dims", [256, 128]),
    ).to(device)

    optimizer = torch.optim.Adam(
        q_net.parameters(),
        lr=fqe_cfg.get("learning_rate", 0.001),
    )

    # Build transition tuples: (s_t, a_t, r_t, s_{t+1}, done)
    transitions = []
    for i in tqdm(range(len(dataset)), desc="    FQE transitions", leave=False, mininterval=2):
        item = dataset[i]
        for t in range(dataset.n_bins):
            if item["valid_mask"][t] < 0.5:
                continue
            if item["absorbing_mask"][t] < 0.5:
                continue

            s_t = item["temporal"][t].numpy()
            a_t = int(item["action"][t])
            r_t = float(item["reward"][t])
            rule_a_t = int(item["rule_action"][t])

            # Check if next step is valid and non-absorbing
            if (
                t + 1 < dataset.n_bins
                and item["valid_mask"][t + 1] > 0.5
                and item["absorbing_mask"][t + 1] > 0.5
            ):
                s_next = item["temporal"][t + 1].numpy()
                done = 0.0
            else:
                s_next = s_t  # terminal; Q-value is just reward
                done = 1.0

            transitions.append((s_t, a_t, r_t, s_next, done, rule_a_t))

    if not transitions:
        return q_net

    states = torch.tensor(np.array([t[0] for t in transitions]), dtype=torch.float32)
    actions = torch.tensor([t[1] for t in transitions], dtype=torch.long)
    rewards = torch.tensor([t[2] for t in transitions], dtype=torch.float32)
    next_states = torch.tensor(np.array([t[3] for t in transitions]), dtype=torch.float32)
    dones = torch.tensor([t[4] for t in transitions], dtype=torch.float32)
    rule_actions = torch.tensor([t[5] for t in transitions], dtype=torch.long)

    n_trans = len(transitions)
    batch_size = min(fqe_cfg.get("batch_size", 256), n_trans)
    n_iterations = fqe_cfg.get("n_iterations", 50)
    n_epochs = fqe_cfg.get("training_epochs", 100)

    # FQE iterations
    pbar_fqe = tqdm(range(n_iterations), desc="    FQE iters", leave=False, mininterval=2)
    for fqe_iter in pbar_fqe:
        # Compute target Q-values with current Q
        with torch.no_grad():
            q_next_all = q_net.q_all_actions(next_states.to(device))  # (N, n_actions)

            # Expected Q under rule policy (softened deterministic)
            rule_oh = F.one_hot(rule_actions, n_actions).float()
            pi_probs = (1 - epsilon) * rule_oh + epsilon / n_actions
            v_next = (q_next_all.cpu() * pi_probs).sum(dim=-1)  # (N,)

            targets = rewards + gamma * (1 - dones) * v_next  # (N,)

        # Train Q-network to match targets
        indices = torch.randperm(n_trans)
        final_epoch_loss = float("inf")
        converged_epoch = n_epochs
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n_trans, batch_size):
                idx = indices[start: start + batch_size]
                s_b = states[idx].to(device)
                a_b = actions[idx].to(device)
                t_b = targets[idx].to(device)

                q_pred = q_net(s_b, a_b).squeeze(-1)
                loss = F.mse_loss(q_pred, t_b)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            final_epoch_loss = epoch_loss / max(n_batches, 1)
            if final_epoch_loss < fqe_cfg.get("convergence_tol", 1e-4):
                converged_epoch = epoch + 1
                break

        pbar_fqe.set_postfix(
            ep=f"{converged_epoch}/{n_epochs}",
            loss=f"{final_epoch_loss:.2e}",
        )

    return q_net


def evaluate_fqe(
    q_net: FQENetwork,
    dataset,
    device: torch.device,
    epsilon: float = 0.05,
) -> dict:
    """
    Evaluate rule policy using trained FQE Q-network.

    V(π) = E_s0[ Σ_a π(a|s0) * Q(s0, a) ]
    """
    n_actions = dataset.n_actions
    initial_states = []
    actual_mortality = []

    for i in tqdm(range(len(dataset)), desc="    FQE eval", leave=False, mininterval=2):
        item = dataset[i]
        # First valid bin
        for t in range(dataset.n_bins):
            if item["valid_mask"][t] > 0.5:
                initial_states.append(item["temporal"][t].numpy())
                break
        actual_mortality.append(float(item["mortality_28d"]))

    s0 = torch.tensor(np.array(initial_states), dtype=torch.float32).to(device)
    rule_actions_t0 = []
    for i in range(len(dataset)):
        item = dataset[i]
        for t in range(dataset.n_bins):
            if item["valid_mask"][t] > 0.5:
                rule_actions_t0.append(int(item["rule_action"][t]))
                break

    rule_a = torch.tensor(rule_actions_t0, dtype=torch.long).to(device)

    with torch.no_grad():
        q_all = q_net.q_all_actions(s0)  # (N, n_actions)

        # Rule policy probabilities
        rule_oh = F.one_hot(rule_a, n_actions).float()
        pi_probs = (1 - epsilon) * rule_oh + epsilon / n_actions

        # V(π) = Σ_a π(a|s0) Q(s0, a)
        v_rule = (q_all * pi_probs).sum(dim=-1)  # (N,)

    # Transform: Q-values are in reward space (0 = survival, -1 = death)
    # Policy value in mortality probability: V_mort = -V_reward
    v_rule_mort = -v_rule.cpu().numpy().mean()
    v_observed = np.mean(actual_mortality)
    ate = v_rule_mort - v_observed

    return {
        "estimator": "FQE",
        "v_observed": float(v_observed),
        "v_rule": float(v_rule_mort),
        "ate": float(ate),
    }


# ── Bootstrap wrapper ─────────────────────────────────────────────────


def bootstrap_ope(
    eval_fn,
    dataset,
    n_resamples: int = 500,
    seed: int = 7777,
    confidence: float = 0.95,
    **eval_kwargs,
) -> dict:
    """
    Bootstrap confidence intervals for any OPE estimator.

    eval_fn should accept (dataset, **eval_kwargs) and return dict with 'ate' and 'v_rule'.
    We resample stays (not observations) to preserve within-stay correlation.
    """
    rng = np.random.RandomState(seed)
    n = len(dataset)

    boot_ates = []
    boot_v_rules = []

    pbar = tqdm(range(n_resamples), desc="    bootstrap", leave=False, mininterval=2)
    for b in pbar:
        # Resample stay indices with replacement
        idx = rng.choice(n, size=n, replace=True)

        # Create bootstrapped dataset view
        boot_ds = _BootstrapSubset(dataset, idx)

        try:
            result = eval_fn(boot_ds, **eval_kwargs)
            boot_ates.append(result["ate"])
            boot_v_rules.append(result["v_rule"])
        except Exception:
            continue

        if boot_ates:
            pbar.set_postfix(ate=f"{np.mean(boot_ates):+.4f}", ok=len(boot_ates))

    if not boot_ates:
        return {"n_success": 0}

    alpha = (1 - confidence) / 2
    boot_ates = np.array(boot_ates)
    boot_v_rules = np.array(boot_v_rules)

    return {
        "n_success": len(boot_ates),
        "ate_mean": float(boot_ates.mean()),
        "ate_ci_lo": float(np.percentile(boot_ates, 100 * alpha)),
        "ate_ci_hi": float(np.percentile(boot_ates, 100 * (1 - alpha))),
        "v_rule_mean": float(boot_v_rules.mean()),
        "v_rule_ci_lo": float(np.percentile(boot_v_rules, 100 * alpha)),
        "v_rule_ci_hi": float(np.percentile(boot_v_rules, 100 * (1 - alpha))),
    }


def bootstrap_fqe_eval_only(
    q_net,
    dataset,
    device: torch.device,
    n_resamples: int = 500,
    seed: int = 7777,
    confidence: float = 0.95,
    epsilon: float = 0.05,
) -> dict:
    """
    Bootstrap CIs for FQE by resampling evaluation only (no retraining).

    The Q-network is already trained on the full dataset. For each bootstrap
    sample we resample stays and re-evaluate V(π) using the frozen Q-network.
    This is ~1000x faster than retraining FQE per bootstrap sample.
    """
    rng = np.random.RandomState(seed)
    n = len(dataset)

    boot_ates = []
    boot_v_rules = []

    pbar = tqdm(range(n_resamples), desc="    bootstrap(FQE-eval)", leave=False, mininterval=2)
    for b in pbar:
        idx = rng.choice(n, size=n, replace=True)
        boot_ds = _BootstrapSubset(dataset, idx)

        try:
            result = evaluate_fqe(q_net, boot_ds, device, epsilon=epsilon)
            boot_ates.append(result["ate"])
            boot_v_rules.append(result["v_rule"])
        except Exception:
            continue

        if boot_ates:
            pbar.set_postfix(ate=f"{np.mean(boot_ates):+.4f}", ok=len(boot_ates))

    if not boot_ates:
        return {"n_success": 0}

    alpha = (1 - confidence) / 2
    boot_ates = np.array(boot_ates)
    boot_v_rules = np.array(boot_v_rules)

    return {
        "n_success": len(boot_ates),
        "ate_mean": float(boot_ates.mean()),
        "ate_ci_lo": float(np.percentile(boot_ates, 100 * alpha)),
        "ate_ci_hi": float(np.percentile(boot_ates, 100 * (1 - alpha))),
        "v_rule_mean": float(boot_v_rules.mean()),
        "v_rule_ci_lo": float(np.percentile(boot_v_rules, 100 * alpha)),
        "v_rule_ci_hi": float(np.percentile(boot_v_rules, 100 * (1 - alpha))),
    }


class _BootstrapSubset:
    """Lightweight wrapper for bootstrapped dataset (resampled indices)."""

    def __init__(self, dataset, indices: np.ndarray):
        self._dataset = dataset
        self._indices = indices
        # Forward attributes needed by OPE functions
        self.n_actions = dataset.n_actions
        self.n_bins = dataset.n_bins
        self.d_temporal = dataset.d_temporal

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        return self._dataset[self._indices[idx]]
