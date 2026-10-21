"""
DTR Dataset for Step C deep causal model training.

Extends ICUStayDataset with action indices, propensities, rule-policy actions,
and sparse terminal rewards needed for counterfactual training and OPE.

Protocol reference: Step 80 (v3.1 § Step C deep causal models)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset

from .dataset import discover_feature_columns, make_train_val_test_split


_KEY_COLUMNS = ["stay_id", "bin"]
_RULE_COLUMNS = ("rule_action_merged", "rule_action", "rule_policy_action")


def _validate_join_keys(frame: pd.DataFrame, label: str) -> None:
    """Reject ambiguous or missing keys before aligning policy actions."""
    missing = [col for col in _KEY_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing join columns: {missing}")
    if frame[_KEY_COLUMNS].isna().any().any():
        raise ValueError(f"{label} has null stay_id/bin join keys")
    if frame.duplicated(_KEY_COLUMNS).any():
        raise ValueError(f"{label} has duplicate stay_id/bin join keys")


def _validate_action_values(values: pd.Series, label: str) -> None:
    """Allow missing masked bins, but never truncate malformed action codes."""
    try:
        numeric = pd.to_numeric(values, errors="raise").to_numpy(
            dtype=np.float64, na_value=np.nan
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain integer action codes") from exc
    present = numeric[~np.isnan(numeric)]
    if (not np.isfinite(present).all()
            or (present != np.floor(present)).any()
            or (present < 0).any()
            or (present >= 2**63).any()):
        raise ValueError(f"{label} contains invalid integer action codes")


def _attach_rule_actions(
    root: Path, config: dict, propensity_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach raw/merged 70c actions without changing the behavior-policy fit."""
    _validate_join_keys(propensity_df, "propensity table")
    data_cfg = config["data"]
    explicit_path = data_cfg.get("rule_policy_actions")
    embedded = next((c for c in _RULE_COLUMNS if c in propensity_df.columns), None)
    if explicit_path is None and embedded is not None:
        _validate_action_values(propensity_df[embedded], embedded)
        return propensity_df

    if explicit_path is None:
        scheme = config.get("action", {}).get("scheme", "coarse_3")
        propensity_name = Path(data_cfg["propensities_coarse3"]).name
        if scheme != "coarse_3" or propensity_name != "70b_propensity_coarse3_v1_mimic.parquet":
            raise ValueError(
                "No rule-action column is present. Set data.rule_policy_actions "
                "to a cohort- and action-scheme-compatible rule-action parquet; "
                "the automatic 70c default applies only to coarse_3 MIMIC."
            )
        rule_path = root / "data/interim/70c_rule_policy_actions_v1_mimic.parquet"
    else:
        rule_path = root / explicit_path

    if not rule_path.is_file():
        raise FileNotFoundError(
            f"Rule-action parquet not found: {rule_path}. Generate it later with "
            "the corrected 70c script, or set data.rule_policy_actions to an "
            "existing compatible parquet. Missing rules cannot default to action 0."
        )
    rules = pd.read_parquet(rule_path)
    _validate_join_keys(rules, "rule-action table")
    rule_col = next((c for c in _RULE_COLUMNS if c in rules.columns), None)
    if rule_col is None:
        raise ValueError(f"Rule-action table must contain one of {_RULE_COLUMNS}")
    _validate_action_values(rules[rule_col], f"rule-action table {rule_col}")
    target_col = "rule_action_merged" if rule_col == "rule_action_merged" else "rule_action"
    selected = rules[_KEY_COLUMNS + [rule_col]].rename(columns={rule_col: target_col})
    # An explicit sidecar is authoritative; remove stale embedded rule columns.
    base = propensity_df.drop(columns=list(_RULE_COLUMNS), errors="ignore")
    return base.merge(selected, on=_KEY_COLUMNS, how="left", validate="one_to_one", sort=False)


class DTRDataset(Dataset):
    """
    Per-stay dataset for DTR training: state features + actions + rewards.

    Each item contains:
      - temporal: (T, D_temporal) z-scored temporal features
      - static: (D_static,) static covariates
      - valid_mask: (T,) 1 where bin has data
      - absorbing_mask: (T,) 0 for post-absorbing bins
      - action: (T,) integer action index (merged)
      - action_raw: (T,) integer action index (pre-merge, 0-17)
      - propensity: (T,) P(a_t | s_t) from behavior policy
      - rule_action: (T,) integer action index under rule policy
      - reward: (T,) sparse terminal reward
      - mortality_28d: scalar
    """

    def __init__(
        self,
        state_df: pd.DataFrame,
        outcomes_df: pd.DataFrame,
        propensity_df: pd.DataFrame,
        stay_ids: np.ndarray,
        temporal_cols: list[str],
        static_cols: list[str],
        merge_map: dict[int, int] | None = None,
        n_bins: int = 12,
        absorbing_col: str = "absorbing_state_bin",
        action_vocab: list[int] | None = None,
    ):
        self.n_bins = n_bins
        self.temporal_cols = temporal_cols
        self.static_cols = static_cols
        self.d_temporal = len(temporal_cols)
        self.d_static = len(static_cols)

        # Filter to requested stays
        stay_set = set(stay_ids)
        self.stay_ids = np.array(sorted(stay_set))
        self.n_stays = len(self.stay_ids)
        sid_to_idx = {sid: i for i, sid in enumerate(self.stay_ids)}

        # ── Pre-allocate arrays ──
        self.temporal = np.zeros((self.n_stays, n_bins, self.d_temporal), dtype=np.float32)
        self.static = np.zeros((self.n_stays, self.d_static), dtype=np.float32)
        self.valid_mask = np.zeros((self.n_stays, n_bins), dtype=np.float32)
        self.absorbing_mask = np.ones((self.n_stays, n_bins), dtype=np.float32)
        self.action = np.zeros((self.n_stays, n_bins), dtype=np.int64)
        self.action_raw = np.zeros((self.n_stays, n_bins), dtype=np.int64)
        self.propensity = np.zeros((self.n_stays, n_bins), dtype=np.float32)
        self.rule_action = np.zeros((self.n_stays, n_bins), dtype=np.int64)
        self.reward = np.zeros((self.n_stays, n_bins), dtype=np.float32)
        self.mortality_28d = np.zeros(self.n_stays, dtype=np.float32)

        # ── Fill temporal features ──
        df = state_df.loc[state_df["stay_id"].isin(stay_set)].sort_values(["stay_id", "bin"])
        _validate_join_keys(df, "state table for requested stays")
        temporal_vals = df[temporal_cols].values.astype(np.float32)
        bin_vals = df["bin"].values
        sid_vals = df["stay_id"].values

        for row_i in range(len(df)):
            sid = sid_vals[row_i]
            if sid not in sid_to_idx:
                continue
            stay_i = sid_to_idx[sid]
            b = int(bin_vals[row_i])
            if 0 <= b < n_bins:
                self.temporal[stay_i, b, :] = temporal_vals[row_i]
                self.valid_mask[stay_i, b] = 1.0

        # ── Fill static features (first bin per stay) ──
        if static_cols:
            first_rows = df.drop_duplicates(subset="stay_id", keep="first")
            static_vals = first_rows.set_index("stay_id")[static_cols]
            for sid in self.stay_ids:
                if sid in static_vals.index:
                    self.static[sid_to_idx[sid]] = static_vals.loc[sid].values.astype(np.float32)

        # ── Outcomes + absorbing mask ──
        out_indexed = outcomes_df.set_index("stay_id")
        for sid in self.stay_ids:
            if sid not in out_indexed.index:
                continue
            idx = sid_to_idx[sid]
            row = out_indexed.loc[sid]
            self.mortality_28d[idx] = float(row.get("mortality_28d", 0))
            ab_bin = row.get(absorbing_col, np.nan)
            if pd.notna(ab_bin):
                ab_int = int(ab_bin)
                if ab_int + 1 < n_bins:
                    self.absorbing_mask[idx, ab_int + 1:] = 0.0

        # ── Fill actions + propensities from 70b table ──
        prop_df = propensity_df.loc[propensity_df["stay_id"].isin(stay_set)].copy()
        _validate_join_keys(prop_df, "propensity table for requested stays")

        # Determine action column names
        act_col = "action_idx_merged" if "action_idx_merged" in prop_df.columns else "action_idx"
        act_raw_col = "action_idx" if "action_idx_merged" in prop_df.columns else "action_idx"
        prop_col = "propensity_clipped" if "propensity_clipped" in prop_df.columns else "propensity"
        rule_col = next((c for c in _RULE_COLUMNS if c in prop_df.columns), None)
        if rule_col is None:
            raise ValueError(
                "No rule-action column is present. Use load_dtr_data() with a "
                "compatible 70c sidecar; missing rules cannot default to action 0."
            )
        _validate_action_values(prop_df[act_col], act_col)
        _validate_action_values(prop_df[rule_col], rule_col)
        action_present = np.zeros_like(self.valid_mask, dtype=bool)
        rule_present = np.zeros_like(self.valid_mask, dtype=bool)

        for _, row in prop_df.iterrows():
            sid = row["stay_id"]
            if sid not in sid_to_idx:
                continue
            idx = sid_to_idx[sid]
            b = int(row["bin"])
            if 0 <= b < n_bins:
                if pd.notna(row[act_col]):
                    self.action[idx, b] = int(row[act_col])
                    action_present[idx, b] = True
                if act_raw_col in row.index and pd.notna(row[act_raw_col]):
                    self.action_raw[idx, b] = int(row[act_raw_col])
                self.propensity[idx, b] = float(row[prop_col])
                if pd.notna(row[rule_col]):
                    self.rule_action[idx, b] = int(row[rule_col])
                    rule_present[idx, b] = True

        active = (self.valid_mask > 0) & (self.absorbing_mask > 0)
        if np.any(active & ~action_present) or np.any(active & ~rule_present):
            raise ValueError(
                "Observed/rule actions do not cover every active state bin: "
                f"missing observed={int(np.sum(active & ~action_present))}, "
                f"missing rule={int(np.sum(active & ~rule_present))}. "
                "Check the stay_id/bin join and regenerate the corrected 70c "
                "sidecar later if needed. No zero-action fallback is allowed."
            )

        # Raw observed and rule actions need independent, non-cascading merges.
        if merge_map and act_col == "action_idx":
            original_actions = self.action.copy()
            for old, new in merge_map.items():
                mask = original_actions == old
                self.action[mask] = new
        if merge_map and rule_col != "rule_action_merged":
            original_rules = self.rule_action.copy()
            for old, new in merge_map.items():
                mask_r = original_rules == old
                self.rule_action[mask_r] = new

        # ── Sparse terminal reward ──
        # R = -mortality_28d at the last valid bin, 0 elsewhere
        for i in range(self.n_stays):
            last_valid = -1
            for b in range(n_bins):
                if self.valid_mask[i, b] > 0 and self.absorbing_mask[i, b] > 0:
                    last_valid = b
            if last_valid >= 0:
                self.reward[i, last_valid] = -self.mortality_28d[i]

        # ── NaN cleanup ──
        np.nan_to_num(self.temporal, copy=False, nan=0.0)
        np.nan_to_num(self.static, copy=False, nan=0.0)
        np.nan_to_num(self.propensity, copy=False, nan=0.0)

        # ── Build action-to-index mapping for contiguous 0..N_actions-1 ──
        if action_vocab is not None:
            # Use externally provided vocab (e.g. frozen MIMIC training vocab)
            self.action_vocab = sorted(action_vocab)
        else:
            # Auto-discover from data (training mode)
            all_actions = np.unique(self.action[self.valid_mask.astype(bool)])
            self.action_vocab = sorted(all_actions.tolist())
        self.n_actions = len(self.action_vocab)
        self._action_to_contiguous = {a: i for i, a in enumerate(self.action_vocab)}
        for label, values in (("observed", self.action), ("rule", self.rule_action)):
            unsupported = sorted(set(np.unique(values[active])) - set(self.action_vocab))
            if unsupported:
                raise ValueError(
                    f"Active {label} action codes {unsupported} are absent from "
                    "the shared action vocabulary. Check the merge map and "
                    "action scheme; do not silently map them to action 0."
                )

    def remap_action_to_contiguous(self, action_idx: int) -> int:
        """Map merged action index to contiguous 0..n_actions-1."""
        try:
            return self._action_to_contiguous[action_idx]
        except KeyError as exc:
            raise ValueError(f"Action {action_idx} is absent from the shared vocabulary") from exc

    def __len__(self) -> int:
        return self.n_stays

    def __getitem__(self, idx: int) -> dict:
        # Only masked padding/post-absorbing bins receive a placeholder zero.
        active = (self.valid_mask[idx] > 0) & (self.absorbing_mask[idx] > 0)
        action_contig = np.zeros(self.n_bins, dtype=np.int64)
        rule_contig = np.zeros(self.n_bins, dtype=np.int64)
        action_contig[active] = [
            self.remap_action_to_contiguous(a) for a in self.action[idx, active]
        ]
        rule_contig[active] = [
            self.remap_action_to_contiguous(a) for a in self.rule_action[idx, active]
        ]

        return {
            "stay_id": self.stay_ids[idx],
            "temporal": torch.from_numpy(self.temporal[idx]),          # (T, D_t)
            "static": torch.from_numpy(self.static[idx]),              # (D_s,)
            "valid_mask": torch.from_numpy(self.valid_mask[idx]),       # (T,)
            "absorbing_mask": torch.from_numpy(self.absorbing_mask[idx]),  # (T,)
            "action": torch.from_numpy(action_contig),                  # (T,) int
            "action_raw": torch.from_numpy(self.action_raw[idx]),       # (T,) int
            "propensity": torch.from_numpy(self.propensity[idx]),       # (T,)
            "rule_action": torch.from_numpy(rule_contig),               # (T,) int
            "reward": torch.from_numpy(self.reward[idx]),               # (T,)
            "mortality_28d": torch.tensor(self.mortality_28d[idx]),
        }


def load_dtr_data(
    project_root: str | Path,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str], dict[int, int] | None]:
    """
    Load all data artifacts needed for Step C training.

    Returns: (state_df, outcomes_df, propensity_df, temporal_cols, static_cols, merge_map)
    """
    root = Path(project_root)

    state_df = pd.read_parquet(root / config["data"]["state_tensors"])
    outcomes_df = pd.read_parquet(root / config["data"]["outcomes"])
    propensity_df = pd.read_parquet(root / config["data"]["propensities_coarse3"])
    propensity_df = _attach_rule_actions(root, config, propensity_df)

    temporal_cols, static_cols = discover_feature_columns(state_df)

    # Load merge map
    merge_map = None
    merge_path = root / config["data"]["merge_map"]
    if merge_path.exists():
        with open(merge_path) as f:
            mm = yaml.safe_load(f)
        if "merge_map" in mm:
            merge_map = {int(k): int(v) for k, v in mm["merge_map"].items()}

    return state_df, outcomes_df, propensity_df, temporal_cols, static_cols, merge_map


def build_dtr_datasets(
    project_root: str | Path,
    config: dict,
) -> tuple[DTRDataset, DTRDataset, DTRDataset]:
    """
    Build train/val/test DTRDatasets from config.

    Returns: (train_ds, val_ds, test_ds)
    """
    state_df, outcomes_df, propensity_df, temporal_cols, static_cols, merge_map = (
        load_dtr_data(project_root, config)
    )

    all_stay_ids = state_df["stay_id"].unique()
    split_cfg = config["split"]
    train_ids, val_ids, test_ids = make_train_val_test_split(
        all_stay_ids,
        outcomes_df,
        train_frac=split_cfg["train_frac"],
        val_frac=split_cfg["val_frac"],
        test_frac=split_cfg["test_frac"],
        stratify_col=split_cfg["stratify_col"],
        seed=split_cfg["seed"],
    )

    common_kwargs = dict(
        state_df=state_df,
        outcomes_df=outcomes_df,
        propensity_df=propensity_df,
        temporal_cols=temporal_cols,
        static_cols=static_cols,
        merge_map=merge_map,
        n_bins=config["grid"]["n_bins"],
    )

    train_ds = DTRDataset(stay_ids=train_ids, **common_kwargs)

    # Enforce a shared action vocabulary across splits to keep embeddings/OPE consistent.
    shared_vocab = train_ds.action_vocab
    val_ds = DTRDataset(stay_ids=val_ids, action_vocab=shared_vocab, **common_kwargs)
    test_ds = DTRDataset(stay_ids=test_ids, action_vocab=shared_vocab, **common_kwargs)

    print(f"  DTR splits: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    print(f"  Temporal features: {train_ds.d_temporal}, Static: {train_ds.d_static}")
    print(f"  Actions: {train_ds.n_actions} (contiguous from {len(config['action']['valid_merged_indices'])} merged)")

    return train_ds, val_ds, test_ds
