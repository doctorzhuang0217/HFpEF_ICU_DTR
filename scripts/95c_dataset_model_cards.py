#!/usr/bin/env python3
"""95c — Dataset and Model Cards

Protocol: v3.1, Step 95c
Compute: C1

Purpose
-------
Create dataset and model cards for reproducibility and handoff,
following standard ML documentation practices.

Outputs
-------
- reports/95c_dataset_card.md
- reports/95c_model_card.md
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────
ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
TABLES_DIR = ROOT / "tables"
REPORTS_DIR = ROOT / "reports"
CONFIGS_DIR = ROOT / "configs"

for d in [REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("95c")


def _load_registry():
    path = CONFIGS_DIR / "90a_primary_model_registry.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def generate_dataset_card(registry: dict, ts: str) -> str:
    """Generate dataset card markdown."""
    grid = registry.get("training_config", {}).get("grid", {})
    norm = registry.get("normalization", {})
    action = registry.get("training_config", {}).get("action", {})
    active_vars = norm.get("active_vars", [])

    card = f"""# Dataset Card: HFpEF ICU DTR Study

## Dataset Description

- **Project:** HFpEF ICU Dynamic Treatment Regime (DTR) Study
- **Protocol version:** v3.1
- **Generated:** {ts}
- **Primary database:** MIMIC-IV (PhysioNet credentialed access)
- **External validation:** eICU-CRD (PhysioNet credentialed access)
- **Abandoned:** HiRID 1.1.1 (NT-proBNP coverage 1.1%, infeasible)

## Dataset Summary

This study evaluates dynamic fluid/vasopressor/diuretic strategies for HFpEF
patients in the ICU using causal inference on observational data.

### Population

| Parameter | MIMIC-IV | eICU-CRD |
|-----------|----------|----------|
| Cohort | ICU stays with heart failure assessment | Multicenter ICU stays |
| HFpEF ascertainment | HF diagnosis + EF >= 50% (strict) | ICD-based (broader) |
| Time zero | ICU admission + {grid.get('t0_hours', 6)}h | Same |
| Horizon | {grid.get('horizon_hours', 24)}h | Same |
| Bin size | {grid.get('bin_hours', 2)}h | Same |
| N bins | {grid.get('n_bins', 12)} | Same |

### Features

- **State variables:** {len(active_vars)} active features
- **Variables:** {', '.join(active_vars[:10])}{'...' if len(active_vars) > 10 else ''}
- **Normalization:** {norm.get('method', 'z-score')}
- **Imputation:** {norm.get('imputation', 'LOCF + population_median')}

### Action Space

- **Scheme:** {action.get('scheme', 'coarse_3')}
- **Raw cells:** {action.get('n_raw_cells', 18)}
- **Merged cells:** {action.get('n_merged_cells', 13)}
- **Encoding:** {action.get('encoding', 'fluid_level * 6 + vaso_level * 2 + diuretic_level')}

### Outcome

- **Primary outcome:** 28-day mortality (binary)
- **Reward:** R = -1 * mortality_28d (sparse terminal)
- **Discount factor:** gamma = 1.0

## Ethical Considerations

- All data accessed under PhysioNet credentialed data use agreement
- No individual patient data is stored in this repository
- De-identified data only; no re-identification attempted
- Study approved under the PhysioNet credentialed access framework

## Limitations

- Observational data; causal claims require caution
- MIMIC-IV: single-center (Beth Israel Deaconess Medical Center)
- eICU-CRD: broader ascertainment (~35% HFrEF contamination in HFpEF group)
- HiRID: abandoned due to insufficient NT-proBNP coverage
- 28-day mortality in eICU approximated by hospital discharge status

## Citation

Protocol: HFpEF_ICU_DTR_protocol_v3_1_handoff_packet.md
"""
    return card


def generate_model_card(registry: dict, ts: str) -> str:
    """Generate model card markdown."""
    models = registry.get("models", {})

    card = f"""# Model Card: HFpEF ICU DTR Study Models

## Model Details

- **Study:** HFpEF ICU Dynamic Treatment Regime
- **Protocol version:** v3.1
- **Generated:** {ts}
- **Task:** Off-policy evaluation of hemodynamic treatment strategies
- **Primary pair:** 75a MSM-IPTW + 80c G-Transformer

## Model Suite

### Primary Models

| Model | Architecture | Role | Primary Estimator |
|-------|-------------|------|-------------------|
| 75a MSM-IPTW | Marginal structural model | Primary classical | WIS |
| 80c G-Transformer | Transformer (d=128, 4 heads, 4 layers) | Primary deep | DM (rollout) |

### Confirmatory Models

| Model | Architecture | Role | Primary Estimator |
|-------|-------------|------|-------------------|
| 80a GNet | GRU-based (hidden=128, 2 layers) | Confirmatory | DM, FQE |
| 80b ACTIN | Balanced representation | Confirmatory | DR, FQE |

### Ablation / Optional

| Model | Architecture | Role | Primary Estimator |
|-------|-------------|------|-------------------|
| 80d CT-Legacy | Causal transformer (baseline) | Legacy ablation | FQE |
| 80e Mamba-CDSP | Mamba SSM | Optional | FQE |

## Key Results

| Model | Estimator | ATE (pp) |
|-------|-----------|----------|
| 75a MSM-IPTW | WIS | -4.87 [-5.68, -4.05] |
| 80c G-Transformer | DM (rollout) | -1.37 [-1.49, -1.26] |
| 80a GNet | DM | -0.22 |
| 80b ACTIN | DR | -0.79 |
| 80d CT-Legacy | FQE | -2.43 |
| 80e Mamba-CDSP | FQE | -2.16 |

## Evaluation

- **OPE estimators:** Direct Method (DM), Doubly Robust (DR), Fitted Q-Evaluation (FQE), Weighted IS (WIS)
- **Bootstrap:** B=500, percentile method
- **Subgroup analysis:** HF subtype (hfpef, no_hf, hfref, hfmref, hf_ef_unknown) x shock stratum
- **External validation:** eICU-CRD (DM Grade C, sign consistent)

## Sensitivity Analyses

| Analysis | Verdict |
|----------|---------|
| 85a EDQ | ROBUST |
| 85b Sequential deconfounding | SENSITIVE |
| 85d Positivity grid | STABLE |
| 85e Negative control | PASS 3/3 |
| 85f E-value + GOC | WEAK E-value, GOC ROBUST |
| 85g Monitoring intensity | ROBUST |
| 85h RRT sensitivity | ROBUST |
| 85i Horizon variation | SENSITIVE |
| 85j Informative missingness | REFRAMED (methodological strength) |

## Limitations

- Observational study; unmeasured confounding cannot be fully excluded
- E-value = 1.52 (moderate robustness to unmeasured confounding)
- 48h horizon shows sign reversal; phenotype x horizon interaction present
- External validation limited to sign consistency (Grade C); no CI overlap
- State transition model non-transportable (rollout drift 2.55x-35x)
- HiRID validation infeasible

## Intended Use

- Research use only
- Not intended for clinical decision-making without prospective validation
- Hypothesis-generating for future RCT design

## Configuration Files

- Training config: `configs/80_step_c_training_v1.yaml`
- Normalization: `configs/45c_normalization_params_v1.yaml`
- Action merge map: `configs/70b_action_merge_map.yaml`
- Rule policy: `configs/70c_rule_policy_v1.yaml`
- OPE config: `configs/70e_ope_config_v1.yaml`
- Model registry: `configs/90a_primary_model_registry.json`
"""
    return card


def main():
    print("=" * 72)
    print("95c — Dataset and Model Cards")
    print("=" * 72)
    t0 = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    registry = _load_registry()
    log.info("Loaded 90a registry: %s", "yes" if registry else "empty")

    # Dataset card
    log.info("[1/2] Generating dataset card...")
    dc = generate_dataset_card(registry, ts)
    dc_path = REPORTS_DIR / "95c_dataset_card.md"
    with open(dc_path, "w") as f:
        f.write(dc)
    log.info("  Saved: %s", dc_path)

    # Model card
    log.info("[2/2] Generating model card...")
    mc = generate_model_card(registry, ts)
    mc_path = REPORTS_DIR / "95c_model_card.md"
    with open(mc_path, "w") as f:
        f.write(mc)
    log.info("  Saved: %s", mc_path)

    elapsed = time.time() - t0
    print()
    print("=" * 72)
    print(f"95c COMPLETE. Time: {elapsed:.1f}s")
    print(f"  {dc_path}")
    print(f"  {mc_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
