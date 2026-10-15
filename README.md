# HFpEF ICU dynamic treatment rule: analysis code

Analysis code for the study *A two-hourly, perfusion-first haemodynamic treatment
strategy for critically ill patients with heart failure with preserved ejection
fraction: a target trial emulation with external evaluation*.

The study emulates a target trial in MIMIC-IV (development and internal test set)
and evaluates the same frozen models in the multicentre eICU Collaborative Research
Database. A priority-ordered fluid–vasopressor–diuretic rule, reassessed every 2 h
for 24 h after ICU admission + 6 h, is compared with usual care by g-computation
with a deep causal sequence model (G-Transformer), alongside an early haemodynamic
fragility index and early mortality-risk triage models.

## Data availability

**No patient-level data are included in this repository.** MIMIC-IV (v3.1) and
eICU-CRD (v2.0) are available from PhysioNet to credentialed researchers who
complete the required training and sign the PhysioNet Credentialed Health Data Use
Agreement:

- MIMIC-IV: https://doi.org/10.13026/kpb9-mt58
- eICU-CRD: https://doi.org/10.13026/C2WM1R

All intermediate tables, model checkpoints, predictions and result files are written
to local directories (`data/`, `reports/`, `tables/`, `figures/`, `models/`, `logs/`)
that are excluded from version control.

## Repository layout

| Path | Contents |
| --- | --- |
| `env/` | Environment definition (`environment.base.yml`), bootstrap scripts and Grid Engine job templates |
| `configs/` | Analysis configuration: variable map, action space, normalisation parameters, rule specification, rewards, off-policy evaluation settings, model registries |
| `src/hfpef_dtr/` | Python package: configuration loader, datasets, sequence models (G-Transformer, G-Net, ACTIN, Causal Transformer, Mamba) and off-policy evaluation |
| `scripts/` | Step-wise pipeline; the numeric prefix gives the execution order |

Pipeline stages (script prefixes):

| Prefix | Stage |
| --- | --- |
| `00`–`05` | Data layout checks and raw-table inventory |
| `10` | Extraction of ICU stays, admission context, vital signs, laboratory values, medications, fluids and procedures (MIMIC-IV) |
| `20`–`25` | Heart-failure labels, exclusions, index stay, time zero (ICU admission + 6 h) and cohort freeze |
| `29`–`35` | Action-space candidates, feasibility checks and comparator cohorts |
| `40`–`45` | 2-hourly decision grid, state tensors, encoded actions, outcomes and the frozen modelling dataset |
| `50`–`65` | Representation learning and phenotype construction |
| `70` | Policy objects: behaviour policy, the dynamic treatment rule (`70c_define_rule_policy_v1.py`, `configs/70c_rule_policy_v1.yaml`), rewards, evaluation settings and trimming grid |
| `75` | Inverse-probability-weighted dynamic marginal structural model |
| `80` | Sequence models; `80c_train_gtransformer.py` is the primary analysis |
| `85` | Sensitivity analyses (early discharge, sequential deconfounder, positivity, negative controls, E-values, monitoring, renal replacement therapy, horizon, missingness) |
| `90` | Model freeze and external evaluation in eICU-CRD |
| `95`–`96` | Results registry, reproducibility rerun and exploratory trajectory analyses |
| `P1`–`P18` | Supplementary analyses (guideline-derived strategies, heterogeneity, timing, fragility index, trial sizing and others) |
| `manuscript_*` | Builders for tables and figure layouts from aggregate outputs |

## Environment

```bash
cp env/env.paths.example env/env.paths      # set PROJECT_ROOT and data locations
micromamba create -y -f env/environment.base.yml
source env/boot.sh
env/run_py.sh scripts/<step_script>.py
```

GPU and large CPU steps were run through the Grid Engine templates in `env/sge/`
(see `env/sge/README.md`); submit them from the project root.

## Licence

MIT; see `LICENSE`. Citation metadata are in `CITATION.cff`.
