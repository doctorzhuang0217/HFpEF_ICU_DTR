#!/usr/bin/env python3
"""95d — Reproducibility Rerun Pipeline Generator

Protocol: v3.1, Step 95d
Compute: C1

Purpose
-------
Generate a rerun script (rerun_from_frozen.sh) that reproduces the study
from frozen datasets / frozen model stages. Also generates a dependency
DAG manifest for understanding execution order.

Outputs
-------
- scripts/95d_rerun_from_frozen.sh
- reports/95d_pipeline_dag.txt
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────
ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
SCRIPTS_DIR = ROOT / "scripts"
REPORTS_DIR = ROOT / "reports"

for d in [SCRIPTS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("95d")


# ── Pipeline step definitions ─────────────────────────────────────────
# Each tuple: (step_id, script, compute_tier, depends_on, description)
PIPELINE_STEPS = [
    # Phase 0: Bootstrap
    ("00a-00e", "scripts/00a_survey_data_layout.sh", "C1", [], "Bootstrap and data survey"),

    # Phase 1: Cohort construction
    ("10a", "scripts/10a_build_mimic_stays.py", "C2", [], "Build MIMIC ICU stays"),
    ("15a", "scripts/15a_harmonize_units.py", "C1", ["10a"], "Unit harmonization"),
    ("20a", "scripts/20a_build_hf_labels.py", "C2", ["10a"], "HF subtype labels"),
    ("20f", "scripts/20f_demographics.py", "C1", ["10a"], "Demographics"),
    ("25a", "scripts/25a_eligibility_gate.py", "C2", ["20a"], "Eligibility gate"),
    ("25b", "scripts/25b_build_preT0_hemodynamic_gate.py", "C2", ["25a"], "Pre-T0 gate"),
    ("25f", "scripts/25f_build_primary_cohort.py", "C2", ["25b", "20f"], "Primary cohort"),
    ("30a", "scripts/30a_phase05_sample_size.py", "C1", ["25f"], "Phase 0.5 sample size"),
    ("30f", "scripts/30f_phase05_gatekeeper.py", "C1", ["30a"], "Phase 0.5 gatekeeper"),

    # Phase 2: Feature engineering
    ("35a", "scripts/35a_build_temporal_features.py", "C3", ["25f"], "Temporal features"),
    ("40a", "scripts/40a_design_decisions.py", "C1", ["35a"], "Design decisions"),
    ("45a", "scripts/45a_build_state_tensors.py", "C3", ["35a"], "State tensors"),
    ("45b", "scripts/45b_build_actions.py", "C2", ["45a"], "Action discretization"),
    ("45c", "scripts/45c_normalize.py", "C1", ["45a"], "Normalization params"),
    ("45d", "scripts/45d_build_outcomes.py", "C2", ["45a"], "Outcomes"),
    ("45f", "scripts/45f_analysis_ready_freeze.py", "C2", ["45a", "45b", "45c", "45d"], "Freeze dataset"),

    # Phase 3: Representation learning
    ("50a", "scripts/50a_train_encoder.py", "G1", ["45f"], "Train encoder"),
    ("55a", "scripts/55a_encoder_eval.py", "C2", ["50a"], "Encoder evaluation"),
    ("60a", "scripts/60a_extract_baseline_features.py", "C2", ["55a"], "Baseline features"),
    ("60d", "scripts/60d_fallback_rule_phenotype.py", "C2", ["60a"], "Rule-based phenotype"),
    ("65a", "scripts/65a_trajectory_clustering.py", "G1", ["45f", "60a"], "Trajectory clustering"),

    # Phase 4: Policy + OPE
    ("70a", "scripts/70a_build_policy_objects.py", "C2", ["60d", "65a"], "Policy objects"),
    ("70b", "scripts/70b_action_merge_map.py", "C1", ["70a"], "Action merge map"),
    ("70c", "scripts/70c_rule_policy.py", "C1", ["70b"], "Rule policy"),
    ("70e", "scripts/70e_ope_config.py", "C1", ["70c"], "OPE config"),
    ("75a", "scripts/75a_dynamic_msm_iptw.py", "C3", ["70e", "45f"], "MSM-IPTW OPE"),

    # Phase 5: Step C models
    ("80a", "scripts/80a_gnet.py", "G1", ["75a"], "GNet training + OPE"),
    ("80b", "scripts/80b_actin.py", "G1", ["75a"], "ACTIN training + OPE"),
    ("80c", "scripts/80c_gtransformer.py", "G1", ["75a"], "G-Transformer training + OPE"),
    ("80d", "scripts/80d_ct_legacy.py", "G1", ["75a"], "CT-Legacy training + OPE"),
    ("80e", "scripts/80e_mamba_cdsp.py", "G1", ["75a"], "Mamba-CDSP training + OPE"),
    ("80g", "scripts/80g_hf_subtype_x_shock_interaction.py", "C3", ["80a", "80b", "80c"], "HFpEF x shock"),

    # Phase 6: Sensitivity
    ("85a", "scripts/85a_edq_sensitivity.py", "C3", ["80c"], "EDQ sensitivity"),
    ("85b", "scripts/85b_seqdeconf_sensitivity.py", "C3", ["80c"], "Sequential deconfounding"),
    ("85d", "scripts/85d_positivity_grid.py", "C3", ["80c"], "Positivity grid"),
    ("85e", "scripts/85e_negative_control.py", "C3", ["80c"], "Negative control"),
    ("85f", "scripts/85f_evalue_sensitivity.py", "C2", ["80c"], "E-value"),
    ("85g", "scripts/85g_monitoring_sensitivity.py", "C3", ["80c"], "Monitoring sensitivity"),
    ("85h", "scripts/85h_rrt_sensitivity.py", "C3", ["80c"], "RRT sensitivity"),
    ("85i", "scripts/85i_horizon_sensitivity.py", "G1", ["80c"], "Horizon sensitivity"),
    ("85j", "scripts/85j_missingness_sensitivity.py", "C3", ["80c"], "Informative missingness"),

    # Phase 7: External validation
    ("90a", "scripts/90a_freeze_primary_models.py", "C1", ["80a", "80b", "80c", "80d", "80e"], "Freeze models"),
    ("90b", "scripts/90b_05_evaluate_eicu_frozen_models.py", "C3+G1", ["90a"], "eICU validation"),
    ("90d", "scripts/90d_transportability_summary.py", "C1", ["90b"], "Transportability summary"),
    ("90e", "scripts/90e_cross_dataset_consistency_report.py", "C2", ["90d", "80g"], "Consistency report"),

    # Phase 8: Packaging
    ("95a", "scripts/95a_master_results_registry.py", "C1", ["90e"], "Master results registry"),
    ("95b", "scripts/95b_generate_figures_tables.py", "C1", ["95a"], "Figures and tables"),
    ("95c", "scripts/95c_dataset_model_cards.py", "C1", ["95a"], "Dataset/model cards"),
    ("95d", "scripts/95d_repro_rerun_pipeline.py", "C1", ["95c"], "Rerun pipeline generator"),
    ("95e", "scripts/95e_archive_release_bundle.py", "C1", ["95d"], "Release archive bundle"),
]


def generate_rerun_script(ts: str) -> str:
    """Generate a bash rerun script."""
    lines = [
        "#!/usr/bin/env bash",
        f"# 95d — Rerun From Frozen Pipeline",
        f"# Generated: {ts}",
        f"# Protocol: v3.1",
        "#",
        "# Usage: bash scripts/95d_rerun_from_frozen.sh [--from STEP_ID]",
        "#",
        "# This script reruns the full DTR pipeline from a specified starting point.",
        "# Default: runs everything from 10a.",
        "# Use --from to skip already-completed steps.",
        "#",
        "# Requirements:",
        "#   - MIMIC-IV data linked at data/raw/mimiciv",
        "#   - eICU-CRD data linked at data/raw/eicu",
        "#   - micromamba environment 'hfpef-icu-dtr' active",
        "#",
        "set -euo pipefail",
        "",
        "cd \"$(dirname \"$0\")/..\"",
        "PROJECT_ROOT=$(pwd)",
        "export PROJECT_ROOT",
        "",
        "# Parse --from argument",
        "START_FROM=\"${2:-10a}\"",
        "if [[ \"${1:-}\" == \"--from\" ]]; then",
        "    START_FROM=\"$2\"",
        "fi",
        "",
        "source env/boot.sh",
        "",
        "run_step() {",
        "    local step_id=$1",
        "    local script=$2",
        "    local compute=$3",
        "    echo \"\"",
        "    echo \"════════════════════════════════════════════════════════════\"",
        "    echo \"Step $step_id ($compute): $script\"",
        "    echo \"════════════════════════════════════════════════════════════\"",
        "    if [[ ! -f \"$script\" ]]; then",
        "        echo \"[WARN] Script not found, skipping: $script\"",
        "        return 0",
        "    fi",
        "    if [[ \"$compute\" == *G* ]] || [[ \"$compute\" == C4* ]]; then",
        "        local job_name=\"s${step_id//[^a-zA-Z0-9_]/_}\"",
        "        echo \"[GPU] Submit via qsub for step $step_id\"",
        "        echo \"  qsub -N ${job_name} -q gpua.q -l a100=1 env/sge/submit_g1.sh $script\"",
        "        echo \"  Waiting for manual GPU job completion...\"",
        "        echo \"  Press Enter when job completes, or Ctrl-C to abort.\"",
        "        read -r",
        "    else",
        "        env/run_py.sh \"$script\"",
        "    fi",
        "}",
        "",
        "STARTED=false",
        "",
    ]

    for step_id, script, compute, deps, desc in PIPELINE_STEPS:
        lines.append(f"# Step {step_id}: {desc}")
        lines.append(f"if [[ \"$STARTED\" == true ]] || [[ \"$START_FROM\" == \"{step_id}\" ]]; then")
        lines.append(f"    STARTED=true")
        lines.append(f"    run_step \"{step_id}\" \"{script}\" \"{compute}\"")
        lines.append(f"fi")
        lines.append("")

    lines.extend([
        "echo \"\"",
        "echo \"════════════════════════════════════════════════════════════\"",
        "echo \"Pipeline complete.\"",
        "echo \"════════════════════════════════════════════════════════════\"",
    ])

    return "\n".join(lines) + "\n"


def generate_dag(ts: str) -> str:
    """Generate a text-based DAG visualization."""
    lines = [
        "=" * 72,
        "95d — Pipeline Dependency DAG",
        "=" * 72,
        f"Generated: {ts}",
        "",
        "Format: step_id [compute] → depends_on",
        "",
    ]

    # Group by phase
    phases = {}
    for step_id, script, compute, deps, desc in PIPELINE_STEPS:
        phase_num = step_id[:2] if step_id[:2].isdigit() else step_id[:1]
        if phase_num not in phases:
            phases[phase_num] = []
        phases[phase_num].append((step_id, compute, deps, desc))

    phase_names = {
        "00": "Bootstrap",
        "10": "Cohort Construction",
        "15": "Unit Harmonization",
        "20": "Labels & Demographics",
        "25": "Eligibility & Cohort",
        "30": "Phase 0.5 Gating",
        "35": "Temporal Features",
        "40": "Design Decisions",
        "45": "State/Action/Outcome",
        "50": "Encoder Training",
        "55": "Encoder Evaluation",
        "60": "Phenotyping",
        "65": "Trajectory Clustering",
        "70": "Policy & OPE Config",
        "75": "Primary OPE",
        "80": "Step C Models",
        "85": "Sensitivity Analyses",
        "90": "External Validation",
        "95": "Final Packaging",
    }

    for phase_num in sorted(phases.keys()):
        pname = phase_names.get(phase_num, f"Phase {phase_num}")
        lines.append(f"── {pname} ──")
        for step_id, compute, deps, desc in phases[phase_num]:
            dep_str = ", ".join(deps) if deps else "(none)"
            lines.append(f"  {step_id:<8s} [{compute:<6s}] {desc:<40s} ← {dep_str}")
        lines.append("")

    # Critical path summary
    lines.append("─" * 72)
    lines.append("CRITICAL PATH (longest dependency chain):")
    lines.append("  10a → 20a → 25a → 25b → 25f → 35a → 45a → 45f")
    lines.append("  → 50a → 60a → 65a → 70a → 75a → 80c → 85* → 90a → 90b → 90e → 95a → 95b → 95c → 95d → 95e")
    lines.append("")
    lines.append("GPU-REQUIRING STEPS: 50a, 65a, 80a, 80b, 80c, 80d, 80e, 85i")

    return "\n".join(lines) + "\n"


def main():
    print("=" * 72)
    print("95d — Reproducibility Rerun Pipeline")
    print("=" * 72)
    t0 = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Generate rerun script
    log.info("[1/2] Generating rerun script...")
    rerun_script = generate_rerun_script(ts)
    rerun_path = SCRIPTS_DIR / "95d_rerun_from_frozen.sh"
    with open(rerun_path, "w") as f:
        f.write(rerun_script)
    os.chmod(rerun_path, 0o755)
    log.info("  Saved: %s", rerun_path)

    # Generate DAG
    log.info("[2/2] Generating pipeline DAG...")
    dag = generate_dag(ts)
    dag_path = REPORTS_DIR / "95d_pipeline_dag.txt"
    with open(dag_path, "w") as f:
        f.write(dag)
    log.info("  Saved: %s", dag_path)

    elapsed = time.time() - t0
    print()
    print("=" * 72)
    print(f"95d COMPLETE. Time: {elapsed:.1f}s")
    print(f"  {rerun_path}")
    print(f"  {dag_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
