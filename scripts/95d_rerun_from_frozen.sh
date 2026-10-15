#!/usr/bin/env bash
# 95d — Rerun From Frozen Pipeline
# Generated: 2026-03-14 04:27:52
# Protocol: v3.1
#
# Usage: bash scripts/95d_rerun_from_frozen.sh [--from STEP_ID]
#
# This script reruns the full DTR pipeline from a specified starting point.
# Default: runs everything from 10a.
# Use --from to skip already-completed steps.
#
# Requirements:
#   - MIMIC-IV data linked at data/raw/mimiciv
#   - eICU-CRD data linked at data/raw/eicu
#   - micromamba environment 'hfpef-icu-dtr' active
#
set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)
export PROJECT_ROOT

# Parse --from argument
START_FROM="${2:-10a}"
if [[ "${1:-}" == "--from" ]]; then
    START_FROM="$2"
fi

source env/boot.sh

run_step() {
    local step_id=$1
    local script=$2
    local compute=$3
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "Step $step_id ($compute): $script"
    echo "════════════════════════════════════════════════════════════"
    if [[ ! -f "$script" ]]; then
        echo "[WARN] Script not found, skipping: $script"
        return 0
    fi
    if [[ "$compute" == *G* ]] || [[ "$compute" == C4* ]]; then
        local job_name="s${step_id//[^a-zA-Z0-9_]/_}"
        echo "[GPU] Submit via qsub for step $step_id"
        echo "  qsub -N ${job_name} -q gpua.q -l a100=1 env/sge/submit_g1.sh $script"
        echo "  Waiting for manual GPU job completion..."
        echo "  Press Enter when job completes, or Ctrl-C to abort."
        read -r
    else
        env/run_py.sh "$script"
    fi
}

STARTED=false

# Step 00a-00e: Bootstrap and data survey
if [[ "$STARTED" == true ]] || [[ "$START_FROM" == "00a-00e" ]]; then
    STARTED=true
    run_step "00a-00e" "scripts/00a_survey_data_layout.sh" "C1"
fi

# Step 10a: Build MIMIC ICU stays
if [[ "$STARTED" == true ]] || [[ "$START_FROM" == "10a" ]]; then
    STARTED=true
    run_step "10a" "scripts/10a_build_mimic_stays.py" "C2"
fi

# Step 15a: Unit harmonization
if [[ "$STARTED" == true ]] || [[ "$START_FROM" == "15a" ]]; then
    STARTED=true
    run_step "15a" "scripts/15a_harmonize_units.py" "C1"
fi

# Step 20a: HF subtype labels
if [[ "$STARTED" == true ]] || [[ "$START_FROM" == "20a" ]]; then
    STARTED=true
    run_step "20a" "scripts/20a_build_hf_labels.py" "C2"
fi

# Step 20f: Demographics
if [[ "$STARTED" == true ]] || [[ "$START_FROM" == "20f" ]]; then
    STARTED=true
    run_step "20f" "scripts/20f_demographics.py" "C1"
fi

# Step 25a: Eligibility gate
if [[ "$STARTED" == true ]] || [[ "$START_FROM" == "25a" ]]; then
    STARTED=true
    run_step "25a" "scripts/25a_eligibility_gate.py" "C2"
fi

# Step 25b: Pre-T0 gate
if [[ "$STARTED" == true ]] || [[ "$START_FROM" == "25b" ]]; then
    STARTED=true
    run_step "25b" "scripts/25b_build_preT0_hemodynamic_gate.py" "C2"
fi

# Step 25f: Primary cohort
if [[ "$STARTED" == true ]] || [[ "$START_FROM" == "25f" ]]; then
    STARTED=true
    run_step "25f" "scripts/25f_build_primary_cohort.py" "C2"
fi

# Step 30a: Phase 0.5 sample size
if [[ "$STARTED" == true ]] || [[ "$START_FROM" == "30a" ]]; then
    STARTED=true
    run_step "30a" "scripts/30a_phase05_sample_size.py" "C1"
fi

# Step 30f: Phase 0.5 gatekeeper
if [[ "$STARTED" == true ]] || [[ "$START_FROM" == "30f" ]]; then
    STARTED=true
    run_step "30f" "scripts/30f_phase05_gatekeeper.py" "C1"
fi

# Step 35a: Temporal features
if [[ "$STARTED" == true ]] || [[ "$START_FROM" == "35a" ]]; then
    STARTED=true
    run_step "35a" "scripts/35a_build_temporal_features.py" "C3"
fi

# Step 40a: Design decisions
if [[ "$STARTED" == true ]] || [[ "$START_FROM" == "40a" ]]; then
    STARTED=true
    run_step "40a" "scripts/40a_design_decisions.py" "C1"
fi

# Step 45a: State tensors
if [[ "$STARTED" == true ]] || [[ "$START_FROM" == "45a" ]]; then
    STARTED=true
    run_step "45a" "scripts/45a_build_state_tensors.py" "C3"
fi

# Step 45b: Action discretization
if [[ "$STARTED" == true ]] || [[ "$START_FROM" == "45b" ]]; then
    STARTED=true
    run_step "45b" "scripts/45b_build_actions.py" "C2"
fi

# Step 45c: Normalization params
if [[ "$STARTED" == true ]] || [[ "$START_FROM" == "45c" ]]; then
    STARTED=true
    run_step "45c" "scripts/45c_normalize.py" "C1"
fi

# Step 45d: Outcomes
if [[ "$STARTED" == true ]] || [[ "$START_FROM" == "45d" ]]; then
    STARTED=true
    run_step "45d" "scripts/45d_build_outcomes.py" "C2"
fi

# Step 45f: Freeze dataset
if [[ "$STARTED" == true ]] || [[ "$START_FROM" == "45f" ]]; then
    STARTED=true
    run_step "45f" "scripts/45f_analysis_ready_freeze.py" "C2"
fi

# Step 50a: Train encoder
if [[ "$STARTED" == true ]] || [[ "$START_FROM" == "50a" ]]; then
    STARTED=true
    run_step "50a" "scripts/50a_train_encoder.py" "G1"
fi

# Step 55a: Encoder evaluation
if [[ "$STARTED" == true ]] || [[ "$START_FROM" == "55a" ]]; then
    STARTED=true
    run_step "55a" "scripts/55a_encoder_eval.py" "C2"
fi

# Step 60a: Baseline features
if [[ "$STARTED" == true ]] || [[ "$START_FROM" == "60a" ]]; then
    STARTED=true
    run_step "60a" "scripts/60a_extract_baseline_features.py" "C2"
fi

# Step 60d: Rule-based phenotype
if [[ "$STARTED" == true ]] || [[ "$START_FROM" == "60d" ]]; then
    STARTED=true
    run_step "60d" "scripts/60d_fallback_rule_phenotype.py" "C2"
fi

# Step 65a: Trajectory clustering
if [[ "$STARTED" == true ]] || [[ "$START_FROM" == "65a" ]]; then
    STARTED=true
    run_step "65a" "scripts/65a_trajectory_clustering.py" "G1"
fi

# Step 70a: Policy objects
if [[ "$STARTED" == true ]] || [[ "$START_FROM" == "70a" ]]; then
    STARTED=true
    run_step "70a" "scripts/70a_build_policy_objects.py" "C2"
fi

# Step 70b: Action merge map
if [[ "$STARTED" == true ]] || [[ "$START_FROM" == "70b" ]]; then
    STARTED=true
    run_step "70b" "scripts/70b_action_merge_map.py" "C1"
fi

# Step 70c: Rule policy
if [[ "$STARTED" == true ]] || [[ "$START_FROM" == "70c" ]]; then
    STARTED=true
    run_step "70c" "scripts/70c_rule_policy.py" "C1"
fi

# Step 70e: OPE config
if [[ "$STARTED" == true ]] || [[ "$START_FROM" == "70e" ]]; then
    STARTED=true
    run_step "70e" "scripts/70e_ope_config.py" "C1"
fi

# Step 75a: MSM-IPTW OPE
if [[ "$STARTED" == true ]] || [[ "$START_FROM" == "75a" ]]; then
    STARTED=true
    run_step "75a" "scripts/75a_dynamic_msm_iptw.py" "C3"
fi

# Step 80a: GNet training + OPE
if [[ "$STARTED" == true ]] || [[ "$START_FROM" == "80a" ]]; then
    STARTED=true
    run_step "80a" "scripts/80a_gnet.py" "G1"
fi

# Step 80b: ACTIN training + OPE
if [[ "$STARTED" == true ]] || [[ "$START_FROM" == "80b" ]]; then
    STARTED=true
    run_step "80b" "scripts/80b_actin.py" "G1"
fi

# Step 80c: G-Transformer training + OPE
if [[ "$STARTED" == true ]] || [[ "$START_FROM" == "80c" ]]; then
    STARTED=true
    run_step "80c" "scripts/80c_gtransformer.py" "G1"
fi

# Step 80d: CT-Legacy training + OPE
if [[ "$STARTED" == true ]] || [[ "$START_FROM" == "80d" ]]; then
    STARTED=true
    run_step "80d" "scripts/80d_ct_legacy.py" "G1"
fi

# Step 80e: Mamba-CDSP training + OPE
if [[ "$STARTED" == true ]] || [[ "$START_FROM" == "80e" ]]; then
    STARTED=true
    run_step "80e" "scripts/80e_mamba_cdsp.py" "G1"
fi

# Step 80g: HFpEF x shock
if [[ "$STARTED" == true ]] || [[ "$START_FROM" == "80g" ]]; then
    STARTED=true
    run_step "80g" "scripts/80g_hf_subtype_x_shock_interaction.py" "C3"
fi

# Step 85a: EDQ sensitivity
if [[ "$STARTED" == true ]] || [[ "$START_FROM" == "85a" ]]; then
    STARTED=true
    run_step "85a" "scripts/85a_edq_sensitivity.py" "C3"
fi

# Step 85b: Sequential deconfounding
if [[ "$STARTED" == true ]] || [[ "$START_FROM" == "85b" ]]; then
    STARTED=true
    run_step "85b" "scripts/85b_seqdeconf_sensitivity.py" "C3"
fi

# Step 85d: Positivity grid
if [[ "$STARTED" == true ]] || [[ "$START_FROM" == "85d" ]]; then
    STARTED=true
    run_step "85d" "scripts/85d_positivity_grid.py" "C3"
fi

# Step 85e: Negative control
if [[ "$STARTED" == true ]] || [[ "$START_FROM" == "85e" ]]; then
    STARTED=true
    run_step "85e" "scripts/85e_negative_control.py" "C3"
fi

# Step 85f: E-value
if [[ "$STARTED" == true ]] || [[ "$START_FROM" == "85f" ]]; then
    STARTED=true
    run_step "85f" "scripts/85f_evalue_sensitivity.py" "C2"
fi

# Step 85g: Monitoring sensitivity
if [[ "$STARTED" == true ]] || [[ "$START_FROM" == "85g" ]]; then
    STARTED=true
    run_step "85g" "scripts/85g_monitoring_sensitivity.py" "C3"
fi

# Step 85h: RRT sensitivity
if [[ "$STARTED" == true ]] || [[ "$START_FROM" == "85h" ]]; then
    STARTED=true
    run_step "85h" "scripts/85h_rrt_sensitivity.py" "C3"
fi

# Step 85i: Horizon sensitivity
if [[ "$STARTED" == true ]] || [[ "$START_FROM" == "85i" ]]; then
    STARTED=true
    run_step "85i" "scripts/85i_horizon_sensitivity.py" "G1"
fi

# Step 85j: Informative missingness
if [[ "$STARTED" == true ]] || [[ "$START_FROM" == "85j" ]]; then
    STARTED=true
    run_step "85j" "scripts/85j_missingness_sensitivity.py" "C3"
fi

# Step 90a: Freeze models
if [[ "$STARTED" == true ]] || [[ "$START_FROM" == "90a" ]]; then
    STARTED=true
    run_step "90a" "scripts/90a_freeze_primary_models.py" "C1"
fi

# Step 90b: eICU validation
if [[ "$STARTED" == true ]] || [[ "$START_FROM" == "90b" ]]; then
    STARTED=true
    run_step "90b" "scripts/90b_05_evaluate_eicu_frozen_models.py" "C3+G1"
fi

# Step 90d: Transportability summary
if [[ "$STARTED" == true ]] || [[ "$START_FROM" == "90d" ]]; then
    STARTED=true
    run_step "90d" "scripts/90d_transportability_summary.py" "C1"
fi

# Step 90e: Consistency report
if [[ "$STARTED" == true ]] || [[ "$START_FROM" == "90e" ]]; then
    STARTED=true
    run_step "90e" "scripts/90e_cross_dataset_consistency_report.py" "C2"
fi

# Step 95a: Master results registry
if [[ "$STARTED" == true ]] || [[ "$START_FROM" == "95a" ]]; then
    STARTED=true
    run_step "95a" "scripts/95a_master_results_registry.py" "C1"
fi

# Step 95b: Figures and tables
if [[ "$STARTED" == true ]] || [[ "$START_FROM" == "95b" ]]; then
    STARTED=true
    run_step "95b" "scripts/95b_generate_figures_tables.py" "C1"
fi

# Step 95c: Dataset/model cards
if [[ "$STARTED" == true ]] || [[ "$START_FROM" == "95c" ]]; then
    STARTED=true
    run_step "95c" "scripts/95c_dataset_model_cards.py" "C1"
fi

# Step 95d: Rerun pipeline generator
if [[ "$STARTED" == true ]] || [[ "$START_FROM" == "95d" ]]; then
    STARTED=true
    run_step "95d" "scripts/95d_repro_rerun_pipeline.py" "C1"
fi

# Step 95e: Release archive bundle
if [[ "$STARTED" == true ]] || [[ "$START_FROM" == "95e" ]]; then
    STARTED=true
    run_step "95e" "scripts/95e_archive_release_bundle.py" "C1"
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "Pipeline complete."
echo "════════════════════════════════════════════════════════════"
