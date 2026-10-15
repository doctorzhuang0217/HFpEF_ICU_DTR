# SGE/UGE Job Submission Templates

Compute-tier-aligned templates for a Univa Grid Engine (UGE) cluster.

## Compute Tiers

| Template | Tier | Cores | RAM/slot | Wall time | GPU | Use case |
|----------|------|-------|----------|-----------|-----|----------|
| `submit_c1.sh` | C1 | 4 | 4G | 1h | — | Configs, aggregation, reports |
| `submit_c2.sh` | C2 | 8 | 8G | 4h | — | Extraction, filtering, comparisons |
| `submit_c3.sh` | C3 | 16 | 8G | 12h | — | Large tensors, clustering, grid builders |
| `submit_c4.sh` | C4 | 32 | 8G | 24h | — | HiRID extraction, massive sparse matrices |
| `submit_g1.sh` | G1 | 8 | 8G | 24h | 1× | Baseline encoders, probe tasks |
| `submit_g2.sh` | G2 | 16 | 8G | 48h | 2× | G-Transformer, ACTIN, Mamba-CDSP |

## Usage

```bash
# Single step:
qsub -N step35a env/sge/submit_c2.sh scripts/35a_build_comparator_cohorts.py

# Chain (35a → 35b using hold_jid):
JID=$(qsub -terse -N step35a env/sge/submit_c2.sh scripts/35a_build_comparator_cohorts.py)
qsub -hold_jid $JID -N step35b env/sge/submit_c2.sh scripts/35b_build_syndrome_matched_nonhf.py

# Override resources:
qsub -N step35d -pe def_slot 16 env/sge/submit_c2.sh scripts/35d_compare_treatment_patterns_6h_24h.py

# Check job status:
qstat -u $(whoami)
```

## Cluster-Specific Configuration

- **PE name**: `def_slot` (confirmed)
- **GPU resource name**: `m_gpu` (not `gpu`)
- **GPU queues**:
  - `gpuv.q` — V100 (resource: `m_gpu`)
  - `gpua.q` — A100 (resource: `m_gpu`, default in G1/G2 templates)
  - `gpuh.q` — H100 (resource: **`h100`**, NOT `m_gpu`)
- **s_vmem / h_vmem**: Per-slot soft/hard memory limits; total = slots × s_vmem
- **Logs**: Written to `logs/<jobname>.<jobid>.log`

### GPU Queue Override Examples

```bash
# A100 (default) — no override needed, uses submit_g1.sh as-is:
qsub -N step50a env/sge/submit_g1.sh scripts/50a_train_encoder_baseline.py

# H100 — MUST request forced resource `h100=N` (not m_gpu):
qsub -N step50a -q gpuh.q -l h100=1 env/sge/submit_g1.sh scripts/50a_train_encoder_baseline.py

# V100:
qsub -N step50a -q gpuv.q env/sge/submit_g1.sh scripts/50a_train_encoder_baseline.py
```

> **gpuh.q gotcha**: `h100` is a **forced resource** on gpuh.q hosts. Jobs that
> do not explicitly request `-l h100=N` will stay in `qw` state indefinitely.
> The embedded `#$ -l m_gpu=1` in submit templates is NOT sufficient for gpuh.q.
